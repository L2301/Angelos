"""Native diffusion contract, genuine optimization, provenance and replay."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch.nn import functional

from angelos_ml.diffusion import ARCHITECTURE, DiffusionStudent, cosine_schedule, render_cpu, sample_ddim, seeded_noise, validate_schedule
from angelos_ml.graph import canonical_json, graph_hash
from angelos_ml.models import model_metadata
from angelos_ml.providers import node_seed
from angelos_ml.teacher import apply_brand, pixel_coordinates
from angelos_ml.train_diffusion import generate_procedural_fixture, load_dataset, resolve_device, shuffled_conditioning_bank, train_diffusion, verify_canonical_graph, write_artifacts
from tests.test_ml import example_graph


ROOT = Path(__file__).resolve().parents[2]


def write_graph(directory, graph):
    normalized = copy.deepcopy(graph)
    normalized["nodes"] = sorted(graph["nodes"], key=lambda node: node["id"])
    path = Path(directory) / "canonical.json"
    path.write_text(canonical_json(normalized), encoding="utf-8")
    return path


class DiffusionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_noise_matches_javascript_lcg_and_box_muller(self):
        expected = [2.6681275367736816, 0.7641785740852356, -1.326427936553955, -2.180623769760132, -0.9322425127029419, 0.09389302879571915, 0.09695857018232346, -0.9650884866714478]
        self.assertEqual(seeded_noise().flatten()[:8].tolist(), expected)
        self.assertEqual(seeded_noise(0).dtype, torch.float32)
        self.assertTrue(torch.equal(seeded_noise(2026), seeded_noise(2026)))
        self.assertFalse(torch.equal(seeded_noise(2026), seeded_noise(2027)))
        self.assertTrue(torch.isfinite(seeded_noise(2 ** 32 - 1)).all())
        self.assertEqual(seeded_noise(2026, (3,)).tolist(), expected[:3])
        for seed in (-1, 2 ** 32, True, 1.2):
            with self.assertRaises(ValueError):
                seeded_noise(seed)

    def test_schedule_and_float32_contract(self):
        schedule = cosine_schedule()
        self.assertEqual(validate_schedule(schedule), schedule)
        self.assertGreater(schedule["alphaCumprod"][-1], 0)
        self.assertLess(schedule["alphaCumprod"][-1], 1e-6)
        self.assertEqual(schedule["timesteps"], [63, 59, 55, 50, 46, 42, 38, 34, 29, 25, 21, 17, 13, 8, 4, 0])
        for field, value in (("predictionType", "x0"), ("alphaCumprod", [0.5] * 64), ("timesteps", list(range(16)))):
            with self.assertRaises(ValueError):
                validate_schedule({**schedule, field: value})

    def test_five_convolutions_input_planes_skip_and_export_layout(self):
        torch.manual_seed(123)
        model = DiffusionStudent()
        upl = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
        noise = seeded_noise(99)
        epsilon, nodes = model(noise, upl, torch.tensor([31]), return_nodes=True)
        self.assertEqual(list(epsilon.shape), [1, 3, 32, 32])
        torch.testing.assert_close(nodes["input"][:, :3], noise)
        torch.testing.assert_close(nodes["input"][0, 3:7, 0, 0], upl[0])
        self.assertAlmostEqual(nodes["input"][0, 7, 0, 0].item(), 31 / 63)
        self.assertEqual(nodes["input"][0, 8, 0, 0].item(), 0.5 / 32)
        self.assertEqual(nodes["input"][0, 9, 31, 0].item(), 31.5 / 32)
        torch.testing.assert_close(nodes["skip"][:, :32], nodes["conv2"].repeat_interleave(2, 2).repeat_interleave(2, 3))
        torch.testing.assert_close(nodes["skip"][:, 32:], nodes["conv0"])
        layers = model.export_layers()
        self.assertEqual([layer["input"] for layer in layers], [10, 16, 32, 48, 16])
        self.assertEqual([layer["stride"] for layer in layers], [1, 2, 1, 1, 1])
        self.assertEqual(layers[-1]["activation"], "linear")
        source = nodes["input"].detach().numpy()
        padded = np.pad(source, ((0, 0), (0, 0), (1, 1), (1, 1)))
        patches = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
        weights = np.asarray(layers[0]["weights"], np.float32).reshape(16, 10, 3, 3)
        expected = np.einsum("bihwkl,oikl->bohw", patches, weights) + np.asarray(layers[0]["bias"], np.float32)[None, :, None, None]
        np.testing.assert_allclose(nodes["conv0"].detach().numpy(), np.maximum(expected, 0), atol=1e-6, rtol=1e-5)
        epsilon.square().mean().backward()
        self.assertTrue(all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        with torch.no_grad():
            model.layers[0].weight[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model.export_layers()

    def test_ddim_step_formula_clipping_and_no_extra_noise(self):
        torch.manual_seed(73)
        model = DiffusionStudent()
        profile = [0.2, 0.3, 0.4, 0.5]
        result, checks = sample_ddim(model, profile, 0, return_trace=True)
        schedule = cosine_schedule()
        noisy = np.array(checks["initialNoise"], dtype=np.float32)
        for index, step in enumerate(checks["steps"]):
            alpha = np.float32(schedule["alphaCumprod"][step["timestep"]])
            epsilon = np.array(step["epsilon"], dtype=np.float32)
            clean = np.clip((noisy - np.sqrt(np.float32(1) - alpha) * epsilon) / np.sqrt(alpha), -1, 1)
            next_alpha = np.float32(schedule["alphaCumprod"][schedule["timesteps"][index + 1]]) if index < 15 else np.float32(1)
            noisy = np.sqrt(next_alpha) * clean + np.sqrt(np.float32(1) - next_alpha) * epsilon
            np.testing.assert_allclose(step["x0"], clean, atol=1e-5)
            np.testing.assert_allclose(step["nextSample"], noisy, atol=1e-5)
        np.testing.assert_allclose(result.flatten()[:16], (noisy + 1) / 2, atol=1e-5)
        self.assertTrue(((result >= 0) & (result <= 1)).all())
        self.assertTrue(torch.equal(result, sample_ddim(model, profile, 0)))

    def test_explicit_device_unavailability(self):
        self.assertEqual(str(resolve_device("cpu")), "cpu")
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                resolve_device("cuda")
        with mock.patch("torch.backends.mps.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                resolve_device("mps")


class DiffusionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="angelos-diffusion-test-")
        cls.directory = Path(cls.temporary.name)
        cls.graph = example_graph()
        cls.identity = graph_hash(cls.graph)
        cls.canonical = write_graph(cls.directory, cls.graph)
        cls.dataset = cls.directory / "fixture.npz"
        generate_procedural_fixture(cls.dataset, cls.graph, cls.identity, cls.canonical, train_count=8, heldout_count=4)
        cls.artifact, cls.report, cls.fixture = train_diffusion(
            cls.graph, cls.identity, cls.dataset, cls.canonical, steps=100,
            batch_size=8, threads=1, max_seconds=90, procedural_fixture=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_actual_noise_training_and_honest_provenance(self):
        metrics = self.artifact["metrics"]
        self.assertEqual(self.artifact["format"], "angelos.diffusion.v1")
        self.assertEqual(self.artifact["architecture"], ARCHITECTURE)
        self.assertTrue(metrics["improvedOverUntrained"])
        self.assertLess(metrics["heldoutMse"], metrics["baselineMse"] * 0.8)
        self.assertIn("noise", metrics["lossSemantics"])
        self.assertGreater(metrics["parameterDeltaL2"], 0)
        self.assertAlmostEqual(metrics["conditioningNoiseMseGain"], metrics["heldoutShuffledUplMse"] - metrics["heldoutMse"])
        self.assertEqual(metrics["improvedWithCorrectUpl"], metrics["conditioningNoiseMseGain"] > 0)
        self.assertEqual(self.artifact["training"]["steps"], 100)
        self.assertTrue(self.artifact["training"]["proceduralFixture"])
        self.assertFalse(self.artifact["training"]["pretrained"])
        self.assertFalse(self.artifact["training"]["promoted"])
        self.assertEqual(self.artifact["training"]["source"], "procedural-fixture-explicit")
        self.assertFalse(set(map(tuple, self.report["trainProfiles"])) & set(map(tuple, self.report["heldoutProfiles"])))

    def test_genuine_seed_diversity_and_conditioning_in_trained_model(self):
        model = DiffusionStudent.from_artifact(self.artifact)
        profile = self.fixture["rasterCases"][0]["upl"]
        first = sample_ddim(model, profile, 2026, self.artifact["schedule"])
        second = sample_ddim(model, profile, 2027, self.artifact["schedule"])
        changed = sample_ddim(model, [1 - value for value in profile], 2026, self.artifact["schedule"])
        self.assertTrue(torch.equal(first, sample_ddim(model, profile, 2026, self.artifact["schedule"])))
        self.assertGreater(functional.mse_loss(first, second).item(), 1e-8)
        self.assertGreater(functional.mse_loss(first, changed).item(), 1e-8)
        self.assertGreater(self.artifact["metrics"]["seedDiversityMse"], 1e-8)
        self.assertGreater(self.artifact["metrics"]["uplChangeMse"], 1e-10)

    def test_export_roundtrip_raster_parity_and_exact_requested_plate(self):
        self.assertEqual(len(self.fixture["rasterCases"]), 2)
        with np.load(self.dataset) as arrays:
            for case in self.fixture["rasterCases"]:
                self.assertEqual((case["width"], case["height"]), (32, 32))
                raw = render_cpu(self.artifact, case["upl"], composite=False, seed=case["seed"])
                delivered = render_cpu(self.artifact, case["upl"], seed=case["seed"])
                np.testing.assert_array_equal(raw.flatten().numpy(), case["rawStudentRgba"])
                np.testing.assert_array_equal(delivered.flatten().numpy(), case["studentRgba"])
                expected = arrays["images"][case["datasetIndex"]].transpose(1, 2, 0)
                np.testing.assert_array_equal(np.array(case["teacherRgba"]).reshape(32, 32, 4)[:, :, :3], expected)
                self.assertEqual(len(case["checks"]["steps"]), 16)
                self.assertEqual(case["checks"]["nodes"]["skip"]["shape"], [1, 48, 32, 32])
        self.assertEqual(self.artifact["brand"]["mode"], "composite")
        self.assertGreater(self.artifact["metrics"]["brandMaxError"], self.artifact["metrics"]["brandTolerance"])
        self.assertEqual(self.artifact["metrics"]["compositeBrandMaxError"], 0)
        case = self.fixture["rasterCases"][0]
        for width, height in ((32, 32), (63, 47), (128, 96)):
            delivered = render_cpu(self.artifact, case["upl"], width, height, seed=case["seed"])
            colors = delivered[:, :, :3].reshape(-1, 3)
            expected = apply_brand(colors, pixel_coordinates(width, height), self.artifact["brand"])
            self.assertTrue(torch.equal(colors, expected))

    def test_small_json_artifact_sidecars_and_no_overwrite(self):
        output = self.directory / "student.json"
        paths = write_artifacts(output, self.artifact, copy.deepcopy(self.report), copy.deepcopy(self.fixture))
        self.assertLess(output.stat().st_size, 2_000_000)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        for path in paths[1:]:
            self.assertEqual(json.loads(path.read_text())["artifactSha256"], digest)
        with self.assertRaises(ValueError):
            write_artifacts(output, self.artifact, self.report, self.fixture)

    def test_canonical_hash_and_no_silent_procedural_substitution(self):
        for identity, canonical in (("0" * 64, self.canonical), (self.identity, None)):
            with self.assertRaises(ValueError):
                train_diffusion(self.graph, identity, self.dataset, canonical, steps=1)
        with self.assertRaisesRegex(ValueError, "opt-in"):
            train_diffusion(self.graph, self.identity, self.dataset, self.canonical, steps=1)
        with self.assertRaisesRegex(ValueError, "explicit --dataset"):
            train_diffusion(self.graph, self.identity, canonical_path=self.canonical, steps=1)
        changed = copy.deepcopy(self.graph)
        changed["id"] = "changed"
        with self.assertRaisesRegex(ValueError, "does not match"):
            verify_canonical_graph(changed, self.identity, self.canonical)

    def test_matching_hash_cannot_authorize_an_invalid_typed_graph(self):
        for fault in ("unknown-operation", "invalid-port-type", "unsupported-encoder"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                graph = copy.deepcopy(self.graph)
                target = next(node for node in graph["nodes"] if node["op"] == "gradient")
                if fault == "unknown-operation":
                    target["op"] = "unknown_model"
                elif fault == "invalid-port-type":
                    target["inputs"]["condition"] = "person"
                else:
                    graph["encoder"]["id"] = "unsupported"
                canonical = write_graph(directory, graph)
                identity = hashlib.sha256(canonical.read_bytes()).hexdigest()
                with self.assertRaises(ValueError):
                    verify_canonical_graph(graph, identity, canonical)

    def test_dataset_validation_and_generated_path(self):
        with np.load(self.dataset) as archive:
            original = {key: archive[key].copy() for key in archive.files}
        source_manifest = json.loads(self.dataset.with_suffix(".manifest.json").read_text())
        for fault in ("graph", "overlap", "seeds", "nan", "range", "dtype", "split", "samples", "sample-record", "incomplete", "duplicate", "fixture-records", "large-teacher-seeds"):
            with self.subTest(fault=fault):
                arrays = copy.deepcopy(original)
                manifest = copy.deepcopy(source_manifest)
                if fault == "graph":
                    manifest["graphHash"] = "0" * 64
                elif fault == "overlap":
                    arrays["upl"][-1] = arrays["upl"][0]
                elif fault == "seeds":
                    arrays["teacher_seed"][:] = 10
                elif fault == "nan":
                    arrays["images"][0, 0, 0, 0] = np.nan
                elif fault == "range":
                    arrays["images"][0, 0, 0, 0] = 1.1
                elif fault == "dtype":
                    arrays["images"] = arrays["images"].astype(np.float64)
                elif fault == "split":
                    arrays["split"][0] = 2
                elif fault == "samples":
                    manifest["samples"] += 1
                elif fault == "sample-record":
                    manifest["samples"] = [{"index": index, "teacherSeed": -1} for index in range(len(arrays["images"]))]
                elif fault == "incomplete":
                    manifest["graphComplete"] = False
                elif fault == "duplicate":
                    arrays["upl"][1] = arrays["upl"][0]
                    arrays["teacher_seed"][1] = arrays["teacher_seed"][0]
                else:
                    if fault == "large-teacher-seeds":
                        arrays["teacher_seed"] += 2 ** 40
                    manifest["samples"] = [{"index": index, "upl": profile.tolist(), "split": int(arrays["split"][index]), "teacherSeed": int(arrays["teacher_seed"][index])} for index, profile in enumerate(arrays["upl"])]
                path = self.directory / f"{fault}.npz"
                np.savez_compressed(path, **arrays)
                path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
                if fault in ("fixture-records", "large-teacher-seeds"):
                    loaded, provenance = load_dataset(path, self.graph, self.identity, allow_procedural=True)
                    np.testing.assert_array_equal(loaded["images"], original["images"])
                    self.assertEqual(provenance["manifest"]["source"], "procedural-fixture-explicit")
                else:
                    with self.assertRaises(ValueError):
                        load_dataset(path, self.graph, self.identity, allow_procedural=True)

    def test_refuses_nonimproving_export(self):
        with mock.patch("angelos_ml.train_diffusion.evaluate_noise", side_effect=[1.0, 1.1]):
            with self.assertRaisesRegex(ValueError, "did not improve"):
                train_diffusion(self.graph, self.identity, self.dataset, self.canonical, steps=1, threads=1, batch_size=2, procedural_fixture=True)

    def test_shuffled_conditioning_changes_only_upl(self):
        clean = torch.rand(4, 3, 32, 32)
        upl = torch.tensor([[0.1] * 4, [0.1] * 4, [0.5] * 4, [0.9] * 4])
        timesteps = torch.arange(4)
        noise = torch.randn_like(clean)
        shuffled = shuffled_conditioning_bank((clean, upl, timesteps, noise))
        self.assertIs(shuffled[0], clean)
        self.assertIs(shuffled[2], timesteps)
        self.assertIs(shuffled[3], noise)
        self.assertTrue((shuffled[1] != upl).any(dim=1).all())
        self.assertEqual(set(map(tuple, shuffled[1].tolist())), set(map(tuple, upl.tolist())))

    def test_no_brand_is_not_composited(self):
        artifact = copy.deepcopy(self.artifact)
        artifact["brand"] = None
        case = self.fixture["rasterCases"][0]
        raw = render_cpu(artifact, case["upl"], composite=False, seed=case["seed"])
        delivered = render_cpu(artifact, case["upl"], seed=case["seed"])
        self.assertTrue(torch.equal(raw, delivered))

    @unittest.skipUnless(shutil.which("node") and (ROOT / "src/browser/diffusion.js").exists(), "Browser diffusion reference unavailable")
    def test_browser_cpu_reference_matches_one_native_case(self):
        case = self.fixture["rasterCases"][0]
        script = "import {validateDiffusion,renderDiffusionCpu} from './src/browser/diffusion.js'; let input=''; for await(const chunk of process.stdin) input+=chunk; const {artifact,item}=JSON.parse(input); validateDiffusion(artifact); const raw=Array.from(renderDiffusionCpu(artifact,item.upl,32,32,{seed:item.seed,composite:false})); console.log(JSON.stringify(raw));"
        process = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps({"artifact": self.artifact, "item": case}), text=True, capture_output=True, cwd=ROOT, check=True, timeout=90)
        np.testing.assert_allclose(json.loads(process.stdout), case["rawStudentRgba"], atol=0.001, rtol=0)


class DatasetProvenanceTests(unittest.TestCase):
    """Synthetic files exercise provenance consistency, never provider attestation."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="angelos-provenance-test-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "injected.npz"
        self.graph = example_graph()
        generation = next(node for node in self.graph["nodes"] if node["id"] == "field")
        generation.update(op="generate_image", inputs={"prompt": "scene"}, params={"model": "tiny-sd", "steps": 2, "guidance": 1, "width": 256, "height": 256})
        self.graph["nodes"].append({"id": "scene", "op": "prompt", "inputs": {"condition": "energy"}, "params": {"low": "quiet", "high": "vivid", "threshold": 0.5}})
        self.identity = graph_hash(self.graph)
        self.profiles = np.array([[0.1] * 4, [0.2] * 4, [0.7] * 4, [0.9] * 4], dtype=np.float32)
        np.savez_compressed(self.path, images=np.zeros((4, 3, 32, 32), dtype=np.float32), upl=self.profiles, split=np.array([0, 0, 1, 1]), teacher_seed=np.arange(4, dtype=np.int64))
        metadata = {**model_metadata("tiny-sd"), "backend": "injected-test-pipeline", "device": "cpu", "dtype": "torch.float32"}
        self.manifest = {
            "graphHash": self.identity, "graph": self.graph, "encoder": self.graph["encoder"],
            "models": [metadata], "source": "injected-test-pipeline", "synthetic": True,
            "samples": [{"index": index, "upl": profile.tolist(), "split": int(index >= 2), "teacherSeed": index,
                         "generations": [{"node": "field", "model": copy.deepcopy(metadata), "params": copy.deepcopy(generation["params"]), "seed": node_seed(index, "field")}]} for index, profile in enumerate(self.profiles)],
        }

    def load(self, manifest, allow_fixture=True):
        self.path.with_suffix(".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return load_dataset(self.path, self.graph, self.identity, allow_procedural=allow_fixture)

    def test_valid_injected_and_generated_metadata(self):
        _, provenance = self.load(self.manifest)
        self.assertTrue(provenance["proceduralFixture"])
        self.assertEqual(provenance["manifest"]["source"], "injected-test-pipeline")
        generated = copy.deepcopy(self.manifest)
        generated["source"] = "diffusers-local-open-weights"
        generated["models"][0]["backend"] = "diffusers"
        for sample in generated["samples"]:
            sample["generations"][0]["model"]["backend"] = "diffusers"
        _, provenance = self.load(generated, allow_fixture=False)
        self.assertFalse(provenance["proceduralFixture"])

    def test_rejects_missing_mismatched_duplicate_and_extra_models(self):
        for fault in ("missing", "mismatched", "duplicate", "extra", "not-list", "missing-id"):
            with self.subTest(fault=fault):
                manifest = copy.deepcopy(self.manifest)
                if fault == "missing":
                    manifest["models"] = []
                elif fault == "mismatched":
                    manifest["models"] = [{**model_metadata("ltx-video"), "backend": "injected-test-pipeline"}]
                elif fault == "duplicate":
                    manifest["models"].append(copy.deepcopy(manifest["models"][0]))
                elif fault == "extra":
                    manifest["models"].append({**model_metadata("flux2-klein-4b"), "backend": "injected-test-pipeline"})
                elif fault == "not-list":
                    manifest["models"] = {"tiny-sd": manifest["models"][0]}
                else:
                    del manifest["models"][0]["id"]
                with self.assertRaises(ValueError):
                    self.load(manifest)

    def test_rejects_wrong_or_missing_pinned_fields(self):
        for field in ("repo", "revision", "pipeline", "kind", "license"):
            for missing in (False, True):
                with self.subTest(field=field, missing=missing):
                    manifest = copy.deepcopy(self.manifest)
                    if missing:
                        del manifest["models"][0][field]
                    else:
                        manifest["models"][0][field] = "not-the-pinned-value"
                    with self.assertRaisesRegex(ValueError, "pinned registry"):
                        self.load(manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest["models"][0]["safetensors"] = True
        with self.assertRaisesRegex(ValueError, "pinned registry"):
            self.load(manifest)

    def test_injected_backend_requires_fixture_optin_and_honest_source(self):
        with self.assertRaisesRegex(ValueError, "opt-in"):
            self.load(self.manifest, allow_fixture=False)
        for fault in ("unmarked-source", "unknown-backend", "missing-backend", "not-synthetic"):
            with self.subTest(fault=fault):
                manifest = copy.deepcopy(self.manifest)
                if fault == "unmarked-source":
                    manifest["source"] = "diffusers-local-open-weights"
                elif fault == "unknown-backend":
                    manifest["models"][0]["backend"] = "mock"
                elif fault == "missing-backend":
                    del manifest["models"][0]["backend"]
                else:
                    manifest["synthetic"] = False
                with self.assertRaises(ValueError):
                    self.load(manifest)

    def test_per_sample_generation_metadata_must_agree(self):
        for fault in ("model", "revision", "backend", "device", "params", "seed", "node", "duplicate", "missing-node", "not-list"):
            with self.subTest(fault=fault):
                manifest = copy.deepcopy(self.manifest)
                sample = manifest["samples"][0]
                generation = sample["generations"][0]
                if fault == "model":
                    generation["model"]["id"] = "ltx-video"
                elif fault in ("revision", "backend", "device"):
                    generation["model"][fault] = "contradiction"
                elif fault == "params":
                    generation["params"]["steps"] += 1
                elif fault == "seed":
                    generation["seed"] += 1
                elif fault == "node":
                    generation["node"] = "unknown"
                elif fault == "duplicate":
                    sample["generations"].append(copy.deepcopy(generation))
                elif fault == "missing-node":
                    sample["generations"] = []
                else:
                    sample["generations"] = {}
                with self.assertRaises(ValueError):
                    self.load(manifest)

    def test_explicit_fixtures_can_omit_optional_sample_details(self):
        for form in ("count", "minimal-samples", "minimal-generations"):
            with self.subTest(form=form):
                manifest = copy.deepcopy(self.manifest)
                if form == "count":
                    manifest["samples"] = 4
                elif form == "minimal-samples":
                    manifest["samples"] = [{"index": index} for index in range(4)]
                else:
                    for sample in manifest["samples"]:
                        sample["generations"] = [{"node": "field", "model": {"id": "tiny-sd"}}]
                self.load(manifest)


if __name__ == "__main__":
    unittest.main()
