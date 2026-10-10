"""Graph/dataset migration fixtures, not semantic-model or HD quality benchmarks."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from angelos_ml.compile import compile_teacher
from angelos_ml.diffusion import DiffusionStudent
from angelos_ml.graph import canonical_json, graph_hash, validate_graph
from angelos_ml.teacher import RasterTeacher, TensorTeacher
from angelos_ml.train_diffusion import load_dataset, main, train_diffusion, verify_canonical_graph
from angelos_ml.upl import DIMENSIONS, encoder_metadata


def semantic_graph(dimension=128):
    return {"version": "angelos.graph.v2", "id": "synthetic_campaign", "encoder": encoder_metadata(dimension),
            "nodes": [{"id": "person", "op": "upl", "inputs": {}, "params": {}},
                      {"id": "coordinate", "op": "feature", "inputs": {"input": "person"}, "params": {"index": dimension - 1}},
                      {"id": "image", "op": "gradient", "inputs": {"condition": "coordinate"},
                       "params": {"low": [0, 0, 0], "high": [1, 1, 1], "angle": 0}}],
            "output": "image"}


def frozen_fixture(directory):
    graph = semantic_graph()
    digest = graph_hash(graph)
    graph_path = Path(directory) / "graph.json"
    graph_path.write_text(json.dumps(graph))
    canonical_path = Path(directory) / "canonical.json"
    canonical_path.write_text(canonical_json({**graph, "nodes": sorted(graph["nodes"], key=lambda node: node["id"])}))
    generator = torch.Generator().manual_seed(72)
    vectors = torch.randn(8, 128, generator=generator)
    vectors = (vectors / vectors.norm(dim=1, keepdim=True)).numpy()
    split = np.array([0] * 4 + [1] * 4, dtype=np.int64)
    seeds = np.arange(8, dtype=np.int64) + 42
    samples = []
    for index, vector in enumerate(vectors):
        bundle = {"format": "angelos.upl-input.v2", "synthetic": True, "adult": True,
                  "factsSchema": "angelos.facts.v2", "facts": {"ageBand": "25-34", "interests": ["synthetic fixture"]},
                  "encoder": copy.deepcopy(graph["encoder"]), "upl": vector.tolist(),
                  "context": {"campaignId": graph["id"], "factRevision": 1, "eventRevision": index + 1,
                              "consentVersion": 1, "asOf": 100 + index, "ingredientsHash": f"{index + 1:064x}"},
                  "recentIntent": [{"kind": "save", "topic": f"synthetic topic {index}", "occurredAt": 90, "expiresAt": 2000}]}
        samples.append({"index": index, "profileId": f"synthetic-person-{index // 2}", "upl": vector.tolist(),
                        "encoder": copy.deepcopy(graph["encoder"]), "uplBundle": bundle,
                        "split": int(split[index]), "teacherSeed": int(seeds[index])})
    manifest = {"format": "angelos.teacher-dataset.v1", "graph": graph, "graphHash": digest,
                "encoder": copy.deepcopy(graph["encoder"]), "models": [], "samples": samples,
                "source": "synthetic-native32-fixture", "synthetic": True,
                "limitations": "Constant synthetic test pixels; no teacher or embedding model executed."}
    dataset = Path(directory) / "fixture.npz"
    np.savez(dataset, images=np.full((8, 3, 32, 32), 0.5, dtype=np.float32),
             upl=vectors, split=split, teacher_seed=seeds)
    sidecar = dataset.with_suffix(".manifest.json")
    sidecar.write_text(json.dumps(manifest))
    return graph, digest, graph_path, canonical_path, dataset, sidecar, manifest


class UPLGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_v2_full_metadata_dynamic_width_and_indices(self):
        for dimension in DIMENSIONS:
            with self.subTest(dimension=dimension):
                graph = semantic_graph(dimension)
                self.assertEqual(validate_graph(graph)[-1]["id"], "image")
                for invalid_index in (-1, dimension, True, 1.5):
                    invalid = copy.deepcopy(graph)
                    invalid["nodes"][1]["params"]["index"] = invalid_index
                    with self.assertRaises(ValueError):
                        validate_graph(invalid)
        for field in encoder_metadata(128):
            graph = semantic_graph()
            del graph["encoder"][field]
            with self.assertRaisesRegex(ValueError, "encoder"):
                validate_graph(graph)
        for field, value in (("revision", "a" * 40), ("pooling", "mean"), ("dimension", 4), ("normalization", "none")):
            graph = semantic_graph()
            graph["encoder"][field] = value
            with self.assertRaisesRegex(ValueError, "encoder"):
                validate_graph(graph)

    def test_v1_and_v2_encoder_contracts_cannot_be_interchanged(self):
        graph = semantic_graph()
        graph["version"] = "angelos.graph.v1"
        with self.assertRaises(ValueError):
            validate_graph(graph)
        graph["version"] = "angelos.graph.v2"
        graph["encoder"] = {"id": "structured-demo-v1", "dimension": 4}
        with self.assertRaises(ValueError):
            validate_graph(graph)

    def test_v2_teachers_and_compiler_fail_before_execution(self):
        graph = semantic_graph()
        with self.assertRaisesRegex(ValueError, "cannot execute graph.v2"):
            TensorTeacher(graph)
        with patch("angelos_ml.teacher.PipelineRegistry") as registry:
            with self.assertRaisesRegex(ValueError, "cannot execute graph.v2"):
                RasterTeacher(graph)
            registry.assert_not_called()
        with patch("angelos_ml.compile.profile_splits") as profiles, patch("torch.export.export") as export:
            with self.assertRaisesRegex(ValueError, "cannot execute graph.v2"):
                compile_teacher(graph, graph_hash(graph), samples=2)
            profiles.assert_not_called()
            export.assert_not_called()

    def test_dataset_groups_by_person_not_vector_or_changing_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            graph, digest, _, canonical_path, dataset, _, manifest = frozen_fixture(directory)
            self.assertEqual(verify_canonical_graph(graph, digest, canonical_path), digest)
            arrays, provenance = load_dataset(dataset, graph, digest, True)
            self.assertEqual(arrays["upl"].shape, (8, 128))
            self.assertEqual(provenance["trainProfileCount"], 2)
            self.assertEqual(provenance["heldoutProfileCount"], 2)
            self.assertEqual(provenance["trainUplCount"], 4)
            self.assertEqual(provenance["heldoutUplCount"], 4)
            self.assertEqual(provenance["splitProfileIds"]["train"], ["synthetic-person-0", "synthetic-person-1"])
            self.assertEqual(provenance["manifest"]["samples"][0]["uplBundle"], manifest["samples"][0]["uplBundle"])

    def test_bad_person_splits_and_metadata_fail_before_model_initialization(self):
        for fault in ("missing-person", "person-overlap", "one-person", "encoder", "bundle-encoder", "bundle-vector", "bundle-missing"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                graph, digest, _, canonical_path, dataset, sidecar, manifest = frozen_fixture(directory)
                if fault == "missing-person":
                    del manifest["samples"][0]["profileId"]
                elif fault == "person-overlap":
                    manifest["samples"][4]["profileId"] = manifest["samples"][0]["profileId"]
                elif fault == "one-person":
                    for sample in manifest["samples"][:4]:
                        sample["profileId"] = "one-person"
                elif fault == "encoder":
                    manifest["encoder"]["revision"] = "0" * 40
                elif fault == "bundle-encoder":
                    manifest["samples"][0]["uplBundle"]["encoder"]["revision"] = "0" * 40
                elif fault == "bundle-vector":
                    manifest["samples"][0]["uplBundle"]["upl"] = manifest["samples"][1]["upl"]
                else:
                    del manifest["samples"][0]["uplBundle"]
                sidecar.write_text(json.dumps(manifest))
                with patch("angelos_ml.train_diffusion.DiffusionStudent") as student:
                    with self.assertRaises(ValueError):
                        train_diffusion(graph, digest, dataset=dataset, canonical_path=canonical_path,
                                        steps=1, batch_size=2, threads=1, procedural_fixture=True)
                    student.assert_not_called()

    def test_v2_training_cli_one_step_exports_only_native32_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            _, digest, graph_path, canonical_path, dataset, _, manifest = frozen_fixture(directory)
            output = Path(directory) / "fixture-student.json"
            arguments = ["--graph", str(graph_path), "--graph-canonical", str(canonical_path), "--graph-hash", digest,
                         "--dataset", str(dataset), "--output", str(output), "--procedural-fixture", "--steps", "1",
                         "--batch-size", "2", "--threads", "1", "--device", "cpu", "--seed", "2026"]
            with redirect_stdout(io.StringIO()), patch("angelos_ml.train_diffusion.generate_procedural_fixture") as fabricate:
                main(arguments)
                fabricate.assert_not_called()
            artifact = json.loads(output.read_text())
            report = json.loads(output.with_suffix(".metrics.json").read_text())
            checks = json.loads(output.with_suffix(".parity.json").read_text())
            self.assertEqual(artifact["format"], "angelos.diffusion.v2")
            self.assertEqual(artifact["training"]["steps"], 1)
            self.assertTrue(artifact["training"]["fixtureOnly"])
            self.assertFalse(artifact["training"]["browserExportSupported"])
            self.assertFalse(artifact["training"]["meets1080pRequirement"])
            self.assertFalse(artifact["training"]["promoted"])
            self.assertIn("semantic-vector-only", artifact["training"]["conditioning"])
            self.assertEqual(artifact["training"]["trainProfileCount"], 2)
            self.assertEqual(artifact["training"]["heldoutProfileCount"], 2)
            self.assertEqual(report["dataset"]["manifest"]["samples"][0]["uplBundle"], manifest["samples"][0]["uplBundle"])
            self.assertEqual(checks["format"], "angelos.diffusion-python-checks.v2")
            self.assertEqual(DiffusionStudent.from_artifact(artifact).upl_dimension, 128)


if __name__ == "__main__":
    unittest.main()
