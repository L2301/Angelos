"""Offline contract tests: injected pixels are never evidence of model inference."""

import copy
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn.functional as functional

from angelos_ml.compile import TeacherModule
from angelos_ml.dataset import build_dataset, main, split_profiles, write_dataset
from angelos_ml.graph import graph_hash, validate_graph
from angelos_ml.models import MODELS, default_cache_dir, download_model, model_spec
from angelos_ml.providers import PipelineRegistry, load_pipeline, node_seed
from angelos_ml.teacher import RasterTeacher, TensorTeacher, apply_brand, pixel_coordinates


ROOT = Path(__file__).resolve().parents[2]


def example(name="aurora-generative"):
    source = "import {readFileSync} from 'node:fs'; import {parseSource} from './src/core/graph.js'; console.log(JSON.stringify(parseSource(readFileSync(process.argv[1], 'utf8'))));"
    result = subprocess.run(["node", "--input-type=module", "-e", source, str(ROOT / "examples" / f"{name}.ang")], cwd=ROOT, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def named(graph, operation):
    return next(node for node in graph["nodes"] if node["op"] == operation)


class FakePipeline:
    def __init__(self, video=False):
        self.calls = []
        self.video = video

    def __call__(self, **arguments):
        self.calls.append(arguments)
        shape = (1, arguments["num_frames"], 3, arguments["height"], arguments["width"]) if self.video else (1, 3, arguments["height"], arguments["width"])
        pixels = torch.rand(shape, generator=arguments["generator"])
        return SimpleNamespace(frames=pixels) if self.video else SimpleNamespace(images=pixels)


class GenerativeGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.image_graph = example()
        cls.video_graph = example("aurora-ltx")

    def test_examples_roundtrip_cross_language_hash_and_immutable_models(self):
        for name in ("aurora-generative", "aurora-flux", "aurora-ltx"):
            graph = example(name)
            self.assertEqual(graph["id"], "aurora")
            validate_graph(graph)
            source = "import {graphHash} from './src/core/graph.js'; console.log(await graphHash(JSON.parse(process.argv[1])));"
            result = subprocess.run(["node", "--input-type=module", "-e", source, json.dumps(graph)], cwd=ROOT, text=True, capture_output=True, check=True)
            self.assertEqual(graph_hash(graph), result.stdout.strip())
        self.assertEqual(len(MODELS), 5)
        for specification in MODELS.values():
            self.assertRegex(specification.revision, r"^[0-9a-f]{40}$")
        with self.assertRaises(TypeError):
            MODELS["other"] = None
        for model in ("http://example.org/model", "segmind/tiny-sd", "tiny-sd@main", {}, None):
            with self.assertRaises(ValueError):
                model_spec(model)

    def test_invalid_parameters_and_types(self):
        for operation, field, value in (
            ("generate_image", "width", 32), ("generate_image", "height", 257),
            ("generate_image", "steps", True), ("generate_image", "steps", 1.5),
            ("generate_image", "guidance", float("nan")), ("generate_image", "model", "ltx-video"),
            ("prompt", "threshold", 2), ("prompt", "low", "{condition.__class__}"),
            ("prompt", "high", ""), ("prompt", "high", "a" * 2049),
        ):
            with self.subTest(operation=operation, field=field, value=value):
                graph = copy.deepcopy(self.image_graph)
                named(graph, operation)["params"][field] = value
                with self.assertRaises(ValueError):
                    validate_graph(graph)
        for operation, field, value in (("generate_video", "num_frames", 8), ("generate_video", "num_frames", 9.5), ("frame", "index", 9), ("frame", "index", True)):
            graph = copy.deepcopy(self.video_graph)
            named(graph, operation)["params"][field] = value
            with self.assertRaises(ValueError):
                validate_graph(graph)
        graph = copy.deepcopy(self.image_graph)
        named(graph, "generate_image")["params"]["seed"] = 42
        with self.assertRaises(ValueError):
            validate_graph(graph)
        graph = copy.deepcopy(self.image_graph)
        named(graph, "generate_image")["inputs"]["prompt"] = "openness"
        with self.assertRaisesRegex(ValueError, "Type mismatch"):
            validate_graph(graph)

    def test_coordinate_teacher_and_compiler_fail_actionably(self):
        for teacher in (TensorTeacher, TeacherModule):
            with self.assertRaisesRegex(ValueError, "RasterTeacher.*no procedural fallback"):
                teacher(self.image_graph)

    def test_deterministic_device_and_native_previews(self):
        registry = PipelineRegistry(pipelines={"tiny-sd": FakePipeline()})
        with tempfile.TemporaryDirectory() as directory:
            previews = Path(directory) / "previews"
            arrays, manifest = build_dataset(self.image_graph, train_count=1, heldout_count=1, seeds_per_profile=1, registry=registry, preview_dir=previews)
            from PIL import Image

            with Image.open(previews / "sample-000000.png") as image:
                self.assertEqual(image.size, (256, 256))
            self.assertEqual(manifest["samples"][0]["deterministicTensorDevice"], "cpu")
            self.assertEqual(arrays["images"].shape, (2, 3, 32, 32))
            with self.assertRaises(FileExistsError):
                build_dataset(self.image_graph, registry=registry, preview_dir=previews)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable in this execution environment")
    def test_mps_raster_transform_device_without_model_download(self):
        registry = PipelineRegistry(device="mps", pipelines={"tiny-sd": FakePipeline()})
        teacher = RasterTeacher(self.image_graph, registry)
        images = teacher.render([[0.2, 0.3, 0.7, 0.4]], seed=11)
        self.assertEqual(images.device.type, "mps")
        self.assertEqual(teacher.last_provenance[0]["deterministicTensorDevice"], "mps:0")
        arrays, manifest = build_dataset(self.image_graph, train_count=1, heldout_count=1, seeds_per_profile=1, registry=registry)
        self.assertEqual(arrays["images"].dtype, np.float32)
        self.assertTrue(manifest["deterministicTensorDevice"].startswith("mps"))
        graph = example("aurora")
        profiles = [[0.2, 0.3, 0.7, 0.4]]
        accelerated = RasterTeacher(graph, PipelineRegistry(device="mps")).render(profiles, seed=11, width=64, height=64)
        reference = RasterTeacher(graph).render(profiles, seed=11, width=64, height=64)
        torch.testing.assert_close(accelerated.cpu(), reference, atol=1e-6, rtol=0)

    def test_real_routing_explicit_seed_and_prompt_interpolation(self):
        graph = copy.deepcopy(self.image_graph)
        named(graph, "prompt")["params"].update(low="quiet {condition}", high="vivid {condition}")
        pipeline = FakePipeline()
        registry = PipelineRegistry(pipelines={"tiny-sd": pipeline})
        teacher = RasterTeacher(graph, registry)
        profiles = [[0, 0, 0.25, 0.2], [1, 1, 0.5, 0.8]]
        state = torch.random.get_rng_state().clone()
        first = teacher.render(profiles, seed=[42, 43])
        torch.testing.assert_close(torch.random.get_rng_state(), state)
        second = teacher.render(profiles, seed=[42, 43])
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, teacher.render(profiles, seed=[44, 45])))
        self.assertEqual(first.shape, (2, 3, 256, 256))
        self.assertEqual(pipeline.calls[0]["prompt"], "quiet 0.250000")
        self.assertEqual(pipeline.calls[1]["prompt"], "vivid 0.500000")
        self.assertEqual(pipeline.calls[0]["generator"].initial_seed(), node_seed(42, "picture"))
        self.assertEqual(pipeline.calls[0]["output_type"], "pt")
        self.assertEqual(pipeline.calls[0]["num_inference_steps"], 12)
        self.assertEqual(pipeline.calls[0]["guidance_scale"], 7.5)
        self.assertEqual(registry.metadata("tiny-sd")["backend"], "injected-test-pipeline")
        self.assertNotEqual(node_seed(42, "picture"), node_seed(42, "other"))

    def test_native_graph_brand_then_downsample(self):
        graph = copy.deepcopy(self.image_graph)
        pipeline = FakePipeline()
        registry = PipelineRegistry(pipelines={"tiny-sd": pipeline})
        arrays, manifest = build_dataset(graph, train_count=1, heldout_count=1, seeds_per_profile=1, registry=registry)
        self.assertEqual(len(pipeline.calls), 2)
        self.assertEqual(pipeline.calls[0]["width"], 256)
        profile = torch.from_numpy(arrays["upl"][0])
        native = RasterTeacher(graph, registry).render(profile, seed=int(arrays["teacher_seed"][0]))
        expected = functional.interpolate(native, size=(32, 32), mode="area")[0]
        torch.testing.assert_close(torch.from_numpy(arrays["images"][0]), expected)
        coordinates = pixel_coordinates(256, 256)
        pixels = native[0].permute(1, 2, 0).reshape(-1, 3)
        torch.testing.assert_close(pixels, apply_brand(pixels, coordinates, named(graph, "brand")["params"]))
        self.assertEqual(manifest["samples"][0]["nativeShape"], [3, 256, 256])
        self.assertEqual(manifest["graphHash"], graph_hash(graph))

    def test_all_deterministic_raster_ops_match_coordinate_teacher(self):
        graph = example("aurora")
        profiles = torch.tensor([[0.2, 0.3, 0.4, 0.5], [1.0, 0.0, 0.7, 0.3]])
        coordinates = pixel_coordinates(32, 24)
        raster = RasterTeacher(graph).render(profiles, seed=1, width=32, height=24)
        for index, profile in enumerate(profiles):
            expected = TensorTeacher(graph)(profile.expand(len(coordinates), -1), coordinates)
            torch.testing.assert_close(raster[index].permute(1, 2, 0).reshape(-1, 3), expected, atol=1e-6, rtol=0)

    def test_video_generation_selects_exact_frame_before_transforms(self):
        graph = copy.deepcopy(self.video_graph)
        pipeline = FakePipeline(video=True)
        registry = PipelineRegistry(pipelines={"ltx-video": pipeline})
        image = RasterTeacher(graph, registry).render([[0, 0, 0.7, 0.5]], seed=19)
        self.assertEqual(image.shape, (1, 3, 512, 512))
        self.assertEqual(pipeline.calls[0]["num_frames"], 9)
        generator = torch.Generator().manual_seed(node_seed(19, "clip"))
        frame = torch.rand((1, 9, 3, 512, 512), generator=generator)[0, 4]
        coordinates = pixel_coordinates(512, 512)
        distance = (2 * ((coordinates[:, 0] - 0.5).square() + (coordinates[:, 1] - 0.5).square())).clamp(0, 1)
        pixels = frame.permute(1, 2, 0).reshape(-1, 3) * (1 - 0.2 * distance[:, None])
        expected = apply_brand(pixels, coordinates, named(graph, "brand")["params"])
        torch.testing.assert_close(image[0].permute(1, 2, 0).reshape(-1, 3), expected)

    def test_dataset_split_provenance_npz_and_no_training(self):
        registry = PipelineRegistry(pipelines={"tiny-sd": FakePipeline()})
        arrays, manifest = build_dataset(self.image_graph, train_count=2, heldout_count=1, seeds_per_profile=2, registry=registry)
        self.assertEqual(arrays["images"].shape, (6, 3, 32, 32))
        self.assertEqual(arrays["images"].dtype, np.float32)
        self.assertEqual(arrays["upl"].shape, (6, 4))
        self.assertTrue(np.isfinite(arrays["images"]).all())
        self.assertTrue(((arrays["images"] >= 0) & (arrays["images"] <= 1)).all())
        train = {tuple(profile) for profile in arrays["upl"][arrays["split"] == 0]}
        heldout = {tuple(profile) for profile in arrays["upl"][arrays["split"] == 1]}
        self.assertFalse(train & heldout)
        self.assertEqual((len(train), len(heldout)), (2, 1))
        self.assertTrue(manifest["synthetic"])
        self.assertFalse(manifest["training"])
        self.assertEqual(manifest["source"], "injected-test-pipeline")
        self.assertEqual(manifest["deterministicTensorDevice"], "cpu")
        self.assertEqual(manifest["models"][0]["revision"], MODELS["tiny-sd"].revision)
        self.assertEqual(manifest["samples"][0]["generations"][0]["seed"], node_seed(11, "picture"))
        np.testing.assert_array_equal(arrays["teacher_seed"], np.arange(11, 17))
        with tempfile.TemporaryDirectory() as directory:
            output, sidecar = write_dataset(Path(directory) / "dataset.npz", arrays, manifest)
            self.assertEqual(sidecar.name, "dataset.manifest.json")
            with np.load(output, allow_pickle=False) as loaded:
                self.assertEqual(set(loaded.files), {"images", "upl", "split", "teacher_seed"})
                np.testing.assert_array_equal(loaded["images"], arrays["images"])
            self.assertEqual(json.loads(sidecar.read_text())["graphHash"], manifest["graphHash"])
            with self.assertRaises(ValueError):
                write_dataset(output, arrays, manifest)

    def test_api_and_cli_defaults_match_forty_image_contract(self):
        pipeline = FakePipeline()
        registry = PipelineRegistry(pipelines={"tiny-sd": pipeline})
        arrays, manifest = build_dataset(self.image_graph, registry=registry)
        self.assertEqual(arrays["images"].shape, (40, 3, 32, 32))
        self.assertEqual((manifest["trainProfiles"], manifest["heldoutProfiles"], manifest["seedsPerProfile"]), (16, 4, 2))
        np.testing.assert_array_equal(arrays["split"], [0] * 32 + [1] * 8)
        self.assertEqual(len({tuple(profile) for profile in arrays["upl"]}), 20)
        self.assertEqual(len(pipeline.calls), 40)
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "graph.json"
            graph_path.write_text(json.dumps(self.image_graph))
            output = Path(directory) / "defaults.npz"
            with patch("angelos_ml.dataset.PipelineRegistry", return_value=registry), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                main(["--graph", str(graph_path), "--output", str(output)])
            self.assertEqual(json.loads(stdout.getvalue())["samples"], 40)
            with np.load(output, allow_pickle=False) as loaded:
                for name, expected in arrays.items():
                    np.testing.assert_array_equal(loaded[name], expected)
            self.assertEqual(json.loads(output.with_suffix(".manifest.json").read_text()), manifest)

    def test_seeds_profiles_and_generator_results_fail_closed(self):
        teacher = RasterTeacher(self.image_graph, PipelineRegistry(pipelines={"tiny-sd": FakePipeline()}))
        for seed in (-1, True, 2 ** 53, 1.1, [1, 2]):
            with self.assertRaises(ValueError):
                teacher.render([[0, 0, 0, 0]], seed=seed)
        for profile in ([[0, 0, 0]], [[0, 0, float("nan"), 0]], [[0, 0, 2, 0]]):
            with self.assertRaises(ValueError):
                teacher.render(profile, seed=1)
        for pixels in (torch.zeros(1, 3, 32, 32), torch.full((1, 3, 256, 256), float("nan")), torch.full((1, 3, 256, 256), 2.0)):
            registry = PipelineRegistry(pipelines={"tiny-sd": Mock(return_value=SimpleNamespace(images=pixels))})
            with self.assertRaises(ValueError):
                RasterTeacher(self.image_graph, registry).render([[0, 0, 0, 0]], seed=1)
        with self.assertRaises(ValueError):
            split_profiles(7, 0, 1)

    def test_offline_adapter_loads_exact_local_snapshot_and_never_downloads(self):
        for model, specification in MODELS.items():
            if specification.kind == "text":
                continue
            pipeline = Mock()
            pipeline.to.return_value = pipeline
            with patch(f"diffusers.{specification.pipeline}.from_pretrained", return_value=pipeline) as loader, patch("huggingface_hub.snapshot_download", side_effect=AssertionError("Runtime attempted a download")):
                loaded = load_pipeline(model, cache_dir="/offline/cache")
                self.assertIs(loaded, pipeline)
                arguments = loader.call_args.kwargs
                self.assertEqual(loader.call_args.args, (specification.repo,))
                self.assertEqual(arguments["revision"], specification.revision)
                self.assertTrue(arguments["local_files_only"])
                self.assertFalse(arguments["trust_remote_code"])
                self.assertEqual(arguments["use_safetensors"], specification.safetensors)
            with patch(f"diffusers.{specification.pipeline}.from_pretrained", side_effect=OSError("missing cache")):
                with self.assertRaisesRegex(RuntimeError, "models download.*no procedural fallback"):
                    load_pipeline(model)

    def test_download_is_explicit_pinned_and_filters_executable_files(self):
        with patch("huggingface_hub.snapshot_download", return_value="/snapshot") as download:
            self.assertEqual(download_model("tiny-sd"), "/snapshot")
            self.assertEqual(download.call_args.kwargs["revision"], MODELS["tiny-sd"].revision)
            self.assertFalse(download.call_args.kwargs["local_files_only"])
            self.assertIn("*/*.bin", download.call_args.kwargs["allow_patterns"])
            self.assertNotIn("*.py", download.call_args.kwargs["allow_patterns"])
            self.assertEqual(download.call_args.kwargs["cache_dir"], default_cache_dir())

    def test_installed_bin_loader_uses_restricted_weights_only(self):
        from diffusers.models.model_loading_utils import load_state_dict

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.bin"
            torch.save({"weight": torch.ones(2)}, checkpoint)
            with patch("torch.load", wraps=torch.load) as loader:
                loaded = load_state_dict(str(checkpoint))
            self.assertTrue(loader.call_args.kwargs["weights_only"])
            torch.testing.assert_close(loaded["weight"], torch.ones(2))

    def test_cli_writes_contract_and_missing_models_write_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            graph_path = Path(directory) / "graph.json"
            graph_path.write_text(json.dumps(self.image_graph))
            output = Path(directory) / "samples.npz"
            registry = PipelineRegistry(pipelines={"tiny-sd": FakePipeline()})
            with patch("angelos_ml.dataset.PipelineRegistry", return_value=registry), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                main(["--graph", str(graph_path), "--output", str(output), "--train-count", "1", "--heldout-count", "1", "--seeds-per-profile", "1"])
            self.assertEqual(json.loads(stdout.getvalue())["samples"], 2)
            failed = Path(directory) / "failed.npz"
            with patch("angelos_ml.providers.load_pipeline", side_effect=RuntimeError("Model absent; model-download tiny-sd")), patch("sys.stderr", new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit) as error:
                    main(["--graph", str(graph_path), "--output", str(failed)])
            self.assertEqual(error.exception.code, 2)
            self.assertIn("No procedural fallback", stderr.getvalue())
            self.assertFalse(failed.exists())
            self.assertFalse(failed.with_suffix(".manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
