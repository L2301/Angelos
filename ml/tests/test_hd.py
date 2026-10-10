"""Offline fixtures exercise HD contracts, never pretrained quality or GPU success."""

import copy
import fnmatch
import inspect
import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderDC, SanaSprintPipeline, SCMScheduler

from angelos_ml.hd import CROP_BOX, LATENT_SHAPE, NATIVE_SIZE, crop_native, generate, main, native_from_float, read_source, validate_source
from angelos_ml.hd_models import COMPONENTS, DIFFUSERS_VERSION, DOWNLOAD_PATTERNS, MODEL_ID, REPOSITORY, REVISION
from angelos_ml.hd_models import default_cache_dir, download_model, load_pipeline, model_metadata, validate_device


ROOT = Path(__file__).resolve().parents[2]
PINNED_MODEL_INDEX = {
    "_class_name": "SanaSprintPipeline", "_diffusers_version": "0.33.0.dev0",
    "scheduler": ["diffusers", "SCMScheduler"], "text_encoder": ["transformers", "Gemma2Model"],
    "tokenizer": ["transformers", "GemmaTokenizerFast"],
    "transformer": ["diffusers", "SanaTransformer2DModel"], "vae": ["diffusers", "AutoencoderDC"],
}


class FixtureTokenizer:
    def __init__(self, tokens=8):
        self.tokens = tokens

    def encode(self, text, **arguments):
        return [1] * self.tokens


class FixturePipeline:
    def __init__(self, size=NATIVE_SIZE, latent_shape=LATENT_SHAPE, tokens=8, fail=False):
        self.tokenizer = FixtureTokenizer(tokens)
        self.size = size
        self.latent_shape = latent_shape
        self.fail = fail
        self.calls = []

    def __call__(self, **arguments):
        self.calls.append(arguments)
        if self.fail:
            raise RuntimeError("fixture operator failure")
        latents = torch.zeros(self.latent_shape)
        for step in range(arguments["num_inference_steps"]):
            arguments["callback_on_step_end"](self, step, torch.tensor(0.0), {"latents": latents})
        rows = [bytes((row % 256, (row // 4) % 256, 123)) * self.size[0] for row in range(self.size[1])]
        native = Image.frombytes("RGB", self.size, b"".join(rows))
        return SimpleNamespace(images=np.asarray(native, dtype=np.float32)[None, ...] / 255)


class FixtureComponent(torch.nn.Module):
    def __init__(self, kind, dtype):
        super().__init__()
        self.kind = kind
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=dtype))
        self.config = SimpleNamespace(encoder_block_out_channels=[1] * 6, scaling_factor=0.41407,
                                      in_channels=32, sample_size=32, guidance_embeds_scale=0.1)

    @property
    def device(self):
        return self.anchor.device

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, hidden_states, **arguments):
        return (torch.zeros_like(hidden_states),)

    def decode(self, latents, **arguments):
        height, width = latents.shape[2] * 32, latents.shape[3] * 32
        return (torch.linspace(-0.8, 0.8, height).reshape(1, 1, height, 1).expand(1, 3, height, width),)

    def enable_tiling(self, **arguments):
        self.tiling = arguments


class HDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = read_source(ROOT / "examples/aurora-hd.json")
        self.source["output_dir"] = str(self.directory / "new-run")

    def test_contract_accepts_only_native_hd_and_bounded_fictional_source(self):
        self.assertEqual(validate_source(self.source), self.source)
        for field, value in (
            ("schema", "other"), ("campaign", "Real brand"), ("fictional", 1), ("synthetic_adults", False),
            ("model", "https://untrusted/model"), ("steps", True), ("steps", 2.0), ("steps", 8),
            ("native_size", [1024, 576]), ("native_size", [1920, 1080]),
            ("native_size", [1920.0, 1088]), ("native_size", [3840, 2176]),
            ("output_size", [1280, 720]), ("variants", []), ("output_dir", ""),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                validate_source({**self.source, field: value})
        with self.assertRaises(ValueError):
            validate_source({**self.source, "trust_remote_code": True})
        for field, value in (("seed", True), ("seed", -1), ("seed", 2 ** 32), ("seed", 1.0),
                             ("prompt", " "), ("prompt", "🌌" * 1025), ("profile", "real-user")):
            source = copy.deepcopy(self.source)
            source["variants"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_source(source)
        source = copy.deepcopy(self.source)
        source["variants"][1] = source["variants"][0]
        with self.assertRaises(ValueError):
            validate_source(source)
        source["variants"].append(source["variants"][0])
        with self.assertRaises(ValueError):
            validate_source(source)

    def test_source_read_rejects_duplicates_and_oversized_json(self):
        path = self.directory / "source.json"
        path.write_text('{"schema":"one","schema":"two"}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON"):
            read_source(path)
        path.write_text(" " * 16385)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            read_source(path)

    def test_exact_crop_preserves_pixels_without_resize(self):
        rows = [bytes((row % 256, 41, 200)) * 1920 for row in range(1088)]
        native = Image.frombytes("RGB", NATIVE_SIZE, b"".join(rows))
        with patch.object(Image.Image, "resize", side_effect=AssertionError("resize forbidden")):
            cropped = crop_native(native)
        self.assertEqual(cropped.size, (1920, 1080))
        self.assertEqual(cropped.tobytes(), b"".join(rows[4:1084]))
        self.assertEqual(native.tobytes(), b"".join(rows))
        for size in ((1024, 1024), (1920, 1080), (1920, 1089), (1921, 1088)):
            with self.assertRaises(ValueError):
                crop_native(Image.new("RGB", size))
        with self.assertRaises(ValueError):
            crop_native(Image.new("RGBA", NATIVE_SIZE))

    def test_fixture_requires_flag_and_cannot_masquerade_as_model_evidence(self):
        pipeline = FixturePipeline()
        with self.assertRaisesRegex(ValueError, "test_only"):
            generate(self.source, device="cpu", pipeline=pipeline)
        with self.assertRaisesRegex(ValueError, "test_only"):
            generate(self.source, device="cpu", test_only=True)
        self.assertFalse(Path(self.source["output_dir"]).exists())
        with patch.object(Image.Image, "resize", side_effect=AssertionError("resize forbidden")):
            manifest = generate(self.source, device="cpu", pipeline=pipeline, test_only=True)
        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(manifest["test_only"])
        self.assertEqual(manifest["evidence"], "test-fixture-not-model-evidence")
        self.assertFalse(manifest["runtime"]["weights_verified"])
        self.assertFalse(manifest["model_quality_assessed"])
        self.assertEqual(manifest["model"]["revision"], REVISION)
        self.assertEqual(manifest["budget"]["selected_images"], 1)
        self.assertEqual(len(pipeline.calls), 1)
        call = pipeline.calls[0]
        self.assertIs(call["use_resolution_binning"], False)
        self.assertEqual(call["output_type"], "np")
        self.assertEqual((call["width"], call["height"]), NATIVE_SIZE)
        self.assertEqual(call["num_images_per_prompt"], 1)
        self.assertEqual(call["intermediate_timesteps"], 1.3)
        self.assertIsNone(call["complex_human_instruction"])
        self.assertEqual(call["generator"].initial_seed(), 1701)
        record = manifest["images"][0]
        self.assertEqual(record["prompt"], self.source["variants"][0]["prompt"])
        self.assertEqual(record["effective_prompt"], record["prompt"].lower().strip())
        self.assertEqual(record["latent_observations"][0]["shape"], list(LATENT_SHAPE))
        self.assertTrue(record["float_validation"]["finite"])
        self.assertFalse(record["float_validation"]["quality_assessment"])
        directory = Path(self.source["output_dir"])
        self.assertEqual(json.loads((directory / "manifest.json").read_text()), manifest)
        self.assertTrue(record["output"]["file"].startswith("TEST_ONLY-"))
        for kind in ("native", "output"):
            self.assertEqual(len(record[kind]["sha256"]), 64)
        with Image.open(directory / record["native"]["file"]) as native:
            with Image.open(directory / record["output"]["file"]) as cropped:
                self.assertEqual(native.size, NATIVE_SIZE)
                self.assertEqual(cropped.size, (1920, 1080))
                self.assertEqual(cropped.tobytes(), native.crop(CROP_BOX).tobytes())

    def test_float_validation_rejects_nan_inf_range_and_near_constant_before_png(self):
        pixels = np.linspace(0.1, 0.9, 1088, dtype=np.float32).reshape(1, 1088, 1, 1)
        pixels = np.broadcast_to(pixels, (1, 1088, 1920, 3)).copy()
        native, statistics = native_from_float(pixels)
        self.assertEqual(native.tobytes(), (pixels[0] * 255).round().astype(np.uint8).tobytes())
        self.assertTrue(statistics["finite"])
        for value in (float("nan"), float("inf"), float("-inf"), -0.01, 1.01):
            invalid = pixels.copy()
            invalid[0, 10, 20, 1] = value
            with self.subTest(value=value), patch.object(Image, "fromarray") as conversion:
                with self.assertRaises(ValueError):
                    native_from_float(invalid)
                conversion.assert_not_called()
        for value in (0.0, 1.0, 0.5):
            with self.assertRaisesRegex(ValueError, "near-constant"):
                native_from_float(np.full_like(pixels, value))
        solid_red = np.zeros_like(pixels)
        solid_red[..., 0] = 1
        with self.assertRaisesRegex(ValueError, "near-constant"):
            native_from_float(solid_red)
        with self.assertRaisesRegex(ValueError, "near-constant"):
            native_from_float(0.5 + pixels * 0.0001)
        for malformed in (pixels[0], pixels.astype(np.uint8), [native], pixels[:, :1024]):
            with self.assertRaises(ValueError):
                native_from_float(malformed)

    def test_budget_and_new_directory_fail_before_loading(self):
        for options in ({"max_images": 3}, {"max_images": True}, {"max_images": 0},
                        {"variant": "both"}, {"variant": "unknown"}):
            with patch("angelos_ml.hd.load_pipeline") as loader, self.assertRaises(ValueError):
                generate(self.source, device="cpu", **options)
            loader.assert_not_called()
        directory = Path(self.source["output_dir"])
        directory.mkdir()
        marker = directory / "keep.txt"
        marker.write_text("untouched")
        with patch("angelos_ml.hd.load_pipeline") as loader, self.assertRaises(FileExistsError):
            generate(self.source, device="cpu")
        loader.assert_not_called()
        self.assertEqual(marker.read_text(), "untouched")

    def test_two_images_four_steps_are_explicit_and_sequential(self):
        self.source["steps"] = 4
        pipeline = FixturePipeline()
        manifest = generate(self.source, device="cpu", variant="both", max_images=2,
                            pipeline=pipeline, test_only=True)
        self.assertEqual(manifest["budget"]["max_denoising_steps_total"], 8)
        self.assertEqual([call["generator"].initial_seed() for call in pipeline.calls], [1701, 1702])
        self.assertTrue(all(call["intermediate_timesteps"] is None for call in pipeline.calls))
        self.assertEqual(len(manifest["images"]), 2)

    def test_small_native_and_latent_images_fail_with_failed_manifest(self):
        for index, pipeline in enumerate((FixturePipeline(size=(1024, 1024)),
                                          FixturePipeline(latent_shape=(1, 32, 32, 32)),
                                          FixturePipeline(fail=True), FixturePipeline(tokens=301))):
            directory = self.directory / f"bad-{index}"
            with self.assertRaises((ValueError, RuntimeError)):
                generate(self.source, device="cpu", pipeline=pipeline, test_only=True, output_dir=directory)
            manifest = json.loads((directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["images"], [])
            self.assertLessEqual(len(pipeline.calls), 1)
            self.assertEqual(list(directory.glob("*.png")), [])

    def test_devices_explicit_and_no_mps_or_cuda_fallback(self):
        for device in (None, "auto", "metal", "cuda:arbitrary"):
            with self.assertRaises(ValueError):
                validate_device(device)
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}):
            with self.assertRaisesRegex(ValueError, "fallback"):
                validate_device("mps")
        with patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}):
            with patch("torch.backends.mps.is_available", return_value=False):
                with self.assertRaisesRegex(ValueError, "MPS is unavailable"):
                    validate_device("mps")
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                validate_device("cuda")

    def test_explicit_download_is_immutable_data_only(self):
        with patch("huggingface_hub.snapshot_download", return_value="cached") as download:
            self.assertEqual(download_model(MODEL_ID, "/cache"), "cached")
        arguments = download.call_args.kwargs
        self.assertEqual(arguments["repo_id"], REPOSITORY)
        self.assertEqual(arguments["revision"], REVISION)
        self.assertIs(arguments["local_files_only"], False)
        self.assertEqual(arguments["max_workers"], 2)
        for filename in ("pipeline.py", "transformer/evil.py", "text_encoder/pytorch_model.bin",
                         "script.sh", "model.pkl", "tokenizer/runner.js"):
            self.assertFalse(any(fnmatch.fnmatch(filename, pattern) for pattern in DOWNLOAD_PATTERNS))
        for filename in ("model_index.json", "text_encoder/model-00001-of-00002.safetensors",
                         "transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"):
            self.assertTrue(any(fnmatch.fnmatch(filename, pattern) for pattern in DOWNLOAD_PATTERNS))
        with patch("huggingface_hub.snapshot_download") as download, self.assertRaises(ValueError):
            download_model("main")
        download.assert_not_called()

    def test_loader_calls_real_api_with_pinned_metadata_offline_and_mixed_dtypes(self):
        snapshot = self.directory / REVISION
        snapshot.mkdir()
        (snapshot / "model_index.json").write_text(json.dumps(PINNED_MODEL_INDEX))
        pipeline = SimpleNamespace(
            transformer=FixtureComponent("transformer", torch.bfloat16),
            text_encoder=FixtureComponent("text_encoder", torch.bfloat16),
            vae=FixtureComponent("vae", torch.float32), scheduler=SCMScheduler(), to=Mock(),
        )
        with patch.dict(os.environ), patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download:
            with patch.object(SanaSprintPipeline, "from_pretrained", return_value=pipeline) as loader:
                loaded, metadata = load_pipeline(MODEL_ID, "cpu", str(self.directory))
        self.assertIs(loaded, pipeline)
        self.assertIs(download.call_args.kwargs["local_files_only"], True)
        self.assertEqual(download.call_args.kwargs["revision"], REVISION)
        self.assertEqual(loader.call_args.args, (str(snapshot),))
        options = loader.call_args.kwargs
        self.assertTrue(options["local_files_only"])
        self.assertTrue(options["use_safetensors"])
        self.assertFalse(options["trust_remote_code"])
        self.assertEqual(options["dtype"]["text_encoder"], torch.bfloat16)
        self.assertEqual(options["dtype"]["transformer"], torch.bfloat16)
        self.assertEqual(options["dtype"]["vae"], torch.float32)
        pipeline.to.assert_called_once_with("cpu")
        self.assertEqual(pipeline.vae.tiling["tile_sample_min_height"], 512)
        self.assertEqual(pipeline.vae.tiling["tile_sample_stride_width"], 448)
        self.assertEqual(metadata["model_index"], PINNED_MODEL_INDEX)
        self.assertEqual(metadata["components"]["vae"]["dtype"], "torch.float32")
        self.assertEqual(len(metadata["model_index_sha256"]), 64)

    def test_missing_cache_and_custom_metadata_never_download_or_load(self):
        with patch.dict(os.environ), patch("huggingface_hub.snapshot_download", side_effect=FileNotFoundError("missing")) as download:
            with patch.object(SanaSprintPipeline, "from_pretrained") as loader, self.assertRaises(FileNotFoundError):
                load_pipeline(MODEL_ID, "cpu")
        self.assertTrue(download.call_args.kwargs["local_files_only"])
        loader.assert_not_called()
        snapshot = self.directory / REVISION
        snapshot.mkdir()
        for index in ({**PINNED_MODEL_INDEX, "transformer": ["evil", "Code"]},
                      {**PINNED_MODEL_INDEX, "custom_pipeline": "evil"}):
            (snapshot / "model_index.json").write_text(json.dumps(index))
            with patch.dict(os.environ), patch("huggingface_hub.snapshot_download", return_value=str(snapshot)):
                with patch.object(SanaSprintPipeline, "from_pretrained") as loader, self.assertRaises(ValueError):
                    load_pipeline(MODEL_ID, "cpu")
            loader.assert_not_called()

    def test_loader_rejects_wrong_precision_without_fallback(self):
        snapshot = self.directory / REVISION
        snapshot.mkdir()
        (snapshot / "model_index.json").write_text(json.dumps(PINNED_MODEL_INDEX))
        pipeline = SimpleNamespace(
            transformer=FixtureComponent("transformer", torch.float16),
            text_encoder=FixtureComponent("text_encoder", torch.bfloat16),
            vae=FixtureComponent("vae", torch.float32), scheduler=SCMScheduler(), to=Mock(),
        )
        with patch.dict(os.environ), patch("huggingface_hub.snapshot_download", return_value=str(snapshot)):
            with patch.object(SanaSprintPipeline, "from_pretrained", return_value=pipeline) as loader:
                with self.assertRaisesRegex(ValueError, "device/dtype"):
                    load_pipeline(MODEL_ID, "cpu")
        self.assertEqual(loader.call_count, 1)
        pipeline.to.assert_called_once_with("cpu")

    def test_float_failure_is_recorded_and_partial_results_are_not_success(self):
        pipeline = FixturePipeline()
        call = Mock(side_effect=[pipeline, RuntimeError("second image failed")])

        def invoke(**arguments):
            result = call()
            return result(**arguments)

        injected = Mock(side_effect=invoke)
        injected.tokenizer = FixtureTokenizer()
        with self.assertRaisesRegex(RuntimeError, "second image"):
            generate(self.source, device="cpu", pipeline=injected, test_only=True, variant="both", max_images=2)
        directory = Path(self.source["output_dir"])
        manifest = json.loads((directory / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(len(manifest["images"]), 1)
        self.assertEqual(len(list(directory.glob("*.png"))), 2)
        self.assertEqual(call.call_count, 2)
        bad = FixturePipeline()
        original = bad.__class__.__call__

        def nonfinite(instance, **arguments):
            result = original(instance, **arguments)
            result.images[0, 0, 0, 0] = float("nan")
            return result

        failed_directory = self.directory / "nonfinite"
        with patch.object(FixturePipeline, "__call__", nonfinite), self.assertRaisesRegex(ValueError, "Nonfinite"):
            generate(self.source, device="cpu", pipeline=bad, test_only=True, output_dir=failed_directory)
        failed = json.loads((failed_directory / "manifest.json").read_text())
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(list(failed_directory.glob("*.png")), [])

    def test_installed_api_and_cached_real_metadata_match_audited_contract(self):
        import diffusers

        self.assertEqual(diffusers.__version__, DIFFUSERS_VERSION)
        call = inspect.signature(SanaSprintPipeline.__call__).parameters
        self.assertTrue(call["use_resolution_binning"].default)
        self.assertEqual(call["intermediate_timesteps"].default, 1.3)
        self.assertIn("tile_sample_min_height", inspect.signature(AutoencoderDC.enable_tiling).parameters)
        self.assertEqual({name: PINNED_MODEL_INDEX[name] for name in COMPONENTS}, COMPONENTS)
        snapshot = Path(default_cache_dir()) / ("models--" + REPOSITORY.replace("/", "--")) / "snapshots" / REVISION
        if (snapshot / "model_index.json").is_file():
            self.assertEqual(json.loads((snapshot / "model_index.json").read_text()), PINNED_MODEL_INDEX)

    def test_real_official_pipeline_and_scheduler_control_flow_with_fixture_components(self):
        for steps in (2, 4):
            pipeline = SanaSprintPipeline(
                tokenizer=FixtureTokenizer(3),
                text_encoder=FixtureComponent("text_encoder", torch.bfloat16),
                transformer=FixtureComponent("transformer", torch.bfloat16),
                vae=FixtureComponent("vae", torch.float32), scheduler=SCMScheduler(),
            )
            pipeline.set_progress_bar_config(disable=True)
            self.source["steps"] = steps
            with patch.object(pipeline, "encode_prompt", return_value=(torch.zeros(1, 3, 2304), torch.ones(1, 3))):
                with patch.object(pipeline.image_processor, "resize_and_crop_tensor", side_effect=AssertionError("no resize")):
                    with patch.object(pipeline.image_processor, "classify_height_width_bin", side_effect=AssertionError("no binning")):
                        manifest = generate(self.source, device="cpu", pipeline=pipeline, test_only=True,
                                            output_dir=self.directory / f"real-api-{steps}")
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["test_only"])
            self.assertEqual(len(manifest["images"][0]["latent_observations"]), steps)
            self.assertIn("SanaSprintPipeline", manifest["runtime"]["pipeline"])

    def test_cli_list_and_wrapper_are_no_inference_and_device_is_required(self):
        with patch("angelos_ml.hd.load_pipeline") as loader, patch("angelos_ml.hd.download_model") as download:
            with redirect_stdout(io.StringIO()) as output:
                main(["list"])
        loader.assert_not_called()
        download.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())[0], model_metadata())
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["generate", str(ROOT / "examples/aurora-hd.json")])
        self.assertEqual(error.exception.code, 2)
        result = subprocess.run(["node", "scripts/hd.js", "list"], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)[0]["revision"], REVISION)


if __name__ == "__main__":
    unittest.main()
