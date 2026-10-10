"""Bounded local Sana Sprint baseline: native 1920x1088, crop-only 1920x1080."""

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .hd_models import MODEL_ID, download_model, load_pipeline, model_metadata, validate_device


SCHEMA = "angelos.hd.v1"
NATIVE_SIZE = (1920, 1088)
OUTPUT_SIZE = (1920, 1080)
CROP_BOX = (0, 4, 1920, 1084)
LATENT_SHAPE = (1, 32, 34, 60)
MAX_SOURCE_BYTES = 16384
MIN_SPATIAL_VARIANCE = 1e-6


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def validate_source(source):
    if isinstance(source, dict) and any(field in source for field in ("upl", "encoder", "uplBundle")):
        raise ValueError("HD baseline is prompt-only; UPL conditioning requires a separately trained adapter, not ignored vectors")
    fields = {"schema", "campaign", "fictional", "synthetic_adults", "model", "native_size",
              "output_size", "steps", "variants", "output_dir"}
    if not isinstance(source, dict) or set(source) != fields:
        raise ValueError(f"Source requires exactly: {', '.join(sorted(fields))}")
    if (source["schema"] != SCHEMA or source["campaign"] != "Aurora"
            or source["fictional"] is not True or source["synthetic_adults"] is not True):
        raise ValueError("Only fictional Aurora with synthetic adult demo profiles is supported")
    model_metadata(source["model"])
    for name, expected in (("native_size", NATIVE_SIZE), ("output_size", OUTPUT_SIZE)):
        value = source[name]
        if not isinstance(value, list) or any(type(size) is not int for size in value) or value != list(expected):
            raise ValueError(f"{name} must be {list(expected)} [width, height]; resizing is forbidden")
    if type(source["steps"]) is not int or source["steps"] not in (2, 4):
        raise ValueError("steps must be 2 or 4")
    variants = source["variants"]
    if not isinstance(variants, list) or not 1 <= len(variants) <= 2:
        raise ValueError("Provide one or two quiet/bold variants, not a prompt/seed sweep")
    names = set()
    for variant in variants:
        if not isinstance(variant, dict) or set(variant) != {"profile", "prompt", "seed"}:
            raise ValueError("Each variant requires exactly profile, prompt, seed")
        profile = variant["profile"]
        if not isinstance(profile, str) or profile not in ("quiet", "bold") or profile in names:
            raise ValueError("Profiles must be unique quiet/bold synthetic adult labels")
        names.add(profile)
        prompt = variant["prompt"]
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode("utf-8")) > 4096:
            raise ValueError("prompt must be nonempty UTF-8 text of at most 4096 bytes")
        if type(variant["seed"]) is not int or not 0 <= variant["seed"] <= 2 ** 32 - 1:
            raise ValueError("seed must be an integer in [0, 4294967295]")
    output_dir = source["output_dir"]
    if not isinstance(output_dir, str) or not output_dir.strip() or "\0" in output_dir:
        raise ValueError("output_dir must name a NEW directory")
    return copy.deepcopy(source)


def read_source(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_SOURCE_BYTES + 1)
    if len(payload) > MAX_SOURCE_BYTES:
        raise ValueError("HD source exceeds 16384 bytes")
    return validate_source(json.loads(payload.decode("utf-8"), object_pairs_hook=unique_fields))


def crop_native(image):
    from PIL import Image

    if not isinstance(image, Image.Image) or image.size != NATIVE_SIZE or image.mode != "RGB":
        raise ValueError("Expected raw native RGB 1920x1088; small, resized, or malformed output is rejected")
    return image.crop(CROP_BOX)


def native_from_float(images):
    import numpy as np
    from PIL import Image

    if (not isinstance(images, np.ndarray) or images.shape != (1, 1088, 1920, 3)
            or not np.issubdtype(images.dtype, np.floating)):
        raise ValueError("Expected native floating output [1,1088,1920,3], not PIL or smaller images")
    pixels = images[0]
    if not np.isfinite(pixels).all():
        raise ValueError("Nonfinite native float output; refusing NaN/Inf-to-black PNG conversion")
    minimum, maximum = float(pixels.min()), float(pixels.max())
    if minimum < 0 or maximum > 1:
        raise ValueError("Native floating output must be in [0,1]; no corrective clipping")
    variances = np.var(pixels, axis=(0, 1), dtype=np.float64)
    if float(variances.mean()) <= MIN_SPATIAL_VARIANCE:
        raise ValueError("Degenerate near-constant native output; spatial variance is too small")
    native = Image.fromarray((pixels * 255).round().astype(np.uint8))
    return native, {
        "shape": list(images.shape), "dtype": str(images.dtype), "finite": True,
        "minimum": minimum, "maximum": maximum, "channel_spatial_variance": variances.tolist(),
        "minimum_mean_spatial_variance": MIN_SPATIAL_VARIANCE,
        "quantization": "round(float_pixels * 255) to uint8; no clipping, resizing, or normalization",
        "quality_assessment": False,
    }


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_manifest(directory, manifest):
    temporary = directory / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "manifest.json")


def generate(source, *, device, variant="quiet", max_images=1, output_dir=None, cache_dir=None,
             pipeline=None, test_only=False):
    source = validate_source(source)
    if type(max_images) is not int or max_images not in (1, 2):
        raise ValueError("max_images must be 1 (default) or 2 (hard cap)")
    if variant not in ("quiet", "bold", "both"):
        raise ValueError("variant must be quiet, bold, or both")
    selected = [item for item in source["variants"] if variant == "both" or item["profile"] == variant]
    if not selected or len(selected) > max_images:
        raise ValueError("Selection is empty or exceeds max_images; both variants require --max-images 2")
    if type(test_only) is not bool or (pipeline is not None) != test_only:
        raise ValueError("Injected pipelines require explicit test_only=True; fixtures are never model evidence")
    validate_device(device)
    directory = Path(output_dir if output_dir is not None else source["output_dir"]).absolute()
    directory.mkdir(parents=True, exist_ok=False)
    settings = {
        "width": NATIVE_SIZE[0], "height": NATIVE_SIZE[1], "use_resolution_binning": False,
        "num_inference_steps": source["steps"], "num_images_per_prompt": 1,
        "guidance_scale": 4.5, "max_timesteps": 1.5708,
        "intermediate_timesteps": 1.3 if source["steps"] == 2 else None,
        "timesteps": None, "eta": 0.0, "max_sequence_length": 300,
        "complex_human_instruction": None, "clean_caption": False,
        "output_type": "np", "return_dict": True, "attention_kwargs": None,
        "callback_on_step_end_tensor_inputs": ["latents"],
    }
    source_hash = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "schema": SCHEMA, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source, "source_sha256": source_hash, "output_dir": str(directory),
        "model": model_metadata(source["model"]), "device_requested": device,
        "test_only": test_only, "evidence": "test-fixture-not-model-evidence" if test_only else "pretrained-local-baseline",
        "model_quality_assessed": False, "campaign_trained": False, "webgpu": False,
        "settings": settings, "generator_device": "cpu", "generator_note": "CPU noise RNG only, not model offload",
        "budget": {"max_images": max_images, "selected_images": len(selected),
                   "max_denoising_steps_total": len(selected) * source["steps"], "batch_size": 1},
        "native_shape": [1088, 1920, 3], "output_shape": [1080, 1920, 3], "shape_order": "HWC",
        "crop": {"box_xyxy": list(CROP_BOX), "top_rows": 4, "bottom_rows": 4, "resize": False},
        "runtime": None, "images": [],
    }
    started = time.monotonic()
    write_manifest(directory, manifest)
    try:
        import torch

        manifest["versions"] = {name: importlib.metadata.version(name) for name in
                                ("torch", "diffusers", "transformers", "safetensors", "Pillow", "huggingface-hub")}
        if not test_only:
            pipeline, manifest["runtime"] = load_pipeline(source["model"], device, cache_dir)
        else:
            manifest["runtime"] = {"pipeline": f"{type(pipeline).__module__}.{type(pipeline).__name__}",
                                   "weights_verified": False, "warning": "TEST FIXTURE: NOT PRETRAINED MODEL EVIDENCE"}
        for item in selected:
            effective_prompt = item["prompt"].lower().strip()
            token_count = len(pipeline.tokenizer.encode(effective_prompt, add_special_tokens=True))
            if token_count > settings["max_sequence_length"]:
                raise ValueError("Prompt exceeds 300 tokens; implicit truncation is forbidden")
            observations = []

            def observe_latents(current_pipeline, step, timestep, callback_kwargs):
                latents = callback_kwargs["latents"]
                if tuple(latents.shape) != LATENT_SHAPE or latents.device.type != device:
                    raise ValueError("Native latent shape/device mismatch; lower-resolution generation is forbidden")
                observations.append({"step": step, "shape": list(latents.shape),
                                     "dtype": str(latents.dtype), "device": str(latents.device)})
                return callback_kwargs

            generator = torch.Generator(device="cpu").manual_seed(item["seed"])
            with torch.inference_mode():
                result = pipeline(prompt=item["prompt"], generator=generator,
                                  callback_on_step_end=observe_latents, **settings)
            if len(observations) != source["steps"]:
                raise ValueError("Missing native latent observations for the requested denoising steps")
            if not hasattr(result, "images"):
                raise ValueError("Expected native float images from the pipeline")
            native, float_validation = native_from_float(result.images)
            cropped = crop_native(native)
            stem = ("TEST_ONLY-" if test_only else "") + f"{item['profile']}-{item['seed']}"
            native_path = directory / f"{stem}-native-1920x1088.png"
            cropped_path = directory / f"{stem}-1920x1080.png"
            for path, image in ((native_path, native), (cropped_path, cropped)):
                with path.open("xb") as stream:
                    image.save(stream, format="PNG")
            manifest["images"].append({
                **item, "effective_prompt": effective_prompt, "prompt_tokens": token_count,
                "float_validation": float_validation,
                "latent_observations": observations, "native_shape": [native.height, native.width, 3],
                "output_shape": [cropped.height, cropped.width, 3],
                "native": {"file": native_path.name, "sha256": sha256_file(native_path)},
                "output": {"file": cropped_path.name, "sha256": sha256_file(cropped_path)},
            })
            write_manifest(directory, manifest)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        write_manifest(directory, manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Show the pin; no downloads or inference")
    download = commands.add_parser("download", help="Explicit official checkpoint download only; no inference")
    download.add_argument("model", choices=(MODEL_ID,))
    download.add_argument("--cache-dir")
    run = commands.add_parser("generate", help="Offline local inference only; NEW output directory required")
    run.add_argument("source")
    run.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    run.add_argument("--variant", choices=("quiet", "bold", "both"), default="quiet")
    run.add_argument("--max-images", type=int, choices=(1, 2), default=1)
    run.add_argument("--output-dir")
    run.add_argument("--cache-dir")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "list":
            result = [model_metadata()]
        elif arguments.command == "download":
            result = {"model": model_metadata(arguments.model),
                      "snapshot": str(download_model(arguments.model, arguments.cache_dir)), "inference": False}
        else:
            with contextlib.redirect_stdout(sys.stderr):
                result = generate(read_source(arguments.source), device=arguments.device,
                                  variant=arguments.variant, max_images=arguments.max_images,
                                  output_dir=arguments.output_dir, cache_dir=arguments.cache_dir)
    except Exception as error:
        parser.exit(2, f"HD {arguments.command} failed ({type(error).__name__}): {error}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
