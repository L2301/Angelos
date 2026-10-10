"""Generate full-graph synthetic adult teacher images, without training."""

import argparse
import copy
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .graph import graph_hash, load_graph, verify_graph_hash
from .providers import PipelineRegistry, validate_seed
from .teacher import RasterTeacher, pixel_coordinates
from .diffusion import LEGACY_ENCODER, conditioning_encoder


DEFAULT_TRAIN_COUNT = 16
DEFAULT_HELDOUT_COUNT = 4
DEFAULT_SEEDS_PER_PROFILE = 2


def split_profiles(seed, train_count, heldout_count):
    validate_seed(seed)
    for count in (train_count, heldout_count):
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10000:
            raise ValueError("Each split must have between one and 10000 distinct profiles")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    profiles = []
    seen = set()
    while len(profiles) < train_count + heldout_count:
        candidate = torch.cat((
            torch.randint(0, 6, (1,), generator=generator).float() / 5,
            torch.randint(0, 4, (1,), generator=generator).float() / 3,
            torch.rand(2, generator=generator),
        ))
        key = tuple(candidate.tolist())
        if key not in seen:
            seen.add(key)
            profiles.append(candidate)
    return torch.stack(profiles), np.array([0] * train_count + [1] * heldout_count, dtype=np.int64)


def build_dataset(graph, *, train_count=DEFAULT_TRAIN_COUNT, heldout_count=DEFAULT_HELDOUT_COUNT, seed=7, teacher_seed=11, seeds_per_profile=DEFAULT_SEEDS_PER_PROFILE, registry=None, width=256, height=256, preview_dir=None):
    """Default to 40 images: 16 train and 4 held-out profiles, two seeds each."""
    if conditioning_encoder(graph.get("encoder")) != LEGACY_ENCODER:
        raise ValueError("Teacher profile synthesis is legacy-only; supply a frozen UPLv2 dataset with encoded vectors and input bundles")
    identity = graph_hash(graph)
    validate_seed(teacher_seed)
    pixel_coordinates(width, height)
    if width < 256 or height < 256:
        raise ValueError("Teacher rasters must be at least 256 pixels; student images are downsampled only after the full graph")
    if isinstance(seeds_per_profile, bool) or not isinstance(seeds_per_profile, int) or not 1 <= seeds_per_profile <= 100:
        raise ValueError("seeds_per_profile must be an integer in [1,100]")
    profiles, splits = split_profiles(seed, train_count, heldout_count)
    validate_seed(teacher_seed + len(profiles) * seeds_per_profile - 1)
    teacher = RasterTeacher(graph, registry=registry)
    if preview_dir is not None:
        from PIL import Image

        preview_dir = Path(preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=False)
    images, upl, labels, seeds, samples = [], [], [], [], []
    for profile_index, profile in enumerate(profiles):
        for variation in range(seeds_per_profile):
            sample_seed = teacher_seed + profile_index * seeds_per_profile + variation
            native = teacher.render(profile, seed=sample_seed, width=width, height=height)
            reduced = functional.interpolate(native, size=(32, 32), mode="area")
            images.append(reduced[0].detach().cpu().numpy())
            upl.append(profile.numpy())
            labels.append(int(splits[profile_index]))
            seeds.append(sample_seed)
            samples.append({
                "index": len(samples), "profileIndex": profile_index,
                "upl": profile.tolist(), "split": int(splits[profile_index]),
                "nativeShape": list(native.shape[1:]), **teacher.last_provenance[0],
            })
            if preview_dir is not None:
                preview = preview_dir / f"sample-{len(samples) - 1:06d}.png"
                pixels = (native[0].detach().cpu().permute(1, 2, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                Image.fromarray(pixels).save(preview)
                samples[-1]["preview"] = str(preview)
    arrays = {
        "images": np.stack(images).astype(np.float32),
        "upl": np.stack(upl).astype(np.float32),
        "split": np.asarray(labels, dtype=np.int64),
        "teacher_seed": np.asarray(seeds, dtype=np.int64),
    }
    model_ids = sorted({node["params"]["model"] for node in graph["nodes"] if node["op"] in {"generate_text", "generate_image", "generate_video"}})
    models = [teacher.registry.metadata(model) for model in model_ids]
    source = "injected-test-pipeline" if any(model["backend"] == "injected-test-pipeline" for model in models) else "diffusers-local-open-weights" if models else "deterministic-raster"
    if source == "diffusers-local-open-weights" and any(model["kind"] == "text" for model in models):
        source = "transformers-diffusers-local-open-weights"
    packages = {}
    for package in ("torch", "diffusers", "transformers", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    manifest = {
        "format": "angelos.teacher-dataset.v1", "graphHash": identity,
        "encoder": copy.deepcopy(graph["encoder"]), "graph": copy.deepcopy(graph),
        "models": models, "samples": samples, "source": source, "synthetic": True,
        "deterministicTensorDevice": samples[0]["deterministicTensorDevice"],
        "profiles": "synthetic-adult-structured-demo-v1", "training": False,
        "profileSeed": seed, "teacherSeed": teacher_seed, "seedDerivation": "angelos.teacher-seed.v1: SHA256(namespace:teacherSeed:nodeId), first 8 bytes big-endian modulo 2^63",
        "splitPolicy": "unique UPL profiles split before generation; all seeds for each profile stay in its split",
        "trainProfiles": train_count, "heldoutProfiles": heldout_count, "seedsPerProfile": seeds_per_profile,
        "images": {"shape": list(arrays["images"].shape), "dtype": "float32", "range": [0, 1], "downsample": "area after complete native-resolution graph, including brand"},
        "packages": packages,
    }
    return arrays, manifest


def dataset_paths(output):
    output = Path(output)
    if output.suffix != ".npz":
        raise ValueError("Dataset output must have a .npz suffix")
    manifest = output.with_suffix(".manifest.json")
    if output.exists() or manifest.exists():
        raise ValueError("Dataset or manifest already exists; choose a new output path")
    return output, manifest


def write_dataset(output, arrays, manifest):
    output, sidecar = dataset_paths(output)
    encoded = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return output, sidecar


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, help="Validated graph JSON from the source compiler")
    parser.add_argument("--graph-hash")
    parser.add_argument("--graph-canonical")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument("--heldout-count", type=int, default=DEFAULT_HELDOUT_COUNT)
    parser.add_argument("--seed", type=int, default=7, help="UPL split seed")
    parser.add_argument("--teacher-seed", type=int, default=11)
    parser.add_argument("--seeds-per-profile", type=int, default=DEFAULT_SEEDS_PER_PROFILE)
    parser.add_argument("--width", type=int, default=256, help="Native resolution for deterministic-only graphs; generation nodes own their resolution")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--cache-dir")
    parser.add_argument("--save-previews", help="New directory for native-resolution full-graph PNGs; no overwrites")
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args(argv)
    try:
        dataset_paths(arguments.output)
        if not 1 <= arguments.threads <= 8:
            raise ValueError("--threads must be in [1,8]")
        torch.set_num_threads(arguments.threads)
        graph = load_graph(arguments.graph)
        if arguments.graph_canonical and not arguments.graph_hash:
            raise ValueError("--graph-canonical requires --graph-hash")
        if arguments.graph_hash:
            verify_graph_hash(graph, arguments.graph_hash, arguments.graph_canonical)
        arrays, manifest = build_dataset(
            graph, train_count=arguments.train_count, heldout_count=arguments.heldout_count,
            seed=arguments.seed, teacher_seed=arguments.teacher_seed, seeds_per_profile=arguments.seeds_per_profile,
            width=arguments.width, height=arguments.height,
            preview_dir=arguments.save_previews,
            registry=PipelineRegistry(device=arguments.device, dtype=getattr(torch, arguments.dtype), cache_dir=arguments.cache_dir),
        )
        output, sidecar = write_dataset(arguments.output, arrays, manifest)
    except Exception as error:
        parser.exit(2, f"Teacher dataset failed ({type(error).__name__}): {error}\nNo procedural fallback or training was performed.\n")
    print(json.dumps({"output": str(output), "manifest": str(sidecar), "graphHash": manifest["graphHash"], "samples": len(arrays["images"]), "source": manifest["source"], "training": False}))


if __name__ == "__main__":
    main()
