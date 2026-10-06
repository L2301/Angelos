"""Immutable model allowlist and explicit, separate download command."""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class ModelSpec:
    id: str
    repo: str
    revision: str
    pipeline: str
    kind: str
    license: str
    safetensors: bool = True


MODELS = MappingProxyType({
    "tiny-sd": ModelSpec(
        "tiny-sd", "segmind/tiny-sd", "cad0bd7495fa6c4bcca01b19a723dc91627fe84f",
        "StableDiffusionPipeline", "image", "creativeml-openrail-m", False,
    ),
    "flux2-klein-4b": ModelSpec(
        "flux2-klein-4b", "black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "Flux2KleinPipeline", "image", "apache-2.0",
    ),
    "ltx-video": ModelSpec(
        "ltx-video", "Lightricks/LTX-Video", "8984fa25007f376c1a299016d0957a37a2f797bb",
        "LTXPipeline", "video", "ltx-video-open-weights-license",
    ),
})


def model_spec(model):
    if not isinstance(model, str) or model not in MODELS:
        raise ValueError(f"Model must be one of {', '.join(MODELS)}; arbitrary repositories and revisions are forbidden")
    return MODELS[model]


def model_metadata(model):
    return asdict(model_spec(model))


def default_cache_dir():
    home = Path(os.environ.get("HF_HOME", Path(__file__).resolve().parents[2] / "artifacts" / "huggingface"))
    return os.environ.get("HF_HUB_CACHE", str(home / "hub"))


def download_model(model, cache_dir=None):
    specification = model_spec(model)
    from huggingface_hub import snapshot_download

    patterns = ["*.json", "*.txt", "*.model", "*.tiktoken", "README.md", "LICENSE*", "*LICENSE*", "*license*"]
    patterns.append("*/*.safetensors" if specification.safetensors else "*/*.bin")
    return snapshot_download(
        repo_id=specification.repo, revision=specification.revision,
        cache_dir=cache_dir or default_cache_dir(), allow_patterns=patterns, local_files_only=False,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    download = commands.add_parser("download", aliases=["model-download"], help="Explicitly download an allowlisted immutable snapshot; never runs inference")
    download.add_argument("model", choices=tuple(MODELS))
    download.add_argument("--cache-dir")
    arguments = parser.parse_args(argv)
    if arguments.command == "list":
        print(json.dumps([model_metadata(model) for model in MODELS], indent=2))
        return
    try:
        path = download_model(arguments.model, arguments.cache_dir)
    except Exception as error:
        parser.exit(2, f"Model download failed: {error}\n")
    print(json.dumps({"model": model_metadata(arguments.model), "snapshot": str(path), "inference": False}))


if __name__ == "__main__":
    main()
