"""Pinned Sana Sprint catalog, explicit download, and offline official loader."""

import hashlib
import json
import os
from pathlib import Path


MODEL_ID = "sana-sprint-0.6b"
REPOSITORY = "Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers"
REVISION = "aa76e7f4f4928f378716b6716a2130fba3caf5b1"
DIFFUSERS_VERSION = "0.40.0"
COMPONENTS = {
    "scheduler": ["diffusers", "SCMScheduler"],
    "text_encoder": ["transformers", "Gemma2Model"],
    "tokenizer": ["transformers", "GemmaTokenizerFast"],
    "transformer": ["diffusers", "SanaTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderDC"],
}
DOWNLOAD_PATTERNS = (
    "model_index.json", "README.md", "LICENSE", "scheduler/scheduler_config.json",
    "tokenizer/tokenizer.json", "tokenizer/tokenizer.model",
    "tokenizer/tokenizer_config.json", "tokenizer/special_tokens_map.json",
    "text_encoder/config.json", "text_encoder/model.safetensors.index.json",
    "text_encoder/model*.safetensors", "transformer/config.json",
    "transformer/diffusion_pytorch_model*.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json", "vae/config.json",
    "vae/diffusion_pytorch_model*.safetensors", "vae/diffusion_pytorch_model.safetensors.index.json",
)


def model_metadata(model=MODEL_ID):
    if model != MODEL_ID:
        raise ValueError(f"Only {MODEL_ID} is allowed; repositories and revisions are not source inputs")
    return {
        "id": MODEL_ID, "repository": REPOSITORY, "revision": REVISION,
        "pipeline": "diffusers.SanaSprintPipeline", "diffusers_version": DIFFUSERS_VERSION,
        "transformer_parameters_approx": "0.6B", "text_encoder_parameters_approx": "2.2B",
        "text_encoder": "Gemma2Model", "vae": "AutoencoderDC (DC-AE, 32x compression)",
        "whole_pipeline_is_0_6b": False, "pretrained": True, "campaign_trained": False,
        "weights_format": "safetensors", "trust_remote_code": False,
        "license_note": "Model card: Apache-2.0; Gemma terms and prohibited-use policy also apply",
    }


def default_cache_dir():
    home = Path(os.environ.get("HF_HOME", Path(__file__).resolve().parents[2] / "artifacts/huggingface"))
    return os.environ.get("HF_HUB_CACHE", str(home / "hub"))


def download_model(model=MODEL_ID, cache_dir=None):
    model_metadata(model)
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=REPOSITORY, revision=REVISION, cache_dir=cache_dir or default_cache_dir(),
        allow_patterns=list(DOWNLOAD_PATTERNS), local_files_only=False, max_workers=2,
    )


def validate_device(device):
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("An explicit device is required: cuda, mps, or cpu; no automatic fallback")
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("Unset PYTORCH_ENABLE_MPS_FALLBACK: CPU operator fallback is forbidden")
    import torch

    if device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA with bfloat16 support is required; no fallback")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; no fallback")
    torch.empty(1, device=device, dtype=torch.bfloat16)


def load_pipeline(model, device, cache_dir=None):
    model_metadata(model)
    validate_device(device)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import diffusers
    import torch
    from diffusers import SanaSprintPipeline
    from huggingface_hub import snapshot_download

    if diffusers.__version__ != DIFFUSERS_VERSION:
        raise ValueError(f"This audited baseline requires diffusers=={DIFFUSERS_VERSION}")
    snapshot = Path(snapshot_download(
        repo_id=REPOSITORY, revision=REVISION, cache_dir=cache_dir or default_cache_dir(),
        allow_patterns=list(DOWNLOAD_PATTERNS), local_files_only=True,
    ))
    if snapshot.name != REVISION:
        raise ValueError("Cache snapshot is not the immutable revision")
    index_bytes = (snapshot / "model_index.json").read_bytes()
    index = json.loads(index_bytes)
    expected_keys = set(COMPONENTS) | {"_class_name", "_diffusers_version"}
    if (set(index) != expected_keys or index["_class_name"] != "SanaSprintPipeline"
            or any(index.get(name) != value for name, value in COMPONENTS.items())):
        raise ValueError("Unexpected pinned model_index.json; custom components are forbidden")
    pipeline = SanaSprintPipeline.from_pretrained(
        str(snapshot), revision=REVISION, local_files_only=True,
        use_safetensors=True, trust_remote_code=False,
        dtype={"default": torch.bfloat16, "transformer": torch.bfloat16,
               "text_encoder": torch.bfloat16, "vae": torch.float32},
    )
    pipeline.to(device)
    pipeline.vae.enable_tiling(
        tile_sample_min_height=512, tile_sample_min_width=512,
        tile_sample_stride_height=448, tile_sample_stride_width=448,
    )
    observed = {}
    for name, dtype in (("transformer", torch.bfloat16), ("text_encoder", torch.bfloat16), ("vae", torch.float32)):
        component = getattr(pipeline, name)
        component.eval()
        parameters = list(component.parameters())
        if not parameters or any(parameter.device.type != device or parameter.dtype != dtype for parameter in parameters):
            raise ValueError(f"{name} does not match requested device/dtype; no offload or fallback")
        observed[name] = {"device": str(component.device), "dtype": str(component.dtype)}
    return pipeline, {
        "snapshot": str(snapshot), "model_index": index,
        "model_index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "components": observed, "scheduler_config": dict(pipeline.scheduler.config),
        "local_files_only": True, "use_safetensors": True, "trust_remote_code": False,
        "vae_tiling": {"height": 512, "width": 512, "stride_height": 448, "stride_width": 448},
    }
