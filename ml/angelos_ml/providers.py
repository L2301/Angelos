"""Local-only Diffusers adapters. Injected pipelines are explicitly test-only."""

import hashlib
from importlib.metadata import version

import torch

from .models import default_cache_dir, model_metadata, model_spec


MAX_SEED = 2 ** 53 - 1


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
        raise ValueError("Execution seed must be a nonnegative JavaScript-safe integer")
    return seed


def node_seed(seed, node_id):
    validate_seed(seed)
    digest = hashlib.sha256(f"angelos.teacher-seed.v1:{seed}:{node_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def require_safe_loaders():
    from packaging.version import Version

    for package, minimum in (("torch", "2.6.0"), ("diffusers", "0.40.0"), ("transformers", "5.18.0")):
        if Version(version(package)) < Version(minimum):
            raise RuntimeError(f"{package}>={minimum} is required for the pinned adapters and safe weights_only loading")


def load_pipeline(model, device="cpu", dtype=None, cache_dir=None):
    specification = model_spec(model)
    require_safe_loaders()
    import diffusers

    pipeline_class = getattr(diffusers, specification.pipeline)
    try:
        pipeline = pipeline_class.from_pretrained(
            specification.repo, revision=specification.revision,
            local_files_only=True, cache_dir=cache_dir or default_cache_dir(),
            torch_dtype=dtype or torch.float32, use_safetensors=specification.safetensors,
            trust_remote_code=False,
        )
        if model == "tiny-sd":
            pipeline.enable_attention_slicing()
        return pipeline.to(device)
    except Exception as error:
        raise RuntimeError(
            f"Cannot load pinned local model {model}: {error}. "
            f"Run python -m angelos_ml.models download {model} separately "
            "(with the same --cache-dir, if used); check dependencies, license/access and device memory. "
            "Runtime never downloads models and has no procedural fallback."
        ) from error


class PipelineRegistry:
    def __init__(self, device="cpu", dtype=None, cache_dir=None, pipelines=None):
        self.device = str(torch.device(device))
        self.dtype = dtype or torch.float32
        self.cache_dir = cache_dir
        self._pipelines = dict(pipelines or {})
        self.injected_models = frozenset(self._pipelines)
        for model in self._pipelines:
            model_spec(model)

    def get(self, model):
        model_spec(model)
        if model not in self._pipelines:
            self._pipelines[model] = load_pipeline(model, self.device, self.dtype, self.cache_dir)
        return self._pipelines[model]

    def metadata(self, model):
        specification = model_spec(model)
        return {
            **model_metadata(model),
            "backend": "injected-test-pipeline" if model in self.injected_models else "diffusers",
            "device": self.device, "dtype": str(self.dtype),
            "weightLoading": "safetensors" if specification.safetensors else "torch.load(weights_only=True) via modern Diffusers/Transformers",
            "localFilesOnly": True,
        }

    @torch.inference_mode()
    def generate(self, model, prompt, parameters, seed):
        specification = model_spec(model)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
            raise ValueError("Derived generator seed must be a nonnegative signed 64-bit integer")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        arguments = {
            "prompt": prompt, "generator": generator,
            "num_inference_steps": int(parameters["steps"]),
            "guidance_scale": parameters["guidance"],
            "width": int(parameters["width"]), "height": int(parameters["height"]),
            "output_type": "pt",
        }
        if specification.kind == "video":
            arguments["num_frames"] = int(parameters["num_frames"])
        result = self.get(model)(**arguments)
        if specification.kind == "image":
            rejected = getattr(result, "nsfw_content_detected", None)
            if rejected is not None and any(rejected):
                raise RuntimeError("Model safety checker rejected the generated image; no replacement is emitted")
            pixels = result.images
            expected = (1, 3, arguments["height"], arguments["width"])
        else:
            pixels = result.frames
            expected = (1, arguments["num_frames"], 3, arguments["height"], arguments["width"])
        if not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != expected:
            raise ValueError(f"{model} returned an invalid raster shape; expected {expected}")
        pixels = pixels.detach().to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(pixels).all() or ((pixels < 0) | (pixels > 1)).any():
            raise ValueError(f"{model} returned non-finite or out-of-range pixels")
        return pixels[0].contiguous()
