"""Bounded SanaSprint SCM reward backpropagation; not likelihood-ratio RL.

Only q/v LoRA parameters are optimized. The supplied pipeline is restored after
the experiment. No loading, downloads, device fallback, or promotion occurs.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn import functional

from .hd_models import DIFFUSERS_VERSION, REPOSITORY, REVISION
from .rewards import RewardBundle, export_bundle
from .diffusion import LEGACY_ENCODER, conditioning_tensor
from .feedback import frozen_upl


FORMAT = "angelos.hd-alignment.v1"
ADAPTER_FORMAT = "angelos.sana-qv-lora.v1"


@dataclass(frozen=True)
class AlignmentConfig:
    steps: int = 4
    sampling_steps: int = 2
    rank: int = 2
    blocks: int = 2
    learning_rate: float = 0.001
    uncertainty_scale: float = 1.0
    quality_floor: float = 0.6
    brand_floor: float = 0.8
    quality_penalty: float = 2.0
    brand_penalty: float = 2.0
    latent_anchor: float = 0.1
    image_anchor: float = 0.1
    adapter_l2: float = 0.001
    parameter_bound: float = 0.5
    minimum_click_gain: float = 0.001
    fixture_only: bool = False
    fixture_size: int = 32

    def validate(self):
        for name, minimum, maximum in (("steps", 1, 20), ("sampling_steps", 1, 4),
                                       ("rank", 1, 8), ("blocks", 1, 4), ("fixture_size", 32, 64)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
        if self.fixture_size % 32 or type(self.fixture_only) is not bool:
            raise ValueError("Invalid fixture configuration")
        for name, value in asdict(self).items():
            if name in {"steps", "sampling_steps", "rank", "blocks", "fixture_only", "fixture_size"}:
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.learning_rate <= 0.01 or not 0 < self.parameter_bound <= 1:
            raise ValueError("Learning rate or adapter bound exceeds the experiment budget")
        if any(getattr(self, name) > 1 for name in ("quality_floor", "brand_floor", "minimum_click_gain")):
            raise ValueError("Probability thresholds must be in [0,1]")
        if min(self.quality_penalty, self.brand_penalty, self.adapter_l2) <= 0:
            raise ValueError("Separate quality, brand and adapter penalties are required")
        if self.latent_anchor + self.image_anchor <= 0:
            raise ValueError("A matched-parent anchor is required")
        return self


def _seed(value):
    if type(value) is not int or not 0 <= value < 2 ** 32:
        raise ValueError("Seed must be uint32")
    return value


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _cases(train_cases, audit_cases, config, rewards):
    if not all(isinstance(cases, (list, tuple)) and cases for cases in (train_cases, audit_cases)):
        raise ValueError("Nonempty train_cases and audit_cases are required")
    if len(train_cases) + len(audit_cases) > 8:
        raise ValueError("At most eight total cases are allowed")
    partitions = []
    for cases in (train_cases, audit_cases):
        prepared = []
        for case in cases:
            if not isinstance(case, dict):
                raise ValueError("Cases must be dictionaries")
            for name in ("id", "profileId", "prompt"):
                if not isinstance(case.get(name), str) or not case[name].strip() or len(case[name]) > 4096:
                    raise ValueError(f"Case {name} must be nonempty bounded text")
            if case.get("encoder", LEGACY_ENCODER) != rewards.encoder:
                raise ValueError("Alignment case encoder must exactly match the reward encoder")
            upl = conditioning_tensor(case.get("upl"), rewards.encoder).detach()
            entry = {name: case[name] for name in ("id", "profileId", "prompt")}
            entry.update(upl=upl.tolist(), seed=_seed(case.get("seed")))
            if rewards.encoder != LEGACY_ENCODER or "uplBundle" in case:
                entry.update(frozen_upl(case))
            if "prompt_embeds" in case or "prompt_attention_mask" in case:
                if not config.fixture_only:
                    raise ValueError("Supplied embeddings are fixture-only; real prompts are encoded by the pipeline")
                embeds = torch.as_tensor(case.get("prompt_embeds")).detach().clone()
                mask = torch.as_tensor(case.get("prompt_attention_mask")).detach().clone()
                if (embeds.ndim != 3 or embeds.shape[0] != 1 or not 1 <= embeds.shape[1] <= 300
                        or mask.shape != embeds.shape[:2] or not torch.isfinite(embeds).all()
                        or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all()):
                    raise ValueError("Invalid fixture prompt embeddings/mask")
                entry["embeddingHash"] = _json_hash([embeds.tolist(), mask.tolist()])
                entry["embeddings"] = (embeds, mask)
            prepared.append(entry)
        partitions.append(prepared)
    training, audit = partitions
    for field in ("id", "profileId", "seed"):
        if {case[field] for case in training} & {case[field] for case in audit}:
            raise ValueError(f"Train/audit {field} groups must be disjoint")
    identifiers = [case["id"] for case in training + audit]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Case IDs must be unique")
    split = rewards.split_profiles
    if {case["profileId"] for case in training} & set(split["validation"]):
        raise ValueError("Reward validation profiles cannot enter generator training")
    if {case["profileId"] for case in audit} & set(split["train"]):
        raise ValueError("Generator audit profiles cannot be reward-training profiles")
    return training, audit


def _reward_evidence(rewards, config):
    if not isinstance(rewards, RewardBundle):
        raise ValueError("A validated learned RewardBundle is required")
    artifact = export_bundle(rewards)
    metadata = artifact["metadata"]
    for task, loss in (("click", "bce"), ("click", "brier"), ("quality", "mse"), ("brand", "bce")):
        metrics = metadata["metrics"][task]
        if not metrics[loss] < metrics["baseline_" + loss]:
            raise ValueError(f"Reward validation fails {task}.{loss}; training is blocked")
    if not config.fixture_only:
        if metadata["fixtureOnly"] or metadata["humanReviewCount"] < 100:
            raise ValueError("HD adaptation requires observed clicks and human quality/brand evidence: at least 100 operator reviews")
        for part in ("train", "validation"):
            if (metadata["sources"]["click"][part].get("local-ledger", 0) < 1
                    or metadata["sources"]["review"][part].get("operator", 0) < 1):
                raise ValueError("Observed click and operator review evidence is missing")
    for model in (rewards.click_model, rewards.review_model):
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise ValueError("Reward models must be frozen and eval-only")
    return artifact


def _pipeline_contract(pipeline, config):
    import diffusers
    from diffusers import AutoencoderDC, SanaSprintPipeline, SanaTransformer2DModel, SCMScheduler

    if diffusers.__version__ != DIFFUSERS_VERSION:
        raise ValueError(f"Audited sampler requires diffusers=={DIFFUSERS_VERSION}")
    if (type(pipeline) is not SanaSprintPipeline or type(pipeline.scheduler) is not SCMScheduler
            or type(pipeline.transformer) is not SanaTransformer2DModel or type(pipeline.vae) is not AutoencoderDC):
        raise ValueError("Expected official SanaSprint/SCM/SanaTransformer/AutoencoderDC components")
    if pipeline.scheduler.config.prediction_type != "trigflow" or pipeline.scheduler.config.sigma_data <= 0:
        raise ValueError("Expected a valid trigflow SCM scheduler")
    modules = [pipeline.transformer, pipeline.vae]
    if pipeline.text_encoder is not None:
        modules.append(pipeline.text_encoder)
    if any(hasattr(module, "_hf_hook") for component in modules for module in component.modules()):
        raise ValueError("Offload hooks are unsupported; no implicit device fallback")
    if getattr(pipeline.transformer, "peft_config", None):
        raise ValueError("Supply an unadapted parent without existing PEFT adapters")
    device = next(pipeline.transformer.parameters()).device
    if any(parameter.device != device for module in modules for parameter in module.parameters()):
        raise ValueError("All foundation components must remain on the same explicit device")
    if device.type == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise ValueError("MPS CPU fallback must be disabled")
    if not config.fixture_only:
        if device.type not in ("cuda", "mps") or pipeline.text_encoder is None:
            raise ValueError("Real HD alignment requires a loaded GPU pipeline, not a CPU fixture")
        location = str(pipeline.config.get("_name_or_path", ""))
        if Path(location).name != REVISION or Path(location).parent.name != "snapshots":
            raise ValueError("Pipeline must come from the pinned local snapshot loader")
        if pipeline.vae_scale_factor != 32 or pipeline.transformer.config.in_channels != 32:
            raise ValueError("Unexpected native Sprint latent contract")
    if not pipeline.transformer.config.guidance_embeds:
        raise ValueError("Sprint requires guidance-distilled transformer conditioning")
    return modules, device


class BoundedLoRALinear(nn.Module):
    """Frozen linear plus float32 rank update, scaled by 1/rank."""

    def __init__(self, base, rank, generator, bound):
        super().__init__()
        if type(base) is not nn.Linear or rank > min(base.in_features, base.out_features):
            raise ValueError("LoRA target must be an unadapted compatible Linear")
        self.base = base
        self.scale = 1 / rank
        self.enabled = True
        initial = torch.randn(rank, base.in_features, generator=generator, dtype=torch.float32) / math.sqrt(base.in_features)
        self.down = nn.Parameter(initial.clamp(-bound, bound).to(base.weight.device))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        original = self.base(inputs)
        if not self.enabled:
            return original
        update = functional.linear(functional.linear(inputs.float(), self.down), self.up)
        return original + (update * self.scale).to(original.dtype)


@contextmanager
def adapter_scope(transformer, config, seed=2026, state=None):
    """Temporarily install bounded q/v adapters; optionally replay saved tensors."""
    config.validate()
    generator = torch.Generator(device="cpu").manual_seed(_seed(seed))
    targets = []
    for index in range(min(config.blocks, len(transformer.transformer_blocks))):
        attention = transformer.transformer_blocks[index].attn1
        for name in ("to_q", "to_v"):
            original = getattr(attention, name)
            wrapper = BoundedLoRALinear(original, config.rank, generator, config.parameter_bound)
            targets.append((f"transformer_blocks.{index}.attn1.{name}", attention, name, original, wrapper))
    if not targets or sum(wrapper.down.numel() + wrapper.up.numel() for *_, wrapper in targets) > 1000000:
        raise ValueError("Adapter target budget exceeded or no q/v targets found")
    expected = {f"{path}.{name}" for path, *_, wrapper in targets for name in ("down", "up")}
    if state is not None:
        if set(state) != expected:
            raise ValueError("Adapter state keys do not match the bounded target set")
        with torch.no_grad():
            for path, *_, wrapper in targets:
                for name in ("down", "up"):
                    value = state[f"{path}.{name}"]
                    parameter = getattr(wrapper, name)
                    if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                            or not torch.isfinite(value).all() or (value.abs() > config.parameter_bound).any()):
                        raise ValueError("Invalid adapter tensor or bound")
                    parameter.copy_(value)
    base_flags = [(parameter, parameter.requires_grad) for parameter in transformer.parameters()]
    try:
        transformer.requires_grad_(False)
        for _, attention, name, _, wrapper in targets:
            setattr(attention, name, wrapper)
        yield {path: wrapper for path, *_, wrapper in targets}
    finally:
        for _, attention, name, original, _ in targets:
            setattr(attention, name, original)
        for parameter, flag in base_flags:
            parameter.requires_grad_(flag)


def _sample(pipeline, case, config):
    from diffusers import SanaSprintPipeline

    width, height = (config.fixture_size, config.fixture_size) if config.fixture_only else (1920, 1088)
    arguments = {"prompt": case["prompt"], "width": width, "height": height,
                 "num_inference_steps": config.sampling_steps, "intermediate_timesteps": 1.3 if config.sampling_steps == 2 else None,
                 "max_timesteps": 1.5708, "guidance_scale": 4.5, "use_resolution_binning": False,
                 "timesteps": None, "eta": 0.0, "num_images_per_prompt": 1,
                 "complex_human_instruction": None, "clean_caption": False, "max_sequence_length": 300,
                 "output_type": "latent", "return_dict": False,
                 "generator": torch.Generator(device="cpu").manual_seed(case["seed"])}
    if "embeddings" in case:
        arguments.update(prompt=None, prompt_embeds=case["embeddings"][0], prompt_attention_mask=case["embeddings"][1])
    latents = inspect.unwrap(SanaSprintPipeline.__call__)(pipeline, **arguments)[0]
    expected = (1, pipeline.transformer.config.in_channels, height // pipeline.vae_scale_factor, width // pipeline.vae_scale_factor)
    if tuple(latents.shape) != expected or not torch.isfinite(latents).all():
        raise ValueError("SCM produced invalid native latents")
    decoded = pipeline.vae.decode(latents.to(pipeline.vae.dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0]
    if tuple(decoded.shape) != (1, 3, height, width) or not torch.isfinite(decoded).all():
        raise ValueError("Decoder did not produce finite native RGB; resizing is forbidden")
    image = (decoded.float() / 2 + 0.5).clamp(0, 1)
    if not config.fixture_only:
        image = image[:, :, 4:1084, :]
    return latents, image


def _scores(rewards, image, case, config):
    values = rewards.predict(image.to(device="cpu", dtype=torch.float32), torch.tensor([case["upl"]], dtype=torch.float32))
    for name in ("click", "quality", "brand"):
        for field in (name, name + "_std"):
            value = values[field]
            if value.shape != (1,) or not torch.isfinite(value).all() or ((value < 0) | (value > 1)).any():
                raise ValueError("Invalid reward predictions")
    return {name: (values[name] - config.uncertainty_scale * values[name + "_std"]).clamp(0, 1)
            for name in ("click", "quality", "brand")}


def _set_enabled(adapters, enabled):
    for wrapper in adapters.values():
        wrapper.enabled = enabled


def align_pipeline(pipeline, train_cases, audit_cases, rewards, *, output=None, config=None, seed=2026):
    """Return {adapter_state, report}; optionally save a new safetensors/JSON directory.

    Audit runs once after all fixed-budget updates, without gradients or checkpoint
    selection. Rejected candidates remain unpromoted. Exceptions restore modules
    and propagate; OOM never triggers a CPU or reduced-resolution retry.
    """
    config = (config or AlignmentConfig()).validate()
    seed = _seed(seed)
    if torch.is_inference_mode_enabled():
        raise ValueError("Alignment cannot run inside torch.inference_mode")
    if output is not None and Path(output).exists():
        raise ValueError("Output must be a new directory")
    evidence = _reward_evidence(rewards, config)
    training, audit = _cases(train_cases, audit_cases, config, rewards)
    modules, device = _pipeline_contract(pipeline, config)
    parameters = [(parameter, parameter.requires_grad, parameter._version) for module in modules for parameter in module.parameters()]
    modes = [(module, module.training) for component in modules for module in component.modules()]
    started = time.monotonic()
    history, audit_rows = [], []
    try:
        for module in modules:
            module.eval().requires_grad_(False)
        with adapter_scope(pipeline.transformer, config, seed) as adapters:
            trainable = [parameter for wrapper in adapters.values() for parameter in (wrapper.down, wrapper.up)]
            initial = [parameter.detach().cpu().clone() for parameter in trainable]
            optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
            with torch.enable_grad():
                for step in range(config.steps):
                    case = training[step % len(training)]
                    optimizer.zero_grad(set_to_none=True)
                    _set_enabled(adapters, False)
                    with torch.no_grad():
                        parent_latents, parent_image = _sample(pipeline, case, config)
                        parent_latents, parent_image = parent_latents.cpu(), parent_image.cpu()
                    _set_enabled(adapters, True)
                    latents, image = _sample(pipeline, case, config)
                    scores = _scores(rewards, image, case, config)
                    if not all(value.requires_grad for value in scores.values()):
                        raise ValueError("Reward image gradients are detached; refusing an anchor-only update")
                    quality_cost = functional.relu(config.quality_floor - scores["quality"]).mean()
                    brand_cost = functional.relu(config.brand_floor - scores["brand"]).mean()
                    latent_cost = functional.mse_loss(latents.float().cpu(), parent_latents.float())
                    image_cost = functional.mse_loss(image.cpu(), parent_image)
                    regularizer = torch.stack([parameter.square().mean().cpu() for parameter in trainable]).mean()
                    loss = (-scores["click"].mean() + config.quality_penalty * quality_cost
                            + config.brand_penalty * brand_cost + config.latent_anchor * latent_cost
                            + config.image_anchor * image_cost + config.adapter_l2 * regularizer)
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite alignment objective")
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                    if gradient_norm.item() == 0:
                        raise ValueError("No adapter gradient signal")
                    optimizer.step()
                    with torch.no_grad():
                        for parameter in trainable:
                            parameter.clamp_(-config.parameter_bound, config.parameter_bound)
                    history.append({"step": step + 1, "caseId": case["id"], "loss": loss.item(),
                                    "clickLCB": scores["click"].item(), "qualityCost": quality_cost.item(),
                                    "brandCost": brand_cost.item(), "latentAnchor": latent_cost.item(),
                                    "imageAnchor": image_cost.item(), "adapterL2": regularizer.item(),
                                    "gradientNorm": gradient_norm.item()})
                    del loss, scores, latents, image, parent_latents, parent_image
                    del quality_cost, brand_cost, latent_cost, image_cost, regularizer
            with torch.no_grad():
                for case in audit:
                    _set_enabled(adapters, False)
                    _, parent_image = _sample(pipeline, case, config)
                    baseline = _scores(rewards, parent_image, case, config)
                    del parent_image
                    _set_enabled(adapters, True)
                    _, candidate_image = _sample(pipeline, case, config)
                    candidate = _scores(rewards, candidate_image, case, config)
                    audit_rows.append({"caseId": case["id"], "parent": {name: value.item() for name, value in baseline.items()},
                                       "candidate": {name: value.item() for name, value in candidate.items()}})
                    del candidate_image
            adapter_state = {f"{path}.{name}": getattr(wrapper, name).detach().float().cpu().contiguous().clone()
                             for path, wrapper in adapters.items() for name in ("down", "up")}
            delta = math.sqrt(sum((parameter.detach().cpu() - before).square().sum().item()
                                  for parameter, before in zip(trainable, initial)))
            base_unchanged = all(parameter._version == version for parameter, _, version in parameters)
            if not base_unchanged:
                raise RuntimeError("Foundation weights changed; experiment is invalid")
    finally:
        for parameter, flag, _ in parameters:
            parameter.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode
    click_gain = sum(row["candidate"]["click"] - row["parent"]["click"] for row in audit_rows) / len(audit_rows)
    gates = {"rewardValidation": True, "baseUnchanged": base_unchanged, "adapterChanged": delta > 0,
             "predictedClickGain": click_gain > 0 and click_gain >= config.minimum_click_gain}
    for name in ("quality", "brand"):
        gates[name + "Floor"] = all(row["candidate"][name] >= getattr(config, name + "_floor") for row in audit_rows)
        gates[name + "NonRegression"] = all(row["candidate"][name] >= row["parent"][name] - 1e-6 for row in audit_rows)
    passed = all(gates.values())
    cases_metadata = {name: [{key: value for key, value in case.items() if key != "embeddings"} for case in cases]
                      for name, cases in (("train", training), ("audit", audit))}
    report = {"format": FORMAT, "adapterFormat": ADAPTER_FORMAT, "createdAt": datetime.now(timezone.utc).isoformat(),
              "method": "scm-direct-reward-backprop-lora", "likelihoodRatioRL": False, "fixtureOnly": config.fixture_only,
              "parent": {"repository": REPOSITORY, "revision": REVISION} if not config.fixture_only else {"fixture": "random-offline-sana"},
              "diffusersVersion": DIFFUSERS_VERSION, "scheduler": dict(pipeline.scheduler.config),
              "decoderTiling": {"enabled": pipeline.vae.use_tiling,
                                "tileSize": [pipeline.vae.tile_sample_min_width, pipeline.vae.tile_sample_min_height],
                                "stride": [pipeline.vae.tile_sample_stride_width, pipeline.vae.tile_sample_stride_height]},
              "config": asdict(config), "seed": seed, "device": str(device), "noiseGeneratorDevice": "cpu",
              "nativeSize": [config.fixture_size] * 2 if config.fixture_only else [1920, 1088],
              "outputSize": [config.fixture_size] * 2 if config.fixture_only else [1920, 1080],
              "resolutionBinning": False, "guidanceScale": 4.5, "maxTimesteps": 1.5708,
              "rewardId": rewards.id, "rewardEvidence": evidence["metadata"], "cases": cases_metadata,
              "rewardEncoder": rewards.encoder, "generatorConditioning": "caller-supplied-prompt-not-native-UPL-adapter",
              "casesSha256": _json_hash(cases_metadata), "targets": list(adapters), "adapterScale": 1 / config.rank,
              "adapterDeltaL2": delta, "history": history, "audit": audit_rows, "gates": gates,
              "predictedClickLCBGain": click_gain, "status": "surrogate-gates-passed" if passed else "rejected",
              "eligibleForManualReview": passed and not config.fixture_only, "autoPromoted": False,
              "deploymentEligible": False,
              "deploymentGates": {"hdRewardValidation": False, "humanFullResolutionReview": False, "exactBrandReview": False},
              "rewardResolutionCoverage": "unverified",
              "observedCtrLift": None, "conversionLift": None, "hdQualityEstablished": False,
              "elapsedSeconds": time.monotonic() - started,
              "limitations": ["LCB is mean minus ensemble disagreement, not a calibrated confidence bound.",
                              "Pooled reward predictions do not certify high-resolution visual or exact brand quality.",
                              "Reward metadata does not establish HD source coverage; low-resolution fit cannot clear HD deployment gates.",
                              "Human full-resolution and exact brand review remain mandatory before deployment.",
                              "Matched-parent anchoring is not KL; no likelihood ratios or PPO are used.",
                              "A single post-training audit gates this candidate, never selects checkpoints.",
                              "Synthetic fixtures and predicted click gain do not establish real CTR or conversion lift."]}
    if output is not None:
        from safetensors.torch import save_file

        directory = Path(output)
        directory.mkdir(parents=True, exist_ok=False)
        adapter_path = directory / "adapter.safetensors"
        save_file(adapter_state, str(adapter_path), metadata={"format": ADAPTER_FORMAT, "rewardId": rewards.id})
        report["adapterFile"] = {"file": adapter_path.name, "sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest()}
        with (directory / "alignment.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, allow_nan=False, indent=2)
    return {"adapter_state": adapter_state, "report": report}
