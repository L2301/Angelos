"""Small CPU pooled-image reward predictors, not an RL value function.

Records are independent of ledger/export schemas: image [3,H,W], upl [D],
nonempty profileId/creativeId, and source. Clicks use label in {0,1} and source
local-ledger|synthetic; reviews use quality/brand in [0,1] and operator|synthetic.
Images are finite in [0,1]; UPL follows the explicit encoder contract.
Legacy UPLs remain [0,1]^4; embedding UPLs are signed L2-normalized vectors.
Synthetic evidence requires explicit opt-in. Mixed record resolutions are valid.

One seeded, label-blind partition of the union of profile IDs serves both tasks.
Insufficient support in that partition fails rather than searching labels for a
favorable split. The defaults support demos, not scientific confidence. Collect
100--1000 actual human reviews as a target, not a claim about available evidence.
DDPO must use new synthetic UPLs or training profiles, never held-out profiles.

predict accepts nonempty CPU batches and returns float32 [N] means in
[0,1] and population bootstrap-head standard deviations in [0,0.5]. These are
ensemble disagreement, not calibrated uncertainty or confidence intervals.
Reward weights stay frozen, while image and UPL gradients are preserved through
adaptive pooling and scoring. Callers may use torch.no_grad() for inference;
detached DDPO inputs need no gradients. Training evidence is always detached.
export_bundle(bundle, path=None) returns JSON-compatible data and optionally
creates a new JSON file; load_bundle accepts that dict or a JSON file path.
"""

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from .diffusion import LEGACY_ENCODER, conditioning_encoder, conditioning_tensor
from .feedback import frozen_upl


FORMAT = "angelos.rewards.v1"
ARCHITECTURE = {"features": "centered-pool2x2-upl-outer-v1", "featureCount": 64,
                "minImageDimension": 16, "maxImageDimension": 4096,
                "heads": 3, "weightBound": 8, "device": "cpu", "dtype": "float32"}
LIMITATIONS = [
    "Demo holdout metrics are not clinical confidence or evidence of CTR uplift.",
    "Bootstrap disagreement is not calibrated uncertainty.",
    "Quality/brand critic predicts review scores, not RL returns or state values.",
    "Parent RL must use new synthetic UPLs or training-only profiles.",
    "Profile grouping does not guarantee creative-disjoint generalization.",
    "2x2 pooled features discard fine detail; no high-resolution brand fidelity claim.",
    "32px synthetic smoke tests are not a genuine 1080p generation quality milestone.",
]


def reward_architecture(encoder=None):
    encoder = conditioning_encoder(encoder)
    architecture = copy.deepcopy(ARCHITECTURE)
    if encoder != LEGACY_ENCODER:
        architecture.update(features="pool2x2-signed-upl-outer-v2", featureCount=12 + 13 * encoder["dimension"], encoder=encoder)
    return architecture


def _integer(value, name, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _tensor(value, shape, name, lower=0, upper=1, *, preserve_grad=False):
    try:
        tensor = torch.as_tensor(value, device="cpu") if not isinstance(value, torch.Tensor) else value
        if tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError(f"{name} must be a dense CPU tensor")
        if tensor.dtype == torch.bool or tensor.is_complex() or tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"{name} must have numeric shape {shape}")
        if not torch.isfinite(tensor).all() or ((tensor < lower) | (tensor > upper)).any():
            raise ValueError(f"{name} must be finite in [{lower}, {upper}]")
        if not preserve_grad:
            tensor = tensor.detach()
        return tensor.to(dtype=torch.float32).clone()
    except (TypeError, RuntimeError, OverflowError) as error:
        raise ValueError(f"Invalid {name}") from error


def _records(records, task, allow_synthetic, encoder):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError(f"{task} records must be a nonempty list or tuple")
    allowed = {"synthetic", "local-ledger" if task == "click" else "operator"}
    prepared = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each record must be a dict")
        for field in ("profileId", "creativeId"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"{field} must be a nonempty string")
        source = record.get("source")
        if not isinstance(source, str) or source not in allowed:
            raise ValueError(f"Invalid {task} source")
        if source == "synthetic" and not allow_synthetic:
            raise ValueError("Synthetic evidence requires allow_synthetic=True opt-in")
        fields = ("label",) if task == "click" else ("quality", "brand")
        targets = [_tensor(record.get(field), (), field).item() for field in fields]
        if task == "click" and record.get("label") not in (0, 1):
            raise ValueError("Click label must be 0 or 1")
        image = _image(record.get("image"))
        if record.get("encoder", LEGACY_ENCODER) != encoder:
            raise ValueError("Reward record encoder must exactly match the reward contract")
        if "uplBundle" in record or (encoder != LEGACY_ENCODER and source != "synthetic"):
            frozen_upl(record)
        upl = conditioning_tensor(record.get("upl"), encoder).detach().clone()
        prepared.append({
            "features": _features(image.unsqueeze(0), upl.unsqueeze(0), encoder)[0],
            "profileId": record["profileId"], "creativeId": record["creativeId"],
            "source": source, "targets": targets,
        })
    return prepared


def _image(value, batched=False, *, preserve_grad=False):
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device="cpu")
        expected_rank = 4 if batched else 3
        if tensor.ndim != expected_rank or tensor.shape[-3] != 3:
            raise ValueError("Expected images [N,3,H,W] or record image [3,H,W]")
        if any(not 16 <= dimension <= 4096 for dimension in tensor.shape[-2:]):
            raise ValueError("Image height and width must each be in [16,4096]")
        if batched and tensor.shape[0] < 1:
            raise ValueError("Prediction needs at least one image")
        return _tensor(tensor, tensor.shape, "images" if batched else "image", preserve_grad=preserve_grad)
    except (TypeError, RuntimeError, OverflowError) as error:
        raise ValueError("Invalid image tensor") from error


def _features(images, upl, encoder=None):
    pooled = functional.adaptive_avg_pool2d(images, (2, 2)).flatten(1).mul(2).sub(1)
    centered = upl.mul(2).sub(1) if conditioning_encoder(encoder) == LEGACY_ENCODER else upl
    interactions = (pooled.unsqueeze(2) * centered.unsqueeze(1)).flatten(1)
    return torch.cat((pooled, centered, interactions), dim=1)


class _Ensemble(nn.Module):
    def __init__(self, outputs, feature_count=64):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, feature_count, outputs, device="cpu", dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(3, outputs, device="cpu", dtype=torch.float32))

    def forward(self, features):
        return torch.einsum("nf,hfo->hno", features, self.weight) + self.bias[:, None, :]


def _arrays(records):
    return (torch.stack([record["features"] for record in records]),
            torch.tensor([record["targets"] for record in records], dtype=torch.float32, device="cpu"))


def _train(records, task, steps, generator):
    features, targets = _arrays(records)
    model = _Ensemble(targets.shape[1], features.shape[1])
    profiles = sorted({record["profileId"] for record in records})
    profile_indices = {profile: index for index, profile in enumerate(profiles)}
    row_groups = torch.tensor([profile_indices[record["profileId"]] for record in records], device="cpu")
    draws = torch.randint(len(profiles), (3, len(profiles)), generator=generator, device="cpu")
    multiplicities = torch.stack([torch.bincount(draw, minlength=len(profiles)) for draw in draws])
    weights = multiplicities[:, row_groups].to(torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)
    with torch.no_grad():
        model.weight.copy_(torch.randn(model.weight.shape, generator=generator, device="cpu") * 0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.04)
    expanded = targets.unsqueeze(0).expand(3, -1, -1)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        if task == "click":
            losses = functional.binary_cross_entropy_with_logits(logits, expanded, reduction="none").squeeze(-1)
        else:
            losses = (logits[:, :, 0].sigmoid() - expanded[:, :, 0]).square()
            losses = losses + functional.binary_cross_entropy_with_logits(
                logits[:, :, 1], expanded[:, :, 1], reduction="none")
        loss = (losses * weights).sum(dim=1).mean() + 0.002 * model.weight.square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.clamp_(-8, 8)
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    return model.eval()


def _bce(predictions, targets):
    return functional.binary_cross_entropy(predictions.clamp(1e-6, 1 - 1e-6), targets).item()


@torch.no_grad()
def _metrics(click_model, review_model, partitions):
    click_features, click_targets = _arrays(partitions["click"]["validation"])
    review_features, review_targets = _arrays(partitions["review"]["validation"])
    click = click_model(click_features).sigmoid().mean(0)[:, 0]
    reviews = review_model(review_features).sigmoid().mean(0)
    click_targets = click_targets[:, 0]
    click_mean = _arrays(partitions["click"]["train"])[1].mean(0)[0]
    review_mean = _arrays(partitions["review"]["train"])[1].mean(0)
    click_baseline = torch.full_like(click, click_mean.item())
    brand_baseline = torch.full_like(reviews[:, 1], review_mean[1].item())
    return {
        "click": {"bce": _bce(click, click_targets), "brier": (click - click_targets).square().mean().item(),
                  "baseline_bce": _bce(click_baseline, click_targets),
                  "baseline_brier": (click_baseline - click_targets).square().mean().item(),
                  "baseline_probability": click_mean.item()},
        "quality": {"mse": (reviews[:, 0] - review_targets[:, 0]).square().mean().item(),
                    "baseline_mse": (review_mean[0] - review_targets[:, 0]).square().mean().item(),
                    "baseline_mean": review_mean[0].item()},
        "brand": {"bce": _bce(reviews[:, 1], review_targets[:, 1]),
                  "baseline_bce": _bce(brand_baseline, review_targets[:, 1]),
                  "baseline_probability": review_mean[1].item()},
    }


def _canonical(payload):
    try:
        return json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Bundle must contain only finite JSON data") from error


def _identity(payload):
    return "reward-" + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


class RewardBundle:
    """Frozen separate click predictor and two-output human quality/brand critic.

    id binds architecture, float32 weights and metadata. split_profiles contains
    sorted train/validation profile ID lists across BOTH evidence collections.
    Metadata access returns copies; model parameters are frozen and eval-only at
    construction. Public model mutation invalidates export's content identity.
    """

    def __init__(self, click_model, review_model, metadata, encoder=None):
        self._encoder = conditioning_encoder(encoder)
        feature_count = reward_architecture(self._encoder)["featureCount"]
        for model, outputs in ((click_model, 1), (review_model, 2)):
            if model.weight.shape != (3, feature_count, outputs) or model.bias.shape != (3, outputs):
                raise ValueError("Reward weight shapes must match the full encoder contract")
        self.click_model = click_model.eval().requires_grad_(False)
        self.review_model = review_model.eval().requires_grad_(False)
        self._metadata = copy.deepcopy(metadata)
        self.id = _identity(self._payload())

    @property
    def metadata(self):
        return copy.deepcopy(self._metadata)

    @property
    def encoder(self):
        return copy.deepcopy(self._encoder)

    @property
    def split_profiles(self):
        return copy.deepcopy(self._metadata["splitProfiles"])

    def predict(self, images, upl):
        """Score [N,3,H,W], 16 <= H,W <= 4096; return six CPU [N] tensors.

        Image and UPL gradients follow the caller's grad mode; weights remain
        frozen. Wrap in torch.no_grad() when an inference-only result is needed.
        """
        images = _image(images, batched=True, preserve_grad=True)
        count = images.shape[0]
        if isinstance(upl, torch.Tensor) and upl.device.type != "cpu":
            raise ValueError("Reward UPL inputs must be on CPU")
        upl = conditioning_tensor(upl, self.encoder, batched=True)
        if len(upl) != count:
            raise ValueError("Reward image and UPL batch counts must match")
        features = _features(images, upl, self.encoder)
        predictions = torch.cat((self.click_model(features).sigmoid(),
                                 self.review_model(features).sigmoid()), dim=2)
        if not torch.isfinite(predictions).all():
            raise ValueError("Non-finite reward prediction")
        means = predictions.mean(0)
        deviations = predictions.std(0, unbiased=False).clamp(0, 0.5)
        result = {}
        for index, name in enumerate(("click", "quality", "brand")):
            result[name] = means[:, index].contiguous()
            result[name + "_std"] = deviations[:, index].contiguous()
        return result

    def _payload(self):
        weights = {}
        for name, model in (("click", self.click_model), ("review", self.review_model)):
            weights[name] = {key: _tensor(value, value.shape, key, -8, 8).tolist()
                             for key, value in model.state_dict().items()}
        return {"format": FORMAT if self.encoder == LEGACY_ENCODER else "angelos.rewards.v2", "architecture": reward_architecture(self.encoder),
                "metadata": self.metadata, "weights": weights}


def fit_reward_bundle(click_records, review_records, *, seed=2026, steps=200,
                      allow_synthetic=False, validation_fraction=0.25,
                      min_train_records=8, min_validation_records=4,
                      min_train_profiles=2, min_validation_profiles=2, encoder=None):
    """Fit three profile-bootstrap heads per task using training evidence only.

    Evidence minima apply independently to clicks and reviews in each partition.
    Profile minima cannot be reduced below two. Both click partitions need both
    classes; training quality and brand each need score variation. Validation is
    a single label-blind split, not used for fitting, tuning or model selection.
    """
    encoder = conditioning_encoder(encoder)
    _integer(seed, "seed", 0, 2 ** 32 - 1)
    _integer(steps, "steps")
    if type(allow_synthetic) is not bool:
        raise ValueError("allow_synthetic must be bool")
    if type(validation_fraction) not in (float, int) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be finite and in (0,1)")
    evidence = {"min_train_records": min_train_records, "min_validation_records": min_validation_records,
                "min_train_profiles": min_train_profiles, "min_validation_profiles": min_validation_profiles}
    for name, value in evidence.items():
        _integer(value, name, 2)
    records = {"click": _records(click_records, "click", allow_synthetic, encoder),
               "review": _records(review_records, "review", allow_synthetic, encoder)}
    profiles = sorted({record["profileId"] for collection in records.values() for record in collection})
    validation_count = max(min_validation_profiles, math.ceil(len(profiles) * validation_fraction))
    if len(profiles) - validation_count < min_train_profiles:
        raise ValueError("Insufficient profile groups for disjoint train/validation")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(profiles), generator=generator, device="cpu").tolist()
    validation_profiles = {profiles[index] for index in order[:validation_count]}
    split = {"train": sorted(set(profiles) - validation_profiles), "validation": sorted(validation_profiles)}
    partitions, counts, sources = {}, {}, {}
    for task, collection in records.items():
        partitions[task], counts[task], sources[task] = {}, {}, {}
        for part in ("train", "validation"):
            selected = [record for record in collection if (record["profileId"] in validation_profiles) == (part == "validation")]
            group_count = len({record["profileId"] for record in selected})
            if len(selected) < evidence[f"min_{part}_records"] or group_count < evidence[f"min_{part}_profiles"]:
                raise ValueError(f"Insufficient {task} {part} records/profile groups in label-blind split")
            targets = torch.tensor([record["targets"] for record in selected], device="cpu")
            if task == "click" and targets.unique().numel() != 2:
                raise ValueError(f"Click {part} requires both classes (class variation)")
            if task == "review" and part == "train" and any(targets[:, index].unique().numel() < 2 for index in (0, 1)):
                raise ValueError("Review training requires quality and brand score variation")
            partitions[task][part] = selected
            counts[task][part] = {"records": len(selected), "profiles": group_count}
            if task == "click":
                counts[task][part].update({"positive": int(targets.sum().item()),
                                          "negative": len(selected) - int(targets.sum().item())})
            sources[task][part] = {source: sum(record["source"] == source for record in selected)
                                   for source in sorted({record["source"] for record in selected})}
    click_model = _train(partitions["click"]["train"], "click", steps, generator)
    review_model = _train(partitions["review"]["train"], "review", steps, generator)
    metadata = {
        "seed": seed, "steps": steps, "validationFraction": validation_fraction,
        "fixtureOnly": any(record["source"] == "synthetic" for collection in records.values() for record in collection),
        "splitProfiles": split, "counts": counts, "sources": sources,
        "metrics": _metrics(click_model, review_model, partitions), "evidence": evidence,
        "humanReviewCount": sum(record["source"] == "operator" for record in records["review"]),
        "humanReviewTarget": [100, 1000], "limitations": list(LIMITATIONS),
        "bootstrapUnit": "profile", "criticType": "human-quality-brand-not-rl-value",
    }
    if encoder != LEGACY_ENCODER:
        metadata["conditioning"] = "semantic-vector-only; frozen recentIntent retained as provenance, not adapter features"
        metadata["uplProvenance"] = [
            {"profileId": record["profileId"], "creativeId": record["creativeId"],
             "encoder": copy.deepcopy(encoder), "upl": record["upl"].detach().cpu().tolist() if isinstance(record["upl"], torch.Tensor) else copy.deepcopy(record["upl"]),
             **({"uplBundle": copy.deepcopy(record["uplBundle"])} if "uplBundle" in record else {"syntheticVectorFixture": True})}
            for record in list(click_records) + list(review_records)]
    return RewardBundle(click_model, review_model, metadata, encoder)


def _validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Invalid reward metadata")
    try:
        _integer(metadata["seed"], "seed", 0, 2 ** 32 - 1)
        _integer(metadata["steps"], "steps")
        fraction = metadata["validationFraction"]
        if type(fraction) not in (float, int) or not 0 < fraction < 1:
            raise ValueError("Invalid validation fraction")
        if metadata["criticType"] != "human-quality-brand-not-rl-value" or metadata["bootstrapUnit"] != "profile":
            raise ValueError("Invalid reward semantics")
        if metadata["humanReviewTarget"] != [100, 1000] or metadata["limitations"] != LIMITATIONS:
            raise ValueError("Missing evidence limitations")
        split = metadata["splitProfiles"]
        if not isinstance(split, dict) or set(split) != {"train", "validation"}:
            raise ValueError("Invalid split profiles")
        for part in ("train", "validation"):
            group_ids = split[part]
            if not isinstance(group_ids, list) or not group_ids or any(not isinstance(profile, str) or not profile.strip() for profile in group_ids):
                raise ValueError("Invalid split profile IDs")
            if group_ids != sorted(set(group_ids)):
                raise ValueError("Split profile IDs must be sorted and unique")
        if set(split["train"]) & set(split["validation"]):
            raise ValueError("Profile leakage across split")
        synthetic_count, human_count = 0, 0
        for task in ("click", "review"):
            allowed = {"synthetic", "local-ledger" if task == "click" else "operator"}
            for part in ("train", "validation"):
                counts = metadata["counts"][task][part]
                for field in ("records", "profiles"):
                    minimum = _integer(metadata["evidence"][f"min_{part}_{field}"], "evidence minimum", 2)
                    _integer(counts[field], field, minimum)
                if not counts["profiles"] <= min(counts["records"], len(split[part])):
                    raise ValueError("Invalid evidence counts")
                source_counts = metadata["sources"][task][part]
                if not isinstance(source_counts, dict) or not set(source_counts) <= allowed:
                    raise ValueError("Invalid source labels")
                for count in source_counts.values():
                    _integer(count, "source count")
                if sum(source_counts.values()) != counts["records"]:
                    raise ValueError("Source counts disagree with record counts")
                synthetic_count += source_counts.get("synthetic", 0)
                human_count += source_counts.get("operator", 0)
                if task == "click":
                    for field in ("positive", "negative"):
                        _integer(counts[field], field)
                    if counts["positive"] + counts["negative"] != counts["records"]:
                        raise ValueError("Invalid click class counts")
        if type(metadata["fixtureOnly"]) is not bool or metadata["fixtureOnly"] != (synthetic_count > 0):
            raise ValueError("Invalid fixtureOnly provenance")
        if type(metadata["humanReviewCount"]) is not int or metadata["humanReviewCount"] != human_count:
            raise ValueError("Invalid human review count")
        metric_fields = {"click": {"bce", "brier", "baseline_bce", "baseline_brier", "baseline_probability"},
                         "quality": {"mse", "baseline_mse", "baseline_mean"},
                         "brand": {"bce", "baseline_bce", "baseline_probability"}}
        if set(metadata["metrics"]) != set(metric_fields):
            raise ValueError("Invalid metrics")
        for task, fields in metric_fields.items():
            values = metadata["metrics"][task]
            if set(values) != fields:
                raise ValueError("Invalid metric fields")
            for name, value in values.items():
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or ("bce" not in name and value > 1):
                    raise ValueError("Invalid finite bounded metric")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed reward metadata") from error


def export_bundle(bundle, path=None):
    """Return a finite JSON artifact; optional path is created without overwrite."""
    if not isinstance(bundle, RewardBundle):
        raise ValueError("Expected a RewardBundle")
    payload = bundle._payload()
    _validate_metadata(payload["metadata"])
    _validate_upl_metadata(payload["metadata"], bundle.encoder)
    if _identity(payload) != bundle.id:
        raise ValueError("Bundle changed after its content ID was assigned")
    artifact = {"id": bundle.id, **payload}
    if path is not None:
        with Path(path).open("x", encoding="utf-8") as handle:
            handle.write(_canonical(artifact) + "\n")
    return artifact


def _validate_upl_metadata(metadata, encoder):
    if encoder == LEGACY_ENCODER:
        return
    records = metadata.get("uplProvenance")
    if not isinstance(records, list) or not records:
        raise ValueError("UPLv2 reward checkpoints require frozen input provenance")
    for record in records:
        if not isinstance(record, dict) or record.get("encoder") != encoder:
            raise ValueError("Reward provenance encoder mismatch")
        conditioning_tensor(record.get("upl"), encoder)
        if "uplBundle" in record:
            frozen_upl(record)
        elif record.get("syntheticVectorFixture") is not True or not metadata["fixtureOnly"]:
            raise ValueError("Only explicit synthetic vector fixtures may omit frozen input bundles")


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def load_bundle(artifact_or_path):
    """Load only versioned JSON lists of bounded weights; never pickle or code."""
    if isinstance(artifact_or_path, dict):
        artifact = copy.deepcopy(artifact_or_path)
    else:
        try:
            artifact = json.loads(Path(artifact_or_path).read_text(encoding="utf-8"),
                                  object_pairs_hook=_unique_json_object)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("Expected a reward JSON artifact or path") from error
    if not isinstance(artifact, dict) or set(artifact) != {"id", "format", "architecture", "metadata", "weights"}:
        raise ValueError("Invalid reward artifact fields")
    architecture = artifact["architecture"]
    if not isinstance(architecture, dict):
        raise ValueError("Invalid reward architecture")
    encoder = conditioning_encoder(architecture.get("encoder"))
    expected_format = FORMAT if encoder == LEGACY_ENCODER else "angelos.rewards.v2"
    if artifact["format"] != expected_format or architecture != reward_architecture(encoder):
        raise ValueError("Unsupported reward artifact architecture")
    payload = {key: value for key, value in artifact.items() if key != "id"}
    if artifact["id"] != _identity(payload):
        raise ValueError("Reward artifact content ID mismatch")
    _validate_metadata(artifact["metadata"])
    _validate_upl_metadata(artifact["metadata"], encoder)
    weights = artifact["weights"]
    if not isinstance(weights, dict) or set(weights) != {"click", "review"}:
        raise ValueError("Invalid reward model weights")
    models = []
    for name, outputs in (("click", 1), ("review", 2)):
        state = weights[name]
        if not isinstance(state, dict) or set(state) != {"weight", "bias"}:
            raise ValueError("Invalid reward parameter names")
        feature_count = architecture["featureCount"]
        model = _Ensemble(outputs, feature_count)
        model.load_state_dict({"weight": _tensor(state["weight"], (3, feature_count, outputs), "weight", -8, 8),
                               "bias": _tensor(state["bias"], (3, outputs), "bias", -8, 8)})
        models.append(model)
    bundle = RewardBundle(*models, artifact["metadata"], encoder)
    if bundle.id != artifact["id"]:
        raise ValueError("Weights must round-trip exactly as float32 JSON values")
    return bundle
