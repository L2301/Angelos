"""Train the native-32 diffusion student on an explicit graph-complete dataset.

No provider calls, downloads, or procedural substitution occur in this trainer.
The procedural fixture path is opt-in and recorded as such in every artifact.
Provenance checks establish consistency with the pinned registry and graph, not
independent attestation that a provider produced the supplied pixels.
"""

import argparse
import copy
import hashlib
import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from . import __version__
from .data import profile_splits
from .diffusion import LEGACY_ENCODER, DiffusionStudent, conditioning_encoder, conditioning_tensor, composite_rgb, cosine_schedule, sample_ddim, validate_seed
from .evaluate import BRAND_TOLERANCE, byte_rgba, float_rgba
from .graph import validate_graph
from .feedback import frozen_upl
from .models import model_metadata
from .providers import node_seed
from .teacher import TensorTeacher, apply_brand, brand_masks, brand_probe_coordinates, pixel_coordinates
from .train import write_artifacts


def verify_canonical_graph(graph, compiler_hash, canonical_path):
    validate_graph(graph)
    if canonical_path is None:
        raise ValueError("A compiler --graph-canonical file is required")
    if not isinstance(compiler_hash, str) or len(compiler_hash) != 64 or any(character not in "0123456789abcdefABCDEF" for character in compiler_hash):
        raise ValueError("--graph-hash must be a 64-character hexadecimal compiler hash")
    canonical = Path(canonical_path).read_text(encoding="utf-8").strip()
    normalized = copy.deepcopy(graph)
    normalized["nodes"] = sorted(normalized["nodes"], key=lambda node: node["id"])
    if json.loads(canonical) != normalized:
        raise ValueError("Compiler canonical graph does not match the training graph")
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if computed != compiler_hash.lower():
        raise ValueError("Compiler graph hash does not match canonical file")
    conditioning_encoder(graph.get("encoder"))
    return computed


def resolve_device(device):
    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("Device must be cpu, mps, or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable; refusing CPU fallback")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Requested MPS is unavailable; refusing CPU fallback")
    return torch.device(device)


def validate_model_provenance(manifest, graph, allow_fixture):
    nodes = {node["id"]: node for node in graph["nodes"] if node["op"] in ("generate_text", "generate_image", "generate_video")}
    expected_ids = {node["params"]["model"] for node in nodes.values()}
    source = manifest["source"]
    injected = source == "injected-test-pipeline"
    procedural = "procedural" in source.lower() or "fixture" in source.lower() or source == "deterministic-raster"
    if (procedural or injected) and not allow_fixture:
        raise ValueError("Fixture datasets require explicit --procedural-fixture opt-in")
    if not isinstance(manifest["models"], list):
        raise ValueError("Dataset models must be a list with unique pinned model IDs")
    models = {}
    for metadata in manifest["models"]:
        if not isinstance(metadata, dict) or not isinstance(metadata.get("id"), str):
            raise ValueError("Every dataset model requires an ID and pinned metadata")
        identifier = metadata["id"]
        if identifier in models:
            raise ValueError("Duplicate dataset model ID")
        pinned = model_metadata(identifier)
        for field in ("repo", "revision", "pipeline", "kind", "license"):
            if metadata.get(field) != pinned[field]:
                raise ValueError(f"Dataset model {identifier}.{field} does not match the pinned registry")
        if "safetensors" in metadata and metadata["safetensors"] != pinned["safetensors"]:
            raise ValueError("Dataset model safetensors metadata contradicts the pinned registry")
        backend = metadata.get("backend")
        expected_backend = "transformers" if pinned["kind"] == "text" else "diffusers"
        if backend not in (expected_backend, "injected-test-pipeline"):
            raise ValueError("Dataset model backend must match its model kind or identify injected-test-pipeline")
        if backend == "injected-test-pipeline" and not (injected and allow_fixture):
            raise ValueError("Injected model backends require an injected-test-pipeline source and explicit fixture opt-in")
        models[identifier] = metadata
    if set(models) != expected_ids:
        raise ValueError("Dataset model IDs must exactly match the generative models used by the graph")
    if injected:
        if manifest["synthetic"] is not True or not any(metadata["backend"] == "injected-test-pipeline" for metadata in models.values()):
            raise ValueError("Injected fixture provenance requires synthetic=true and an injected model backend")
    elif nodes:
        expected_source = "transformers-diffusers-local-open-weights" if any(node["op"] == "generate_text" for node in nodes.values()) else "diffusers-local-open-weights"
        if source != expected_source:
            raise ValueError(f"Generated model provenance requires a {expected_source} source")
    elif not procedural:
        raise ValueError("Graphs without generative models require explicit procedural fixture provenance")
    return nodes, models, procedural or injected


def validate_sample_generations(sample, nodes, models, teacher_seed, graph=None):
    has_text = any(node["op"] == "generate_text" for node in nodes.values())
    if "generations" not in sample:
        if has_text:
            raise ValueError("Text generation requires complete sample provenance")
        return
    generations = sample["generations"]
    if not isinstance(generations, list):
        raise ValueError("Sample generations must be a list")
    observed = set()
    for generation in generations:
        if not isinstance(generation, dict) or not isinstance(generation.get("node"), str):
            raise ValueError("Every sample generation must identify its graph node")
        identifier = generation["node"]
        if identifier not in nodes or identifier in observed:
            raise ValueError("Sample generation references an unknown or duplicate graph node")
        observed.add(identifier)
        node = nodes[identifier]
        metadata = models[node["params"]["model"]]
        if has_text and (not {"node", "seed", "prompt", "model", "params"}.issubset(generation) or generation["model"] != metadata):
            raise ValueError("Text-conditioned graphs require complete generation bindings")
        if node["op"] == "generate_text":
            from .text_contract import MAX_TEXT_UNITS, bounded_text, validate_text_result

            required = {"node", "seed", "prompt", "model", "params", "text", "usage", "finish_reason"}
            if set(generation) != required or generation["model"] != metadata:
                raise ValueError("Text generation requires complete matching provenance")
            bounded_text(generation["prompt"], MAX_TEXT_UNITS, "Text generation prompt")
            validate_text_result({name: generation[name] for name in ("text", "usage", "finish_reason")}, node["params"]["max_tokens"])
        if "model" in generation:
            supplied = generation["model"]
            if not isinstance(supplied, dict) or supplied.get("id") != metadata["id"]:
                raise ValueError("Sample generation model does not match its graph node")
            if any(key not in metadata or value != metadata[key] for key, value in supplied.items()):
                raise ValueError("Sample generation model metadata contradicts the dataset manifest")
        if "params" in generation and generation["params"] != node["params"]:
            raise ValueError("Sample generation parameters contradict the graph")
        if "seed" in generation and (type(generation["seed"]) is not int or generation["seed"] != node_seed(teacher_seed, identifier)):
            raise ValueError("Sample generation seed contradicts the teacher seed and graph node")
    if observed != set(nodes):
        raise ValueError("Sample generation records must cover all generative graph nodes")
    by_node = {generation["node"]: generation for generation in generations}
    graph_nodes = {node["id"]: node for node in graph["nodes"]} if graph is not None else {}
    for identifier, node in nodes.items():
        source = node["inputs"]["prompt"]
        if source in nodes and nodes[source]["op"] == "generate_text":
            if by_node[identifier].get("prompt") != by_node[source]["text"]:
                raise ValueError("Generated text does not match the downstream prompt binding")
        elif has_text and source in graph_nodes and graph_nodes[source]["op"] == "prompt":
            prompt_node = graph_nodes[source]
            condition_node = graph_nodes[prompt_node["inputs"]["condition"]]
            condition = sample["upl"][int(condition_node["params"]["index"])] if condition_node["op"] == "feature" else condition_node["params"]["value"]
            parameters = prompt_node["params"]
            expected = parameters["low"] if condition < parameters["threshold"] else parameters["high"]
            expected = expected.replace("{condition}", f"{condition:.6f}")
            if by_node[identifier]["prompt"] != expected:
                raise ValueError("Text-conditioned generation prompt contradicts the exposure-time profile")


def load_dataset(path, graph, graph_hash, allow_procedural=False):
    validate_graph(graph)
    encoder = conditioning_encoder(graph.get("encoder"))
    path = Path(path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"graphHash", "encoder", "graph", "models", "samples", "source", "synthetic"}
    if not required.issubset(manifest):
        raise ValueError(f"Dataset manifest requires {sorted(required)}")
    normalized = lambda value: {**value, "nodes": sorted(value["nodes"], key=lambda node: node["id"])}
    if manifest["graphHash"] != graph_hash or manifest["encoder"] != graph["encoder"] or normalized(manifest["graph"]) != normalized(graph):
        raise ValueError("Dataset manifest graphHash, graph, or encoder mismatch")
    if type(manifest["synthetic"]) is not bool or not isinstance(manifest["source"], str) or not manifest["source"].strip():
        raise ValueError("Dataset source and synthetic provenance must be explicit")
    generation_nodes, models, procedural = validate_model_provenance(manifest, graph, allow_procedural)
    has_text = any(node["op"] == "generate_text" for node in generation_nodes.values())
    if has_text and not isinstance(manifest["samples"], list):
        raise ValueError("Text generation requires per-sample provenance, not just a sample count")
    if encoder != LEGACY_ENCODER and not isinstance(manifest["samples"], list):
        raise ValueError("UPLv2 datasets require per-sample frozen input bundles")
    with np.load(path, allow_pickle=False) as archive:
        if not {"images", "upl", "split", "teacher_seed"}.issubset(archive.files):
            raise ValueError("Dataset requires images, upl, split, teacher_seed arrays")
        arrays = {key: archive[key].copy() for key in ("images", "upl", "split", "teacher_seed")}
    images, profiles, split, seeds = (arrays[key] for key in ("images", "upl", "split", "teacher_seed"))
    count = len(images)
    if images.dtype != np.float32 or images.shape != (count, 3, 32, 32) or not np.isfinite(images).all() or ((images < 0) | (images > 1)).any():
        raise ValueError("Dataset images must be float32 [N,3,32,32] in [0,1]")
    if profiles.dtype != np.float32 or profiles.shape != (count, encoder["dimension"]):
        raise ValueError("Dataset UPL must be float32 [N, encoder.dimension]")
    conditioning_tensor(profiles, encoder, batched=True)
    if split.shape != (count,) or split.dtype.kind not in "iu" or not np.isin(split, (0, 1)).all():
        raise ValueError("Dataset split must be integer [N], 0=train and 1=heldout")
    if seeds.shape != (count,) or seeds.dtype.kind not in "iu" or ((seeds < 0) | (seeds >= 2 ** 53)).any():
        raise ValueError("Dataset teacher_seed must be nonnegative JavaScript-safe integer [N]")
    sample_count = len(manifest["samples"]) if isinstance(manifest["samples"], list) else manifest["samples"]
    if type(sample_count) is not int or sample_count != count:
        raise ValueError("Dataset manifest sample count does not match arrays")
    if manifest.get("graphComplete") is False:
        raise ValueError("Dataset must contain complete graph outputs")
    person_sets = [set(), set()]
    if isinstance(manifest["samples"], list):
        for index, sample in enumerate(manifest["samples"]):
            if not isinstance(sample, dict):
                raise ValueError("Dataset sample provenance must be objects")
            expected = {"index": index, "upl": profiles[index].tolist(), "split": int(split[index]), "teacherSeed": int(seeds[index])}
            if has_text and not set(expected).issubset(sample):
                raise ValueError("Text generation requires complete profile/seed sample bindings")
            if any(key in sample and sample[key] != value for key, value in expected.items()):
                raise ValueError("Dataset sample provenance does not match arrays")
            if encoder != LEGACY_ENCODER:
                if not set(expected).issubset(sample) or sample.get("encoder") != encoder:
                    raise ValueError("UPLv2 sample must bind encoder, vector, split and teacher seed")
                profile_id = sample.get("profileId")
                if (not isinstance(profile_id, str) or not profile_id.strip() or profile_id != profile_id.strip()
                        or len(profile_id) > 128 or any(ord(character) < 32 or ord(character) == 127 for character in profile_id)):
                    raise ValueError("UPLv2 sample requires a stable person profileId, not a vector or audience hash")
                person_sets[int(split[index])].add(profile_id)
                frozen_upl(sample, graph["id"])
            validate_sample_generations(sample, generation_nodes, models, int(seeds[index]), graph)
    if encoder != LEGACY_ENCODER:
        if person_sets[0] & person_sets[1]:
            raise ValueError("Training and heldout person profileIds must be disjoint, even when their UPL vectors differ")
        if any(len(people) < 2 for people in person_sets):
            raise ValueError("Each UPLv2 split needs at least two distinct person profileIds")
    profile_sets = [set(map(tuple, profiles[split == partition].tolist())) for partition in (0, 1)]
    if any(len(values) < 2 for values in profile_sets):
        raise ValueError("Each split needs at least two distinct UPLs")
    if profile_sets[0] & profile_sets[1]:
        raise ValueError("Training and heldout UPLs must be disjoint")
    if any(len(np.unique(seeds[split == partition])) < 2 for partition in (0, 1)):
        raise ValueError("Each split needs diversified teacher seeds")
    pairs = [(tuple(profile), int(seed)) for profile, seed in zip(profiles.tolist(), seeds)]
    if len(set(pairs)) != count:
        raise ValueError("Duplicate UPL/teacher_seed examples are not seed diversification")
    provenance = {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "manifest": manifest, "proceduralFixture": procedural,
        "trainProfileCount": len(profile_sets[0]), "heldoutProfileCount": len(profile_sets[1]),
        "teacherSeedCount": len(np.unique(seeds)),
        "trainTeacherSeedCount": len(np.unique(seeds[split == 0])),
        "heldoutTeacherSeedCount": len(np.unique(seeds[split == 1])),
    }
    if encoder != LEGACY_ENCODER:
        provenance.update(trainProfileCount=len(person_sets[0]), heldoutProfileCount=len(person_sets[1]),
                          trainUplCount=len(profile_sets[0]), heldoutUplCount=len(profile_sets[1]),
                          splitProfileIds={"train": sorted(person_sets[0]), "heldout": sorted(person_sets[1])},
                          splitPolicy="person profileId disjoint; never derive identity from changing vectors, facts or recent intent")
    return arrays, provenance


def generate_procedural_fixture(output, graph, compiler_hash, canonical_path, seed=2026, train_count=16, heldout_count=4):
    """Explicit deterministic full-graph fixture, not a generative teacher.

    teacher_seed is retained for interchange but does not affect this procedural
    teacher. Only inference noise in the actual diffusion student is stochastic.
    """
    if conditioning_encoder(graph.get("encoder")) != LEGACY_ENCODER:
        raise ValueError("Procedural profile generation is legacy-only; use explicit frozen UPLv2 data")
    verified = verify_canonical_graph(graph, compiler_hash, canonical_path)
    validate_seed(seed)
    output = Path(output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise ValueError("Fixture output or manifest already exists")
    training, heldout = profile_splits(seed, train_count, heldout_count)
    profiles = torch.cat((training, heldout))
    teacher = TensorTeacher(graph)
    coordinates = pixel_coordinates(32, 32)
    images = torch.stack([teacher(profile.expand(1024, -1), coordinates).T.reshape(3, 32, 32) for profile in profiles])
    seeds = (np.arange(len(profiles), dtype=np.uint64) + seed) % 2 ** 32
    manifest = {
        "graphHash": verified, "encoder": copy.deepcopy(graph["encoder"]), "graph": copy.deepcopy(graph),
        "models": [], "samples": len(profiles), "source": "procedural-fixture-explicit", "synthetic": True,
        "teacherSeedAffectsImages": False, "graphComplete": True,
        "limitations": "Deterministic procedural full-graph fixture; not open-weight generated images.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        np.savez_compressed(handle, images=images.numpy().astype(np.float32), upl=profiles.numpy(), split=np.array([0] * train_count + [1] * heldout_count, dtype=np.int64), teacher_seed=seeds.astype(np.int64))
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, allow_nan=False)
    return output


def evaluation_bank(arrays, seed, count=128):
    heldout = np.flatnonzero(arrays["split"] == 1)
    generator = torch.Generator().manual_seed(seed)
    indices = heldout[np.arange(max(count, len(heldout))) % len(heldout)]
    clean = torch.from_numpy(arrays["images"][indices]) * 2 - 1
    profiles = torch.from_numpy(arrays["upl"][indices])
    timesteps = torch.arange(len(indices)) % 64
    timesteps = timesteps[torch.randperm(len(indices), generator=generator)]
    noise = torch.randn(clean.shape, generator=generator)
    return clean, profiles, timesteps, noise


def shuffled_conditioning_bank(bank):
    """Change every conditioning UPL, retaining identical clean/noise/time pairs."""
    clean, profiles, timesteps, noise = bank
    unique = torch.unique(profiles, dim=0)
    mismatched = torch.empty_like(profiles)
    for index, profile in enumerate(profiles):
        matched = torch.nonzero((unique == profile).all(dim=1), as_tuple=False)[0, 0].item()
        mismatched[index] = unique[(matched + 1) % len(unique)]
    return clean, mismatched, timesteps, noise


@torch.no_grad()
def evaluate_noise(model, bank, schedule, device, batch_size=16):
    alphas = torch.tensor(schedule["alphaCumprod"], device=device)
    total = 0.0
    clean, profiles, timesteps, noise = bank
    for start in range(0, len(clean), batch_size):
        target = clean[start:start + batch_size].to(device)
        upl = profiles[start:start + batch_size].to(device)
        time_batch = timesteps[start:start + batch_size].to(device)
        epsilon = noise[start:start + batch_size].to(device)
        alpha = alphas[time_batch, None, None, None]
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * epsilon
        prediction = model(noisy, upl, time_batch)
        total += functional.mse_loss(prediction, epsilon, reduction="sum").item()
    result = total / noise.numel()
    if not math.isfinite(result):
        raise ValueError("Heldout noise prediction loss became non-finite")
    return result


def graph_brand(graph):
    brands = [node for node in graph["nodes"] if node["op"] == "brand"]
    if len(brands) > 1 or brands and brands[0]["id"] != graph["output"]:
        raise ValueError("Exact brand fallback requires a single terminal brand node")
    return copy.deepcopy(brands[0]["params"]) if brands else None


@torch.no_grad()
def raster_evaluation(model, arrays, graph, schedule, artifact_id, graph_hash, seed):
    heldout = np.flatnonzero(arrays["split"] == 1)
    profiles = arrays["upl"]
    parameters = graph_brand(graph)
    coordinates = pixel_coordinates(32, 32)
    raw_cases = []
    maximum = 0.0
    probe_maximum = 0.0
    raster_mse = []
    for index in heldout[:16]:
        creative_seed = int(arrays["teacher_seed"][index]) % 2 ** 32
        raw, checks = sample_ddim(model, profiles[index], creative_seed, schedule, return_trace=True)
        raw = raw.cpu()
        colors = raw[0].permute(1, 2, 0).reshape(-1, 3)
        teacher = torch.from_numpy(arrays["images"][index]).permute(1, 2, 0).reshape(-1, 3)
        raster_mse.append(functional.mse_loss(colors, teacher).item())
        if parameters:
            plate, _ = brand_masks(coordinates, parameters)
            exact = apply_brand(colors, coordinates, parameters)
            if plate.any():
                maximum = max(maximum, (colors[plate] - exact[plate]).abs().max().item())
            probes = brand_probe_coordinates(parameters)
            pixel_indices = (probes * 32).floor().long().clamp(0, 31)
            sampled = raw[0, :, pixel_indices[:, 1], pixel_indices[:, 0]].T
            expected = apply_brand(sampled, probes, parameters)
            probe_maximum = max(probe_maximum, (sampled - expected).abs().max().item())
        raw_cases.append((index, creative_seed, raw, colors, teacher, checks))
    brand_error = max(maximum, probe_maximum)
    brand = {**parameters, "mode": "composite" if brand_error > BRAND_TOLERANCE else "neural"} if parameters else None
    cases = []
    composite_error = 0.0
    for index, creative_seed, raw, colors, teacher, checks in raw_cases[:2]:
        delivered = composite_rgb(raw, brand).reshape(-1, 3)
        if parameters:
            exact = apply_brand(delivered, coordinates, parameters)
            composite_error = max(composite_error, (delivered - exact).abs().max().item())
        cases.append({
            "upl": profiles[index].tolist(), "seed": creative_seed, "teacherSeed": int(arrays["teacher_seed"][index]),
            "datasetIndex": int(index), "width": 32, "height": 32,
            "teacherRgba": float_rgba(teacher), "rawStudentRgba": float_rgba(colors), "studentRgba": float_rgba(delivered),
            "teacherBytes": byte_rgba(teacher), "rawStudentBytes": byte_rgba(colors), "studentBytes": byte_rgba(delivered),
            "checks": checks,
        })
    first_profile = profiles[heldout[0]]
    second_profile = next(profiles[index] for index in heldout if not np.array_equal(profiles[index], first_profile))
    first = sample_ddim(model, first_profile, seed, schedule)
    different_seed = sample_ddim(model, first_profile, (seed + 1) % 2 ** 32, schedule)
    different_upl = sample_ddim(model, second_profile, seed, schedule)
    metrics = {
        "brandMaxError": brand_error, "brandNativeMaxError": maximum,
        "brandNearestProbeMaxError": probe_maximum, "brandTolerance": BRAND_TOLERANCE,
        "hasBrand": parameters is not None, "compositeBrandMaxError": composite_error,
        "rawSampleTeacherMse": sum(raster_mse) / len(raster_mse),
        "seedDiversityMse": functional.mse_loss(first, different_seed).item(),
        "uplChangeMse": functional.mse_loss(first, different_upl).item(),
        "rasterEvaluationCount": len(raw_cases), "evaluationWidth": 32, "evaluationHeight": 32,
    }
    fixture = {
        "format": "angelos.parity.v1" if model.encoder == LEGACY_ENCODER else "angelos.diffusion-python-checks.v2", "artifactId": artifact_id, "graphHash": graph_hash,
        "floatAbsoluteTolerance": 0.001, "byteAbsoluteTolerance": 1,
        "byteRounding": "floor(clamp(channel,0,1)*255+0.5)", "colorSpace": "display-encoded-rgb",
        "nativeResolution": 32, "rasterCases": cases,
        "teacherSource": "exact dataset images, not re-executed provider outputs",
        "seedDerivation": "fixture student seed = teacher_seed modulo 2^32; student and teacher are different generators",
        "checkOrder": "NCHW flattened first 16 values; nodes at first DDIM timestep; steps in schedule order",
    }
    return brand, metrics, fixture


def train_diffusion(graph, compiler_hash, dataset=None, canonical_path=None, steps=1000, device="cpu", seed=2026, batch_size=16, learning_rate=0.001, max_seconds=180, threads=2, procedural_fixture=False):
    verified = verify_canonical_graph(graph, compiler_hash, canonical_path)
    validate_seed(seed)
    if type(steps) is not int or steps < 1 or type(batch_size) is not int or batch_size < 2 or type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("Use positive integer steps, batch size >=2, and one to eight threads")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 600 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Use positive finite learning rate and max-seconds in (0,600]")
    target_device = resolve_device(device)
    if dataset is None:
        raise ValueError("An explicit --dataset is required; no procedural substitution")
    arrays, provenance = load_dataset(dataset, graph, verified, allow_procedural=procedural_fixture)
    graph_brand(graph)
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    started = time.monotonic()
    model = DiffusionStudent(graph["encoder"]).to(target_device)
    initial = torch.cat([parameter.detach().cpu().flatten() for parameter in model.parameters()])
    schedule = cosine_schedule()
    bank = evaluation_bank(arrays, (seed + 2) % 2 ** 32)
    baseline = evaluate_noise(model, bank, schedule, target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed((seed + 1) % 2 ** 32)
    training_indices = np.flatnonzero(arrays["split"] == 0)
    alphas = torch.tensor(schedule["alphaCumprod"], device=target_device)
    history = []
    completed = 0
    stop_reason = "steps-complete"
    model.train()
    for step in range(steps):
        if time.monotonic() - started >= max_seconds:
            stop_reason = "time-budget"
            break
        indices = training_indices[torch.randint(len(training_indices), (batch_size,), generator=generator).numpy()]
        clean = torch.from_numpy(arrays["images"][indices]).to(target_device) * 2 - 1
        upl = torch.from_numpy(arrays["upl"][indices]).to(target_device)
        timesteps = torch.randint(64, (batch_size,), generator=generator).to(target_device)
        epsilon = torch.randn(clean.shape, generator=generator).to(target_device)
        alpha = alphas[timesteps, None, None, None]
        noisy = alpha.sqrt() * clean + (1 - alpha).sqrt() * epsilon
        loss = functional.mse_loss(model(noisy, upl, timesteps), epsilon)
        if not torch.isfinite(loss):
            raise ValueError("Training loss became non-finite; refusing export")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
        optimizer.step()
        completed = step + 1
        if step == 0 or completed % 100 == 0 or completed == steps:
            history.append({"step": completed, "sampledTrainingNoiseMse": loss.item()})
    if not completed:
        raise ValueError("Time budget expired before any weight updates")
    model.eval()
    heldout = evaluate_noise(model, bank, schedule, target_device)
    final = torch.cat([parameter.detach().cpu().flatten() for parameter in model.parameters()])
    delta = (final - initial).norm().item()
    if not torch.isfinite(final).all() or not math.isfinite(delta) or delta <= 0:
        raise ValueError("Training must produce finite weights and a nonzero weight update")
    if heldout >= baseline:
        raise ValueError("Heldout noise prediction did not improve over untrained; refusing export")
    shuffled_loss = evaluate_noise(model, shuffled_conditioning_bank(bank), schedule, target_device)
    artifact_id = f"{graph['id']}-diffusion-{uuid.uuid4().hex}"
    brand, raster_metrics, fixture = raster_evaluation(model, arrays, graph, schedule, artifact_id, verified, seed)
    metrics = {
        "baselineMse": baseline, "heldoutMse": heldout, "improvedOverUntrained": heldout < baseline,
        "lossSemantics": "heldout epsilon (noise) prediction MSE; not image reconstruction or image quality",
        "relativeMseImprovement": (baseline - heldout) / baseline, "parameterDeltaL2": delta,
        "heldoutShuffledUplMse": shuffled_loss, "conditioningNoiseMseGain": shuffled_loss - heldout,
        "improvedWithCorrectUpl": shuffled_loss > heldout,
        "heldoutProfileCount": provenance["heldoutProfileCount"], **raster_metrics,
    }
    training = {
        "seed": seed, "steps": completed, "requestedSteps": steps, "device": device, "threads": threads,
        "batchSize": batch_size, "learningRate": learning_rate, "optimizer": "Adam", "loss": "epsilon MSE",
        "maxSeconds": max_seconds, "stopReason": stop_reason, "elapsedSeconds": time.monotonic() - started,
        "graphHashVerification": "compiler-canonical-file", "source": provenance["manifest"]["source"],
        "synthetic": provenance["manifest"]["synthetic"], "models": provenance["manifest"]["models"],
        "datasetSha256": provenance["sha256"], "manifestSha256": provenance["manifestSha256"],
        "proceduralFixture": provenance["proceduralFixture"], "teacherIncludesBrand": brand is not None,
        "trainProfileCount": provenance["trainProfileCount"], "heldoutProfileCount": provenance["heldoutProfileCount"],
        "trainSampleCount": int((arrays["split"] == 0).sum()), "heldoutSampleCount": int((arrays["split"] == 1).sum()),
        "teacherSeedCount": provenance["teacherSeedCount"], "trainTeacherSeedCount": provenance["trainTeacherSeedCount"],
        "heldoutTeacherSeedCount": provenance["heldoutTeacherSeedCount"],
        "noiseSchedule": {"name": "cosine", "offset": 0.008, "maxBeta": 0.999, "terminalAlpha": schedule["alphaCumprod"][-1]},
        "evaluationNoiseSeed": (seed + 2) % 2 ** 32, "evaluationNoiseExamples": len(bank[0]),
        "encoderFrozen": True, "pretrained": False, "promoted": False,
        "browserExportSupported": model.encoder == LEGACY_ENCODER,
        "fixtureOnly": provenance["proceduralFixture"] or model.encoder != LEGACY_ENCODER,
        "nativeResolution": 32, "meets1080pRequirement": False,
        "conditioning": "legacy-four-feature" if model.encoder == LEGACY_ENCODER else "semantic-vector-only; frozen recentIntent retained in dataset provenance, not adapter features",
        "torchVersion": torch.__version__, "numpyVersion": np.__version__, "packageVersion": __version__,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    artifact = {
        "format": model.artifact_format, "id": artifact_id, "campaignId": graph["id"], "graphHash": verified,
        "encoder": copy.deepcopy(graph["encoder"]), "architecture": copy.deepcopy(model.architecture),
        "layers": model.export_layers(), "schedule": schedule, "brand": brand, "metrics": metrics, "training": training,
    }
    report = {
        "format": "angelos.training-metrics.v1", "artifactId": artifact_id, "graphHash": verified,
        "metrics": metrics, "training": training, "dataset": provenance, "history": history,
        "trainProfiles": np.unique(arrays["upl"][arrays["split"] == 0], axis=0).tolist(),
        "heldoutProfiles": np.unique(arrays["upl"][arrays["split"] == 1], axis=0).tolist(),
        "brandDecision": {"mode": brand["mode"] if brand else None, "threshold": BRAND_TOLERANCE,
                          "reason": "raw DDIM samples measured against exact plate, including nearest-native cell probes",
                          "scope": "at most 16 heldout native rasters; exact fallback at requested resolution, not learned text"},
        "limitations": [
            "Noise MSE improvement is not evidence of image quality, text fidelity, or preference improvement.",
            "32x32 native generation cannot reproduce exact fine text; scaling adds no generated detail.",
            "Standard cosine beta cap .999 gives near-zero terminal alpha; untrained epsilon errors amplify and x0 clipping is expected.",
            "Heldout noise bank is fixed before training; no heldout examples are optimized against.",
            "Graph, pinned-model, and sample provenance consistency is not independent provider attestation of the supplied pixels.",
            "Seed diversity and UPL changes are numerical checks, not creative quality scores.",
            "Correct versus mismatched heldout UPL noise loss audits useful conditioning; merely changing pixels is not proof of useful personalization.",
            "The training deadline is checked between steps; evaluation and serialization add work.",
            "The Python sampler is an explicit reference, not evidence of browser GPU execution.",
        ],
    }
    return artifact, report, fixture


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--graph-canonical", required=True)
    parser.add_argument("--graph-hash", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--procedural-fixture", action="store_true", help="Explicitly allow procedural fixtures, generating --dataset only if missing")
    arguments = parser.parse_args(argv)
    try:
        graph = json.loads(Path(arguments.graph).read_text(encoding="utf-8"))
        if arguments.procedural_fixture and not Path(arguments.dataset).exists():
            generate_procedural_fixture(arguments.dataset, graph, arguments.graph_hash, arguments.graph_canonical, arguments.seed)
        artifact, report, fixture = train_diffusion(
            graph, arguments.graph_hash, dataset=arguments.dataset, canonical_path=arguments.graph_canonical,
            steps=arguments.steps, device=arguments.device, seed=arguments.seed, batch_size=arguments.batch_size,
            learning_rate=arguments.learning_rate, max_seconds=arguments.max_seconds, threads=arguments.threads,
            procedural_fixture=arguments.procedural_fixture,
        )
        paths = write_artifacts(arguments.output, artifact, report, fixture)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({"output": str(paths[0]), "metrics": str(paths[1]), "parity": str(paths[2]),
                      "steps": artifact["training"]["steps"], "baselineMse": artifact["metrics"]["baselineMse"],
                      "heldoutMse": artifact["metrics"]["heldoutMse"], "improvedOverUntrained": artifact["metrics"]["improvedOverUntrained"],
                      "brand": artifact["brand"]["mode"] if artifact["brand"] else None, "promoted": False}, allow_nan=False))


if __name__ == "__main__":
    main()
