"""Explicit offline candidate experiments; never scheduled and never auto-promoted."""

import argparse
import copy
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from .diffusion import DiffusionStudent, LEGACY_ENCODER, conditioning_encoder
from .feedback import artifact_bank, materialize_feedback
from .rewards import export_bundle, fit_reward_bundle
from .rl_policy import PolicyConfig, evaluate_candidate, optimize_policy
from .graph import graph_hash
from .train_diffusion import evaluation_bank, evaluate_noise, load_dataset, raster_evaluation, resolve_device


def synthetic_evidence(seed=2026, encoder=None):
    """Fabricated brightness-preference labels; not real people, clicks or imagery."""
    encoder = conditioning_encoder(encoder)
    generator = torch.Generator().manual_seed(seed)
    clicks, reviews = [], []
    for profile_index in range(16):
        vector = torch.rand(encoder["dimension"], generator=generator)
        if encoder != LEGACY_ENCODER:
            vector = functional.normalize(vector * 2 - 1, dim=0)
        upl = vector.tolist()
        for variant in range(8):
            brightness = 0.05 + variant * 0.12
            image = torch.full((3, 32, 32), brightness)
            shared = {"image": image, "upl": upl, "encoder": copy.deepcopy(encoder), "profileId": f"sim-{profile_index:02d}", "creativeId": f"sim-{profile_index:02d}-{variant}", "source": "synthetic"}
            clicks.append({**shared, "label": int(variant >= 4)})
            reviews.append({**shared, "quality": 0.1 + brightness * 0.8, "brand": float(variant >= 3)})
    return clicks, reviews


def unique_profiles(records, allowed):
    values = {}
    for record in records:
        if record["profileId"] in allowed:
            values.setdefault(record["profileId"], record["upl"])
    return list(values.values())


def run_experiment(*, parent_path, dataset, output, rounds=4, candidates=2, seed=2026,
                   feedback_path=None, artifact_paths=(), synthetic_demo=False, reward_steps=200,
                   device="cpu", quality_floor=0.6, brand_floor=0.8):
    if type(synthetic_demo) is not bool or type(candidates) is not int or not 1 <= candidates <= 4:
        raise ValueError("Choose 1–4 candidates and an explicit fixture mode")
    if synthetic_demo == (feedback_path is not None):
        raise ValueError("Choose exactly one: a fresh feedback export or --synthetic-demo")
    destination = Path(output)
    if destination.exists():
        raise ValueError("Experiment output must be a new directory")
    config = PolicyConfig(rounds=rounds, quality_floor=quality_floor, brand_floor=brand_floor).validate()
    torch.set_num_threads(2)
    target_device = resolve_device(device)
    parent_bytes = Path(parent_path).read_bytes()
    parent_artifact = json.loads(parent_bytes)
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    parent = DiffusionStudent.from_artifact(parent_artifact, target_device)
    manifest = json.loads(Path(dataset).with_suffix(".manifest.json").read_text())
    graph = manifest["graph"]
    if graph["id"] != parent_artifact["campaignId"] or manifest["graphHash"] != parent_artifact["graphHash"] or graph_hash(graph) != parent_artifact["graphHash"]:
        raise ValueError("Teacher anchor corpus must match parent campaign and graph")
    if parent.encoder != graph["encoder"]:
        raise ValueError("Teacher and parent encoder contracts must match exactly")
    arrays, provenance = load_dataset(dataset, graph, parent_artifact["graphHash"], synthetic_demo)
    if synthetic_demo:
        click_records, review_records = synthetic_evidence(seed, parent.encoder)
        feedback_hash = None
    else:
        feedback_bytes = Path(feedback_path).read_bytes()
        feedback = json.loads(feedback_bytes)
        if not 0 <= int(time.time() * 1000) - feedback.get("asOf", -1) <= 3600000:
            raise ValueError("Re-export consent-filtered feedback within one hour of starting training")
        click_records, review_records = materialize_feedback(feedback, artifact_bank([parent_path, *artifact_paths]), parent=parent_artifact)
        if sum(record["source"] == "operator" for record in review_records) < 100:
            raise ValueError("At least 100 operator reviews are required outside explicit synthetic demonstrations")
        feedback_hash = hashlib.sha256(feedback_bytes).hexdigest()
    rewards = fit_reward_bundle(click_records, review_records, seed=seed, steps=reward_steps, allow_synthetic=synthetic_demo, encoder=parent.encoder)
    train_profiles = unique_profiles(click_records + review_records, rewards.split_profiles["train"])[:8]
    audit_profiles = unique_profiles(click_records + review_records, rewards.split_profiles["validation"])[:16]
    if len(audit_profiles) < 2:
        raise ValueError("Independent profile audit requires at least two held-out groups")
    bank = evaluation_bank(arrays, seed + 3, count=32)
    parent_noise = evaluate_noise(parent, bank, parent_artifact["schedule"], target_device)
    indices = np.flatnonzero(arrays["split"] == 0)
    alphas = torch.tensor(parent_artifact["schedule"]["alphaCumprod"], device=target_device)

    def teacher_anchor(model, round_index):
        generator = torch.Generator().manual_seed(seed + 77 + round_index)
        selected = indices[torch.randint(len(indices), (4,), generator=generator).numpy()]
        clean = torch.from_numpy(arrays["images"][selected]).to(target_device) * 2 - 1
        profiles = torch.from_numpy(arrays["upl"][selected]).to(target_device)
        timesteps = torch.randint(64, (4,), generator=generator).to(target_device)
        noise = torch.randn(clean.shape, generator=generator).to(target_device)
        alpha = alphas[timesteps, None, None, None]
        return functional.mse_loss(model(alpha.sqrt() * clean + (1 - alpha).sqrt() * noise, profiles, timesteps), noise)

    destination.mkdir(parents=True, exist_ok=False)
    export_bundle(rewards, destination / "rewards.json")
    report = {
        "format": "angelos.rl-experiment.v1", "createdAt": datetime.now(timezone.utc).isoformat(),
        "parent": {"id": parent_artifact["id"], "sha256": parent_hash}, "graphHash": parent_artifact["graphHash"],
        "campaignId": parent_artifact["campaignId"], "rewardId": rewards.id,
        "encoder": parent.encoder, "datasetManifestSha256": provenance["manifestSha256"],
        "conditioning": "semantic-vector-only; recentIntent retained in evidence, not adapter features" if parent.encoder != LEGACY_ENCODER else "legacy-four-feature",
        "feedbackSha256": feedback_hash, "datasetSha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        "fixtureOnly": synthetic_demo or parent.encoder != LEGACY_ENCODER, "outcome": "click", "actualConversionsUsed": False,
        "syntheticEvidence": synthetic_demo, "browserExportSupported": parent.encoder == LEGACY_ENCODER,
        "nativeResolution": 32, "meets1080pRequirement": False, "autoPromoted": False,
        "rewardValidation": rewards.metadata["metrics"], "candidates": [],
        "limitations": ["A numerical RL regression, not the target 1080p model.",
                         "Candidate improvement is surrogate prediction, never measured CTR lift.",
                         "Repeated candidate comparison on this audit bank needs a fresh confirmatory experiment.",
                         "The pooled critic cannot certify logo, text, product or other fine-detail fidelity."],
    }
    reward_metrics = rewards.metadata["metrics"]
    reward_valid = all(reward_metrics[task][loss] < reward_metrics[task]["baseline_" + loss]
                       for task, loss in (("click", "brier"), ("quality", "mse"), ("brand", "bce")))
    for candidate_index in range(candidates):
        candidate_seed = (seed + candidate_index * 10007) % 2 ** 32
        candidate, optimization = optimize_policy(parent, train_profiles, rewards, schedule=parent_artifact["schedule"],
                                                 brand=parent_artifact["brand"], config=config, seed=candidate_seed, anchor=teacher_anchor)
        audit = evaluate_candidate(parent, candidate, audit_profiles, rewards, schedule=parent_artifact["schedule"],
                                   brand=parent_artifact["brand"], config=config, seed=(seed + 50000) % 2 ** 32)
        candidate_noise = evaluate_noise(candidate, bank, parent_artifact["schedule"], target_device)
        candidate_id = f"{graph['id']}-rl-{uuid.uuid4().hex}"
        _, raster_metrics, _ = raster_evaluation(candidate, arrays, graph, parent_artifact["schedule"], candidate_id, parent_artifact["graphHash"], candidate_seed)
        audit["gates"].update({"rewardGeneralization": reward_valid, "nonzeroWeightUpdate": optimization["weightDeltaL2"] > 0,
                              "teacherNoiseNonRegression": candidate_noise <= parent_noise * 1.05,
                              "deliveryBrandValid": parent_artifact["brand"] is None or (raster_metrics["compositeBrandMaxError"] <= raster_metrics["brandTolerance"] if parent_artifact["brand"]["mode"] == "composite" else raster_metrics["brandMaxError"] <= raster_metrics["brandTolerance"]),
                              "notSyntheticEvidence": not synthetic_demo})
        if parent.encoder != LEGACY_ENCODER:
            audit["gates"]["notExperimentalConditioning"] = False
        audit["eligibleForFurtherTesting"] = all(audit["gates"].values())
        artifact = copy.deepcopy(parent_artifact)
        artifact["id"] = candidate_id
        artifact["layers"] = candidate.export_layers()
        artifact["metrics"] = {**raster_metrics, "baselineMse": parent_noise, "heldoutMse": candidate_noise,
                               "baselineKind": "frozen-parent-on-matched-teacher-holdout", "improvedOverUntrained": False,
                               "improvedOverParentNoise": candidate_noise < parent_noise, "rlGatePassed": audit["eligibleForFurtherTesting"]}
        artifact["training"] = {"seed": candidate_seed, "steps": len(optimization["history"]), "device": str(target_device),
                                "method": optimization["algorithm"], "parentId": parent_artifact["id"], "parentSha256": parent_hash,
                                "rewardId": rewards.id, "feedbackSha256": feedback_hash, "fixtureOnly": report["fixtureOnly"],
                                "datasetSha256": provenance["sha256"], "manifestSha256": provenance["manifestSha256"],
                                "browserExportSupported": parent.encoder == LEGACY_ENCODER,
                                "conditioning": report["conditioning"], "meets1080pRequirement": False,
                                "nativeResolution": 32, "createdAt": report["createdAt"], "promotion": "candidate-only"}
        filename = f"candidate-{candidate_index + 1}.student.json"
        payload = json.dumps(artifact, allow_nan=False, separators=(",", ":")).encode()
        (destination / filename).write_bytes(payload)
        report["candidates"].append({"id": candidate_id, "file": filename, "sha256": hashlib.sha256(payload).hexdigest(),
                                     "optimization": optimization, "audit": audit, "status": "eligible-for-manual-test" if audit["eligibleForFurtherTesting"] else "rejected"})
    with (destination / "experiment.json").open("x") as handle:
        json.dump(report, handle, allow_nan=False, indent=2)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feedback")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--synthetic-demo", action="store_true")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--reward-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--quality-floor", type=float, default=0.6)
    parser.add_argument("--brand-floor", type=float, default=0.8)
    arguments = parser.parse_args(argv)
    try:
        report = run_experiment(parent_path=arguments.parent, dataset=arguments.dataset, output=arguments.output,
                                feedback_path=arguments.feedback, artifact_paths=arguments.artifact, synthetic_demo=arguments.synthetic_demo,
                                rounds=arguments.rounds, candidates=arguments.candidates, reward_steps=arguments.reward_steps, seed=arguments.seed,
                                device=arguments.device, quality_floor=arguments.quality_floor, brand_floor=arguments.brand_floor)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"Candidate experiment failed: {error}\n")
    print(json.dumps({"output": arguments.output, "fixtureOnly": report["fixtureOnly"], "candidates": [{"file": candidate["file"], "status": candidate["status"]} for candidate in report["candidates"]], "promoted": False}))


if __name__ == "__main__":
    main()
