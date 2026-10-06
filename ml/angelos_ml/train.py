"""CPU-bounded supervised full-graph distillation; artifacts are never promoted."""

import argparse
import copy
import hashlib
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import __version__
from .data import profile_splits, profiles_hash, sample_batch
from .evaluate import BRAND_TOLERANCE, evaluate, parity_fixture
from .graph import load_graph, verify_graph_hash
from .model import Student
from .teacher import TensorTeacher


def train_student(
    graph,
    compiler_hash,
    steps=1200,
    device="cpu",
    seed=2026,
    batch_size=1024,
    train_count=128,
    heldout_count=32,
    width=128,
    height=96,
    learning_rate=0.001,
    max_seconds=180,
    threads=2,
    canonical_path=None,
):
    verified_hash = verify_graph_hash(graph, compiler_hash, canonical_path)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2 ** 53 - 1:
        raise ValueError("Seed must be a nonnegative JavaScript-safe integer")
    if steps < 1 or batch_size < 2 or not 1 <= threads <= 8:
        raise ValueError("Use positive steps, batch size >=2, and one to eight CPU threads")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 600:
        raise ValueError("Wall-clock budget must be in (0,600] seconds")
    if device != "cpu":
        raise ValueError("This resource-bounded first slice supports --device cpu only")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    started = time.monotonic()
    teacher = TensorTeacher(graph)
    training_profiles, heldout_profiles = profile_splits(seed, train_count, heldout_count)
    model = Student().to(device)
    initial_parameters = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()]).clone()
    baseline = evaluate(model, teacher, heldout_profiles, width, height, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    history = []
    completed_steps = 0
    stop_reason = "steps-complete"
    for step in range(steps):
        if time.monotonic() - started >= max_seconds:
            stop_reason = "time-budget"
            break
        upl, coordinates = sample_batch(training_profiles, batch_size, generator, teacher.brand)
        upl, coordinates = upl.to(device), coordinates.to(device)
        with torch.no_grad():
            targets = teacher(upl, coordinates, validate=False)
        predictions = model(torch.cat((coordinates, upl), dim=1))
        loss = torch.nn.functional.mse_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise ValueError("Training loss became non-finite; no artifact was exported")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        optimizer.step()
        completed_steps = step + 1
        if step == 0 or completed_steps % 100 == 0 or completed_steps == steps:
            history.append({"step": completed_steps, "sampledTrainingMse": loss.item()})
    if completed_steps == 0:
        raise ValueError("Time budget expired before training; refusing to export untrained weights")
    model.eval()
    heldout = evaluate(model, teacher, heldout_profiles, width, height, device)
    training_audit = evaluate(model, teacher, training_profiles[:16], width, height, device)
    final_parameters = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    parameter_delta = (final_parameters - initial_parameters).norm().item()
    if not math.isfinite(parameter_delta) or parameter_delta == 0:
        raise ValueError("Training did not produce a finite nonzero weight update")
    brand = None
    brand_mode = "neural"
    if teacher.brand is not None:
        brand_mode = "composite" if heldout["brandMaxError"] > BRAND_TOLERANCE else "neural"
        brand = {"mode": brand_mode, **copy.deepcopy(teacher.brand)}
    artifact_id = f"{graph['id']}-{uuid.uuid4().hex}"
    metrics = {
        "baselineMse": baseline["mse"],
        "heldoutMse": heldout["mse"],
        "trainMse": training_audit["mse"],
        "heldoutMae": heldout["mae"],
        "heldoutCompositeMse": heldout["compositeMse"],
        "baselineBrandMaxError": baseline["brandMaxError"],
        "brandMaxError": heldout["brandMaxError"] if teacher.brand is not None else 0.0,
        "brandTolerance": BRAND_TOLERANCE,
        "hasBrand": teacher.brand is not None,
        "compositeBrandMaxError": heldout["compositeBrandMaxError"],
        "relativeMseImprovement": (baseline["mse"] - heldout["mse"]) / baseline["mse"] if baseline["mse"] > 0 else 0.0,
        "improvedOverUntrained": heldout["mse"] < baseline["mse"],
        "parameterDeltaL2": parameter_delta,
        "evaluationWidth": width,
        "evaluationHeight": height,
        "heldoutProfileCount": heldout_count,
    }
    training = {
        "seed": seed,
        "steps": completed_steps,
        "requestedSteps": steps,
        "device": device,
        "threads": threads,
        "batchSize": batch_size,
        "learningRate": learning_rate,
        "optimizer": "Adam",
        "loss": "full-RGB MSE on 75% uniform / 25% brand-plate coordinates; 100% uniform without brand",
        "trainProfileCount": train_count,
        "heldoutProfileCount": heldout_count,
        "trainProfilesHash": profiles_hash(training_profiles),
        "heldoutProfilesHash": profiles_hash(heldout_profiles),
        "profileSource": "fixed disjoint synthetic adult encoder outputs; no real personal data",
        "encoderFrozen": True,
        "teacherIncludesBrand": teacher.brand is not None,
        "maxSeconds": max_seconds,
        "stopReason": stop_reason,
        "elapsedSeconds": time.monotonic() - started,
        "graphHashVerification": "compiler-canonical-file" if canonical_path else "python-canonical-json",
        "torchVersion": torch.__version__,
        "numpyVersion": np.__version__,
        "packageVersion": __version__,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "pretrained": False,
        "promoted": False,
    }
    artifact = {
        "format": "angelos.student.v1",
        "id": artifact_id,
        "campaignId": graph["id"],
        "graphHash": verified_hash,
        "encoder": copy.deepcopy(graph["encoder"]),
        "inputOrder": ["x", "y", "upl.0", "upl.1", "upl.2", "upl.3"],
        "layers": model.export_layers(),
        "brand": brand,
        "metrics": metrics,
        "training": training,
    }
    report = {
        "format": "angelos.training-metrics.v1",
        "artifactId": artifact_id,
        "graphHash": verified_hash,
        "metrics": metrics,
        "training": training,
        "untrainedHeldout": baseline,
        "rawStudentHeldout": heldout,
        "rawStudentTrainingAudit": training_audit,
        "trainProfiles": training_profiles.tolist(),
        "heldoutProfiles": heldout_profiles.tolist(),
        "history": history,
        "brandDecision": {
            "mode": brand_mode if brand else None,
            "threshold": BRAND_TOLERANCE,
            "reason": "raw held-out maximum channel error exceeds tolerance" if brand_mode == "composite" else "no measured brand failure at the recorded raster and probes",
            "scope": "entire opaque plate; not a proof over all profiles or coordinates",
        },
        "limitations": [
            "Procedural synthetic teacher, not pretrained photorealism.",
            "Held-out profiles are never optimized against; the recorded brand rule uses their evaluation.",
            "No preference critic, click model, CTR estimate, or automatic promotion.",
            "Training deadline is checked between steps; final evaluation and export add bounded work.",
        ],
    }
    fixture = parity_fixture(model, teacher, heldout_profiles, artifact_id, verified_hash, brand_mode, device)
    return artifact, report, fixture


def write_artifacts(output, artifact, report, fixture):
    output = Path(output)
    paths = (output, output.with_suffix(".metrics.json"), output.with_suffix(".parity.json"))
    if any(path.exists() for path in paths):
        raise ValueError("Output or sidecar already exists; choose a new output path")
    payload = json.dumps(artifact, separators=(",", ":"), allow_nan=False) + "\n"
    report["artifactSha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    fixture["artifactSha256"] = report["artifactSha256"]
    payloads = [payload] + [json.dumps(value, indent=2, allow_nan=False) + "\n" for value in (report, fixture)]
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, serialized in zip(paths, payloads):
        with path.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--graph-hash", required=True, help="Authoritative 64-hex hash supplied by the graph compiler")
    parser.add_argument("--graph-canonical", help="Optional compiler canonical JSON for exact JS number serialization")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--train-profiles", type=int, default=128)
    parser.add_argument("--heldout-profiles", type=int, default=32)
    parser.add_argument("--eval-width", type=int, default=128)
    parser.add_argument("--eval-height", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args(argv)
    try:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", arguments.graph_hash):
            raise ValueError("--graph-hash must contain exactly 64 hexadecimal characters")
        graph = load_graph(arguments.graph)
        artifact, report, fixture = train_student(
            graph,
            arguments.graph_hash,
            steps=arguments.steps,
            device=arguments.device,
            seed=arguments.seed,
            batch_size=arguments.batch_size,
            train_count=arguments.train_profiles,
            heldout_count=arguments.heldout_profiles,
            width=arguments.eval_width,
            height=arguments.eval_height,
            learning_rate=arguments.learning_rate,
            max_seconds=arguments.max_seconds,
            threads=arguments.threads,
            canonical_path=arguments.graph_canonical,
        )
        paths = write_artifacts(arguments.output, artifact, report, fixture)
    except (ValueError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({
        "output": str(paths[0]),
        "metrics": str(paths[1]),
        "parity": str(paths[2]),
        "steps": artifact["training"]["steps"],
        "baselineMse": artifact["metrics"]["baselineMse"],
        "heldoutMse": artifact["metrics"]["heldoutMse"],
        "brand": artifact["brand"]["mode"] if artifact["brand"] else None,
        "promoted": False,
    }, allow_nan=False))


if __name__ == "__main__":
    main()
