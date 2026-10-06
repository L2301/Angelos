"""Lower a validated teacher to a fixed-batch ATen graph without training."""

import argparse
import copy
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

from . import __version__
from .data import profile_splits
from .graph import load_graph, verify_graph_hash
from .teacher import TensorTeacher, brand_masks, brand_probe_coordinates


PARITY_TOLERANCE = 0.000001
METADATA_FILE = "angelos.metadata.json"


class TeacherModule(nn.Module):
    def __init__(self, graph):
        super().__init__()
        self.teacher = TensorTeacher(graph)

    def forward(self, upl, coordinates):
        return self.teacher.color_at(upl, coordinates, validate=False)


def audit_aten_graph(program):
    operations = {}
    for node in program.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        schema = getattr(node.target, "_schema", None)
        if node.op != "call_function" or schema is None or not schema.name.startswith("aten::"):
            raise ValueError(f"Export contains a non-ATen operation: {node.op} {node.target}")
        name = str(node.target)
        operations[name] = operations.get(name, 0) + 1
    if not operations:
        raise ValueError("Export did not produce an ATen tensor graph")
    constants = {}
    for name, value in program.constants.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Export contains a non-tensor constant: {name}")
        constants[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    if any(True for _parameter in program.parameters()):
        raise ValueError("A procedural teacher export must not contain trainable parameters")
    return {
        "nodeCount": len(list(program.graph.nodes)),
        "operatorCounts": dict(sorted(operations.items())),
        "constants": constants,
        "parameterCount": 0,
    }


def parity_batches(teacher, samples, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    profiles, _heldout = profile_splits(seed + 2, max(samples, 2), 2)
    for _batch_index in range(2):
        indices = torch.randint(len(profiles), (samples,), generator=generator)
        yield profiles[indices], torch.rand((samples, 2), generator=generator)
    probes = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=torch.float32)
    if teacher.brand is not None:
        probes = torch.cat((probes, brand_probe_coordinates(teacher.brand)))
    for offset in range(0, len(probes), samples):
        coordinates = torch.rand((samples, 2), generator=generator)
        selected = probes[offset:offset + samples]
        coordinates[:len(selected)] = selected
        indices = torch.randint(len(profiles), (samples,), generator=generator)
        yield profiles[indices], coordinates


@torch.no_grad()
def verify_cpu_parity(module, teacher, samples, seed):
    maximum_error = 0.0
    total_error = 0.0
    total_channels = 0
    brand_maximum = 0.0
    brand_channels = 0
    batch_count = 0
    for upl, coordinates in parity_batches(teacher, samples, seed):
        expected = teacher.color_at(upl, coordinates)
        actual = module(upl, coordinates)
        if actual.shape != expected.shape or actual.dtype != torch.float32 or actual.device.type != "cpu":
            raise ValueError("Compiled CPU output violates shape, dtype, or device contract")
        if not torch.isfinite(actual).all():
            raise ValueError("Compiled CPU output contains non-finite values")
        difference = (actual - expected).abs()
        maximum_error = max(maximum_error, difference.max().item())
        total_error += difference.sum().item()
        total_channels += difference.numel()
        batch_count += 1
        if teacher.brand is not None:
            plate, _foreground = brand_masks(coordinates, teacher.brand)
            selected = difference[plate]
            if selected.numel():
                brand_maximum = max(brand_maximum, selected.max().item())
                brand_channels += selected.numel()
    if maximum_error > PARITY_TOLERANCE:
        raise ValueError(f"Loaded ATen teacher CPU parity failed: {maximum_error} > {PARITY_TOLERANCE}")
    return {
        "device": "cpu",
        "fixedBatchSize": samples,
        "batchCount": batch_count,
        "channelCount": total_channels,
        "maxError": maximum_error,
        "meanAbsoluteError": total_error / total_channels,
        "brandMaxError": brand_maximum if teacher.brand is not None else None,
        "brandChannelCount": brand_channels,
        "absoluteTolerance": PARITY_TOLERANCE,
        "passed": True,
        "coverage": "fresh synthetic adult profiles, random coordinates, corners, center, and every brand plate cell",
    }


def compile_teacher(graph, compiler_hash, samples=256, canonical_path=None, seed=2026, threads=2):
    started = time.monotonic()
    verified_hash = verify_graph_hash(graph, compiler_hash, canonical_path)
    if isinstance(samples, bool) or not isinstance(samples, int) or not 1 <= samples <= 4096:
        raise ValueError("Fixed export sample count must be an integer in [1,4096]")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2 ** 53 - 1:
        raise ValueError("Seed must be a nonnegative JavaScript-safe integer")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 8:
        raise ValueError("CPU thread count must be an integer in [1,8]")
    torch.set_num_threads(threads)
    module = TeacherModule(graph).eval().to("cpu")
    profiles, _heldout = profile_splits(seed, max(samples, 2), 2)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    upl = profiles[:samples]
    coordinates = torch.rand((samples, 2), generator=generator)
    export_started = time.monotonic()
    program = torch.export.export(module, (upl, coordinates), strict=True)
    export_seconds = time.monotonic() - export_started
    graph_audit = audit_aten_graph(program)
    metadata = {
        "format": "angelos.teacher-export.v1",
        "campaignId": graph["id"],
        "graphHash": verified_hash,
        "encoder": copy.deepcopy(graph["encoder"]),
        "graphHashVerification": "compiler-canonical-file" if canonical_path else "python-canonical-json",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "packageVersion": __version__,
        "torchVersion": torch.__version__,
        "lowering": "torch.export.export(strict=True)",
        "inputs": [
            {"name": "upl", "shape": [samples, 4], "dtype": "float32", "range": [0, 1]},
            {"name": "coordinates", "shape": [samples, 2], "dtype": "float32", "range": [0, 1]},
        ],
        "output": {"shape": [samples, 3], "dtype": "float32", "range": [0, 1], "colorSpace": "display-encoded-rgb"},
        "coordinateConvention": "normalized xy, origin top-left; pixel centers for raster callers",
        "fixedBatchSize": samples,
        "device": "cpu",
        "seed": seed,
        "threads": threads,
        "training": False,
        "pretrained": False,
        "interpreterFallback": False,
        "cudaTested": False,
        "brandIncluded": module.teacher.brand is not None,
        "runtimeValueValidation": "caller must validate finite normalized UPL/xy before invoking the exported module",
        "graph": graph_audit,
    }
    serialized = io.BytesIO()
    torch.export.save(program, serialized, extra_files={METADATA_FILE: json.dumps(metadata, allow_nan=False)})
    serialized.seek(0)
    loaded_metadata = {METADATA_FILE: ""}
    loaded = torch.export.load(serialized, extra_files=loaded_metadata)
    if json.loads(loaded_metadata[METADATA_FILE]) != metadata:
        raise ValueError("Exported teacher metadata did not survive serialization")
    if audit_aten_graph(loaded) != graph_audit:
        raise ValueError("ATen graph or constant metadata changed during serialization")
    parity_started = time.monotonic()
    parity = verify_cpu_parity(loaded.module(), module.teacher, samples, seed)
    parity_seconds = time.monotonic() - parity_started
    payload = serialized.getvalue()
    report = {
        "format": "angelos.teacher-compile-metrics.v1",
        "campaignId": graph["id"],
        "graphHash": verified_hash,
        "artifactSha256": hashlib.sha256(payload).hexdigest(),
        "artifactBytes": len(payload),
        "exportSeconds": export_seconds,
        "loadedCpuParitySeconds": parity_seconds,
        "elapsedSeconds": time.monotonic() - started,
        "cpuParity": parity,
        "metadata": metadata,
        "limitations": [
            "ATen graph lowering and serialization, not neural distillation or native machine-code compilation.",
            "Fixed sample count and CPU float32 inputs; no dynamic-shape or CUDA evidence.",
            "Finite numerical parity checks do not prove equality for every input.",
            "Only reload locally generated or otherwise trusted, matching-toolchain .pt2 artifacts.",
            "No unsupported-node, tracing, or parity failure is replaced by interpreter execution.",
        ],
    }
    return payload, report


def output_paths(output):
    output = Path(output)
    if output.suffix != ".pt2":
        raise ValueError("Compiled teacher output must have a .pt2 suffix")
    paths = (output, output.with_suffix(".compile.metrics.json"))
    if any(path.exists() for path in paths):
        raise ValueError("Compiled teacher output or metrics already exists; choose a new path")
    return paths


def write_compiled_teacher(output, payload, report):
    artifact_path, metrics_path = output_paths(output)
    encoded_report = json.dumps(report, indent=2, allow_nan=False) + "\n"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("xb") as handle:
        handle.write(payload)
    with metrics_path.open("x", encoding="utf-8") as handle:
        handle.write(encoded_report)
    return artifact_path, metrics_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--graph-hash", required=True, help="Authoritative 64-hex graph compiler hash")
    parser.add_argument("--graph-canonical", help="Compiler canonical JSON for exact JS number serialization")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=256, help="Fixed input batch size, not training examples")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args(argv)
    try:
        output_paths(arguments.output)
        graph = load_graph(arguments.graph)
        payload, report = compile_teacher(
            graph, arguments.graph_hash, samples=arguments.samples,
            canonical_path=arguments.graph_canonical, seed=arguments.seed, threads=arguments.threads,
        )
        artifact_path, metrics_path = write_compiled_teacher(arguments.output, payload, report)
    except Exception as error:
        parser.exit(2, f"Teacher compilation failed ({type(error).__name__}): {error}\nNo interpreter fallback was used.\n")
    print(json.dumps({
        "output": str(artifact_path),
        "metrics": str(metrics_path),
        "graphHash": report["graphHash"],
        "samples": arguments.samples,
        "device": "cpu",
        "maxError": report["cpuParity"]["maxError"],
        "brandMaxError": report["cpuParity"]["brandMaxError"],
        "elapsedSeconds": report["elapsedSeconds"],
        "cudaTested": False,
        "training": False,
    }, allow_nan=False))


if __name__ == "__main__":
    main()
