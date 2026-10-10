"""Offline synthetic semantic smoke test, not a dimension-selection benchmark.

Run from the repository root after separately provisioning the pinned checkpoint:
  PYTHONPATH=ml python -m angelos_ml.upl_evaluate --device cpu --output new-report.json

The cache-only encoder loads once and encodes each fictional adult once at 1024
dimensions. Comparisons use nested 128/256/512/1024 prefixes, each L2-normalized
again. No adaptation, fitting, downloads, ad generation, or outcome simulation
occurs. Person identifiers are retained for potential future holdouts, but this
small manually grouped corpus has no training/test split or generalization claim.
"""

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

from .upl import DIMENSIONS, FACT_FIELDS, canonical_json, encoder_metadata, load_local_encoder, semantic_text, strict_json, validate_facts


CORPUS_FORMAT = "angelos.upl-smoke-corpus.v1"
REPORT_FORMAT = "angelos.upl-dimension-smoke.v1"
DEFAULT_FIXTURE = Path(__file__).resolve().parents[2] / "examples" / "upl-aurora-evaluation.json"
MAX_CORPUS_BYTES = 65536
PROFILE_COUNT = 12
TIE_TOLERANCE = 1e-9
AGE_BANDS = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
REGIONS = ("NA", "EU", "APAC", "OTHER")
BASELINE_FIELDS = ("ageBand", "region", "openness", "energy")
LIMITATIONS = [
    "Synthetic semantic smoke only: expected groups are manually declared fiction, not human ground truth.",
    "Nearest-neighbor purity is not CTR, conversion lift, ad quality, or generalization evaluation.",
    "This corpus does not establish an optimal dimension or prove a dimension-selection decision.",
    "No adaptation, fitting, training split, holdout evaluation, or production-person inference is performed.",
    "The identical 4D baseline omits the varying semantic facts; it is not a no-UPL outcome benchmark.",
    "Fixed inputs, model revision and prefix rules aid replay; timings and numerical results may vary by device/runtime.",
]


def exact_fields(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError(f"{label} requires exactly: {', '.join(fields)}")


def validate_corpus(corpus):
    exact_fields(corpus, ("format", "synthetic", "adult", "campaignId", "labelSource", "profiles"), "Corpus")
    if corpus["format"] != CORPUS_FORMAT or corpus["synthetic"] is not True or corpus["adult"] is not True:
        raise ValueError("Only explicitly synthetic adult smoke corpora are permitted")
    if corpus["campaignId"] != "aurora" or corpus["labelSource"] != "manually-declared-synthetic-interest-groups":
        raise ValueError("Expected the fictional Aurora campaign and manually declared synthetic labels")
    profiles = corpus["profiles"]
    if type(profiles) is not list or len(profiles) != PROFILE_COUNT:
        raise ValueError("The bounded fixture requires exactly 12 fictional profiles")
    identifiers = set()
    groups = {}
    baseline = None
    for profile in profiles:
        exact_fields(profile, ("syntheticPersonId", "expectedGroup", "facts"), "Profile")
        identifier = profile["syntheticPersonId"]
        if type(identifier) is not str or re.fullmatch(r"synthetic-[a-z0-9-]{1,48}", identifier) is None or identifier in identifiers:
            raise ValueError("Each profile needs a unique, bounded syntheticPersonId")
        identifiers.add(identifier)
        group = profile["expectedGroup"]
        if type(group) is not str or re.fullmatch(r"[a-z][a-z0-9-]{0,47}", group) is None:
            raise ValueError("Expected groups must be bounded fictional labels")
        groups[group] = groups.get(group, 0) + 1
        facts = profile["facts"]
        exact_fields(facts, FACT_FIELDS, "Richer synthetic facts")
        validate_facts(facts)
        if facts["region"] not in REGIONS or any(type(facts[field]) not in (int, float) or not math.isfinite(facts[field]) for field in ("openness", "energy")):
            raise ValueError("The frozen baseline requires a region and finite numeric preferences")
        for field in ("interests", "goals", "preferredStyles"):
            if type(facts[field]) is not list or not 1 <= len(facts[field]) <= 4:
                raise ValueError(f"{field} requires one to four declared synthetic phrases")
        if facts["preferredLanguage"] != "English":
            raise ValueError("This bounded English-language fixture does not test multilingual performance")
        current = {field: facts[field] for field in BASELINE_FIELDS}
        if baseline is not None and current != baseline:
            raise ValueError("The four baseline facts must be identical across all profiles")
        baseline = current
    if len(groups) not in (3, 4) or any(count < 2 for count in groups.values()):
        raise ValueError("Use three or four manually declared groups with at least two fictional people each")
    return corpus


def read_corpus(path):
    with Path(path).open("rb") as stream:
        payload = stream.read(MAX_CORPUS_BYTES + 1)
    if len(payload) > MAX_CORPUS_BYTES:
        raise ValueError("Corpus exceeds the 65536-byte offline fixture limit")
    return validate_corpus(strict_json(payload.decode("utf-8"))), hashlib.sha256(payload).hexdigest()


def vector_statistics(vector):
    nonfinite = sum(not math.isfinite(value) for value in vector)
    norm = math.hypot(*vector) if not nonfinite else None
    return {"dimension": len(vector), "nonfiniteCount": nonfinite,
            "l2Norm": norm if norm is not None and math.isfinite(norm) else None,
            "valid": nonfinite == 0 and norm is not None and math.isfinite(norm) and norm > 0}


def normalized_prefix(vector, dimension):
    prefix = vector[:dimension]
    statistics = vector_statistics(prefix)
    if not statistics["valid"]:
        return None, statistics
    return [value / statistics["l2Norm"] for value in prefix], statistics


def nearest_neighbors(profiles, vectors):
    rows = []
    for profile_index, profile in enumerate(profiles):
        scores = []
        for candidate_index, candidate in enumerate(profiles):
            if profile_index == candidate_index:
                continue
            similarity = math.fsum(left * right for left, right in zip(vectors[profile_index], vectors[candidate_index]))
            scores.append({"syntheticPersonId": candidate["syntheticPersonId"], "expectedGroup": candidate["expectedGroup"],
                           "cosineSimilarity": max(-1.0, min(1.0, similarity)),
                           "sameDeclaredGroup": profile["expectedGroup"] == candidate["expectedGroup"]})
        best = max(candidate["cosineSimilarity"] for candidate in scores)
        tied = sorted((candidate for candidate in scores if best - candidate["cosineSimilarity"] <= TIE_TOLERANCE),
                      key=lambda candidate: candidate["syntheticPersonId"])
        purity = sum(candidate["sameDeclaredGroup"] for candidate in tied) / len(tied)
        rows.append({"syntheticPersonId": profile["syntheticPersonId"], "expectedGroup": profile["expectedGroup"],
                     "nearestNeighbors": tied, "nearestTieCount": len(tied), "fractionalDeclaredGroupPurity": purity})
    return {"meanFractionalDeclaredGroupPurity": math.fsum(row["fractionalDeclaredGroupPurity"] for row in rows) / len(rows),
            "profilesWithTies": sum(row["nearestTieCount"] > 1 for row in rows), "profiles": rows,
            "tiePolicy": "Include all nearest candidates within tolerance; purity is same-group count divided by tie count, never optimistic argmax.",
            "cosineTieTolerance": TIE_TOLERANCE, "selfNeighborsExcluded": True}


def dimension_report(profiles, source_vectors, dimension):
    normalized = []
    rows = []
    for profile, vector in zip(profiles, source_vectors):
        prefix, statistics = normalized_prefix(vector, dimension)
        source_valid = vector_statistics(vector)["valid"]
        if not source_valid:
            prefix = None
        normalized.append(prefix)
        rows.append({"syntheticPersonId": profile["syntheticPersonId"], "sourceVectorValid": source_valid,
                     "prefixStatistics": statistics, "normalizedL2Norm": math.hypot(*prefix) if prefix is not None else None,
                     "compactVectorJsonBytes": len(canonical_json(prefix).encode("utf-8")) if prefix is not None else None})
    valid = all(vector is not None for vector in normalized)
    return {"dimension": dimension, "status": "valid" if valid else "invalid-vectors",
            "normalization": "Take the first d coordinates of the same source vector, then L2-renormalize; no random projection or re-encoding.",
            "statistics": rows, "payload": {"scope": "Vector only; excludes facts, identifiers and provenance. FP32 sizes are storage equivalents, not measured wire compression.",
                "float32EquivalentBytesPerProfile": dimension * 4,
                "float32EquivalentBytesForCorpus": dimension * 4 * len(profiles),
                "compactVectorJsonBytesForCorpus": sum(row["compactVectorJsonBytes"] for row in rows) if valid else None},
            "semanticSmoke": nearest_neighbors(profiles, normalized) if valid else None}


def evaluate(fixture=DEFAULT_FIXTURE, *, device="cpu", cache_dir=None, encoder_loader=None, clock=time.perf_counter):
    evaluation_started = clock()
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("Choose an explicit cpu, cuda or mps device")
    corpus, corpus_hash = read_corpus(fixture)
    profiles = corpus["profiles"]
    injected = encoder_loader is not None
    loader = encoder_loader if injected else load_local_encoder
    load_started = clock()
    encoder = loader(device=device, cache_dir=cache_dir)
    load_seconds = clock() - load_started
    if str(getattr(encoder, "device", None)) != device:
        raise ValueError("Encoder device differs from the requested device; no silent fallback")
    vectors = []
    records = []
    for profile in profiles:
        text = semantic_text(profile["facts"])
        started = clock()
        vector = encoder.encode(text, 1024)
        elapsed = clock() - started
        if type(vector) is not list or len(vector) != 1024 or any(type(value) not in (int, float) for value in vector):
            raise ValueError("Encoder must return exactly 1024 numeric, non-boolean coordinates per profile")
        vectors.append(vector)
        records.append({"syntheticPersonId": profile["syntheticPersonId"], "expectedGroup": profile["expectedGroup"],
                        "semanticTextSha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "semanticTextBytes": len(text.encode("utf-8")), "encodeSeconds": elapsed,
                        "source1024Statistics": vector_statistics(vector)})
    ablation_started = clock()
    dimensions = [dimension_report(profiles, vectors, dimension) for dimension in DIMENSIONS]
    facts = profiles[0]["facts"]
    baseline_vector = [AGE_BANDS.index(facts["ageBand"]) / 5, REGIONS.index(facts["region"]) / 3, facts["openness"], facts["energy"]]
    baseline = dimension_report(profiles, [baseline_vector for profile in profiles], 4)
    baseline.update({"encoderId": "structured-demo-v1", "facts": {field: facts[field] for field in BASELINE_FIELDS},
                     "rawVectorForEveryProfile": baseline_vector, "allProfilesIdentical": True,
                     "interpretation": "These frozen four facts omit interests, goals and styles. Identical nonzero vectors tie; zero-vector cosine is undefined. This diagnoses missing information, not a no-UPL ad-performance baseline."})
    ablation_seconds = clock() - ablation_started
    return {"format": REPORT_FORMAT, "status": "valid" if all(item["status"] == "valid" for item in dimensions) else "invalid-vectors",
            "syntheticSemanticSmokeOnly": True, "limitations": LIMITATIONS,
            "corpus": {"format": corpus["format"], "sha256": corpus_hash,
                       "canonicalSha256": hashlib.sha256(canonical_json(corpus).encode("utf-8")).hexdigest(),
                       "profileCount": len(profiles), "groupCounts": {group: sum(profile["expectedGroup"] == group for profile in profiles)
                           for group in sorted({profile["expectedGroup"] for profile in profiles})},
                       "labelSource": corpus["labelSource"], "personIdRole": "Explicit fictional identity, separate from the group label; never an encoder input.",
                       "split": "None; identifiers retained only for possible future person-level holdouts.",
                       "encoderInputs": "semantic_text(facts) only; expectedGroup and syntheticPersonId are excluded."},
            "encoder": encoder_metadata(1024),
            "execution": {"mode": "injected-unit-fixture-not-real-model-output" if injected else "allowlisted-cached-checkpoint",
                          "localCacheOnly": None if injected else True, "requestedDevice": device, "modelDevice": str(encoder.device),
                          "checkpointLoaded": not injected, "modelLoads": 0 if injected else 1,
                          "encoderInstances": 1, "encodeCalls": len(records), "sourceDimension": 1024,
                          "torchThreads": None if injected else sys.modules["torch"].get_num_threads(),
                          "cacheRootOverride": str(cache_dir) if cache_dir is not None else None},
            "latency": {"unit": "seconds", "modelOrFixtureLoad": load_seconds,
                        "evaluationWall": clock() - evaluation_started,
                        "encodeTotal": math.fsum(record["encodeSeconds"] for record in records), "prefixAndNeighborAnalysis": ablation_seconds,
                        "scope": "Wall time includes fixture validation, one load, text construction, encoding and analysis; excludes interpreter startup and report writing. Sequential encode times include tokenization and CPU-list materialization. No warmup, throughput or cross-device speed claim."},
            "profiles": records, "dimensions": dimensions, "frozen4dBaseline": baseline}


def write_report(report, output):
    serialized = json.dumps(report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    with Path(output).open("x", encoding="utf-8") as stream:
        stream.write(serialized)


def main(argv=None):
    cli_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu", help="Explicit model device; no fallback (default: cpu).")
    parser.add_argument("--cache-dir", help="Already provisioned Hugging Face cache root; never downloads.")
    parser.add_argument("--threads", type=int, help="Optional explicit PyTorch intra-op thread count (1–64); no speed claim.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="Bounded 12-profile synthetic corpus (default: bundled Aurora fixture).")
    parser.add_argument("--output", type=Path, required=True, help="New JSON report file; existing files are never overwritten.")
    arguments = parser.parse_args(argv)
    try:
        if arguments.output.exists() or arguments.output.is_symlink():
            raise FileExistsError(f"Output already exists: {arguments.output}")
        if not arguments.output.parent.is_dir():
            raise ValueError("Output parent directory must already exist")
        if arguments.threads is not None:
            if not 1 <= arguments.threads <= 64:
                raise ValueError("--threads must be between 1 and 64")
            import torch
            torch.set_num_threads(arguments.threads)
        with contextlib.redirect_stdout(sys.stderr):
            report = evaluate(arguments.fixture, device=arguments.device, cache_dir=arguments.cache_dir)
        report["execution"]["requestedTorchThreads"] = arguments.threads
        report["latency"]["cliBeforeReportWrite"] = time.perf_counter() - cli_started
        write_report(report, arguments.output)
    except Exception as error:
        parser.exit(2, f"UPL smoke evaluation failed ({type(error).__name__}): {error}\n")
    if report["status"] != "valid":
        parser.exit(2, f"Invalid embedding vectors; diagnostic report written to {arguments.output}\n")
    print(f"Synthetic semantic smoke report: {arguments.output}; not a dimension-selection or CTR evaluation.")


if __name__ == "__main__":
    main()
