"""Reconstruct exposure-time creatives from hash-bound artifacts, never latest UPLs."""

import copy
import hashlib
import json
import re
from pathlib import Path

from .diffusion import DiffusionStudent, LEGACY_ENCODER, conditioning_encoder, conditioning_tensor, composite_rgb, sample_ddim


def validate_public_bundle(bundle):
    """Validate a frozen public bundle without loading or invoking an encoder."""
    from .upl import validate_facts

    fields = {"format", "synthetic", "adult", "factsSchema", "facts", "encoder", "upl", "context", "recentIntent"}
    if (not isinstance(bundle, dict) or set(bundle) != fields or bundle["format"] != "angelos.upl-input.v2"
            or bundle["factsSchema"] != "angelos.facts.v2" or bundle["synthetic"] is not True or bundle["adult"] is not True):
        raise ValueError("Expected a complete synthetic adult UPLv2 public bundle")
    validate_facts(bundle["facts"])
    encoder = conditioning_encoder(bundle["encoder"])
    if encoder == LEGACY_ENCODER:
        raise ValueError("UPLv2 bundles require a semantic encoder")
    conditioning_tensor(bundle["upl"], encoder)
    context = bundle["context"]
    if not isinstance(context, dict) or set(context) != {"campaignId", "factRevision", "eventRevision", "consentVersion", "asOf", "ingredientsHash"}:
        raise ValueError("Invalid frozen UPL context")
    if (not isinstance(context["campaignId"], str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", context["campaignId"]) is None
            or not isinstance(context["ingredientsHash"], str) or re.fullmatch(r"[a-f0-9]{64}", context["ingredientsHash"]) is None):
        raise ValueError("Invalid frozen UPL campaign or ingredients hash")
    for name in ("factRevision", "eventRevision", "consentVersion", "asOf"):
        minimum = 1 if name == "consentVersion" else 0
        if type(context[name]) is not int or not minimum <= context[name] < 2 ** 53:
            raise ValueError("Frozen UPL revisions and timestamps must be safe integers")
    events = bundle["recentIntent"]
    if not isinstance(events, list) or len(events) > 64:
        raise ValueError("Frozen recent intent must contain at most 64 events")
    for event in events:
        if not isinstance(event, dict) or set(event) != {"kind", "topic", "occurredAt", "expiresAt"}:
            raise ValueError("Invalid frozen recent-intent event")
        topic = event["topic"]
        if (event["kind"] not in ("view", "click", "save", "dismiss") or not isinstance(topic, str)
                or not topic or topic != topic.strip() or any(ord(character) < 32 or ord(character) == 127 or 0xD800 <= ord(character) <= 0xDFFF for character in topic)
                or len(topic.encode("utf-16-le")) // 2 > 80):
            raise ValueError("Invalid recent-intent kind or topic")
        if (type(event["occurredAt"]) is not int or not 0 <= event["occurredAt"] <= context["asOf"]
                or type(event["expiresAt"]) is not int or not context["asOf"] < event["expiresAt"] < 2 ** 53):
            raise ValueError("Recent intent must be active at the frozen snapshot time")
    return bundle


def frozen_upl(record, campaign_id=None):
    """Validate and retain the exposure-time encoder, vector and public input bundle."""
    encoder = conditioning_encoder(record.get("encoder"))
    conditioning_tensor(record.get("upl"), encoder)
    result = {"encoder": encoder, "upl": copy.deepcopy(record["upl"])}
    if encoder == LEGACY_ENCODER:
        if "uplBundle" in record:
            raise ValueError("Legacy UPL records cannot carry a semantic input bundle")
        return result
    bundle = copy.deepcopy(record.get("uplBundle"))
    validate_public_bundle(bundle)
    if bundle["encoder"] != encoder or bundle["upl"] != record["upl"]:
        raise ValueError("Frozen UPL bundle must match the complete row encoder and vector")
    context = bundle["context"]
    if campaign_id is not None and context["campaignId"] != campaign_id:
        raise ValueError("Frozen UPL bundle campaign mismatch")
    if "consentVersion" in record and record["consentVersion"] != context["consentVersion"]:
        raise ValueError("Frozen UPL bundle consent version mismatch")
    cutoff = record.get("occurredAt", record.get("receivedAt"))
    if cutoff is not None and context["asOf"] > cutoff:
        raise ValueError("UPL input must be frozen before the exposure/review, not after its outcome")
    result["uplBundle"] = bundle
    return result


def artifact_bank(paths):
    result = {}
    for path in paths:
        payload = Path(path).read_bytes()
        artifact = json.loads(payload)
        identifier = artifact["id"]
        if identifier in result:
            raise ValueError("Duplicate artifact ID in feedback bank")
        model = DiffusionStudent.from_artifact(artifact)
        result[identifier] = {"artifact": artifact, "model": model, "hash": hashlib.sha256(payload).hexdigest()}
    return result


def materialize_feedback(document, bank, *, parent=None):
    if document.get("format") != "angelos.feedback.v1" or document.get("outcome") != "click" or document.get("population") != "local-synthetic-adults":
        raise ValueError("Only explicit local synthetic-adult click feedback is supported")
    if type(document.get("asOf")) is not int or document["asOf"] < 0:
        raise ValueError("Feedback requires an integer cutoff")
    if not isinstance(document.get("clicks"), list) or not isinstance(document.get("reviews"), list):
        raise ValueError("Feedback requires click and review rows")
    if not 1 <= len(document["clicks"]) <= 10000 or not 1 <= len(document["reviews"]) <= 10000:
        raise ValueError("Need 1–10000 mature exposures and reviews; no reward labels are invented")
    if parent is not None:
        if document.get("campaignId") != parent["campaignId"]:
            raise ValueError("Feedback campaign does not match candidate parent")
        if any(row.get("encoder") != parent["encoder"] for row in document["clicks"] + document["reviews"]):
            raise ValueError("Feedback encoder does not match candidate parent")
    snapshots = {}
    frozen = {}
    for row in document["clicks"] + document["reviews"]:
        inputs = frozen_upl(row, document["campaignId"])
        descriptor = row["artifact"]
        entry = bank.get(descriptor["id"])
        if (entry is None or descriptor.get("hash") != entry["hash"]
                or any(descriptor.get(name) != entry["artifact"].get(name) for name in ("id", "campaignId", "graphHash", "encoder"))
                or inputs["encoder"] != descriptor.get("encoder")
                or row["campaignId"] != document["campaignId"] or descriptor.get("campaignId") != document["campaignId"]):
            raise ValueError("Exposure artifact identity/hash or campaign/encoder mismatch")
        if "uplBundle" in inputs and inputs["uplBundle"]["context"]["asOf"] > document["asOf"]:
            raise ValueError("UPL bundle is newer than the feedback cutoff")
        binding = {"profileId": row["profileId"], **inputs}
        snapshot_id = row["snapshotId"]
        if snapshot_id in snapshots and snapshots[snapshot_id] != binding:
            raise ValueError("One frozen snapshot cannot have conflicting profiles, vectors or recent intent")
        snapshots[snapshot_id] = binding
        frozen[id(row)] = inputs
    cache = {}

    def reconstruct(row, creative):
        descriptor = row["artifact"]
        entry = bank.get(descriptor["id"])
        if entry is None:
            raise ValueError("An exposure's exact model artifact is missing")
        artifact = entry["artifact"]
        if descriptor["hash"] != entry["hash"] or any(descriptor[key] != artifact[key] for key in ("id", "campaignId", "graphHash", "encoder")):
            raise ValueError("Exposure artifact identity/hash mismatch")
        if row["campaignId"] != document["campaignId"] or artifact["campaignId"] != document["campaignId"] or row["encoder"] != artifact["encoder"]:
            raise ValueError("Feedback campaign/encoder mismatch")
        match = re.fullmatch(r"creative:([a-f0-9]{64}):(\d+)", creative)
        if not match or not 0 <= int(match[2]) < 2 ** 32:
            raise ValueError("Creative lacks the server's artifact/snapshot/seed identity")
        seed = int(match[2])
        serialized = json.dumps([artifact["id"], row["snapshotId"], seed], separators=(",", ":"), ensure_ascii=False)
        if hashlib.sha256(serialized.encode()).hexdigest() != match[1]:
            raise ValueError("Creative identity does not match frozen snapshot/model/seed")
        key = (creative, tuple(row["upl"]))
        if key not in cache:
            image = sample_ddim(entry["model"], row["upl"], seed, artifact["schedule"])
            cache[key] = composite_rgb(image, artifact["brand"]).permute(2, 0, 1).cpu()
        return {"image": cache[key], **copy.deepcopy(frozen[id(row)]), "profileId": row["profileId"], "creativeId": creative,
                "snapshotId": row["snapshotId"]}

    clicks = []
    seen = set()
    for row in document["clicks"]:
        if row["impressionId"] in seen:
            raise ValueError("Duplicate impression label")
        seen.add(row["impressionId"])
        if type(row["label"]) is not int or row["label"] not in (0, 1) or type(row["maturesAt"]) is not int or row["maturesAt"] > document["asOf"]:
            raise ValueError("Only mature binary click labels can train the click predictor")
        clicks.append({**reconstruct(row, row["context"]["creativeId"]), "label": row["label"], "source": "local-ledger"})
    reviews = []
    seen.clear()
    for row in document["reviews"]:
        if row["id"] in seen:
            raise ValueError("Duplicate review")
        seen.add(row["id"])
        if row["source"] not in ("operator", "synthetic") or type(row["qualityRating"]) is not int or not 1 <= row["qualityRating"] <= 5 or type(row["brandPass"]) is not bool:
            raise ValueError("Invalid review rating/source")
        if type(row["receivedAt"]) is not int or row["receivedAt"] > document["asOf"]:
            raise ValueError("Review is newer than the export cutoff")
        reviews.append({**reconstruct(row, row["creativeId"]), "quality": (row["qualityRating"] - 1) / 4, "brand": float(row["brandPass"]), "source": row["source"]})
    return clicks, reviews
