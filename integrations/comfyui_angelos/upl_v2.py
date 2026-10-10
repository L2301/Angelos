"""Standalone, standard-library validation for opaque embedding bundles."""

import hashlib
import json
import math
import re


QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
WIRE_FORMAT = "angelos.upl-input.v2"
AUDIENCE_FORMAT = "angelos.comfyui-audience.v2"
MAX_BYTES = 65536
MAX_SAFE_INTEGER = 9007199254740991
MAX_RECENT_INTENT = 64
AGE_BANDS = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
FACT_FIELDS = {"ageBand", "region", "openness", "energy", "interests", "goals", "preferredStyles", "preferredLanguage"}
WIRE_FIELDS = {"format", "synthetic", "adult", "factsSchema", "facts", "encoder", "upl", "context", "recentIntent"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def text(value, name, maximum):
    if (type(value) is not str or not value.strip() or value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)):
        raise ValueError(f"{name} must be nonblank text without control characters")
    if len(value.encode("utf-16-le")) // 2 > maximum:
        raise ValueError(f"{name} exceeds {maximum} UTF-16 code units")
    return value


def keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"Invalid {name} fields")


def integer(value, name):
    if type(value) is not int or not 0 <= value <= MAX_SAFE_INTEGER:
        raise ValueError(f"{name} must be a nonnegative safe integer")
    return value


def make_audience(document):
    keys(document, WIRE_FIELDS, "UPL v2 wire")
    if document["format"] != WIRE_FORMAT or document["factsSchema"] != "angelos.facts.v2":
        raise ValueError("Unsupported UPL v2 wire/facts version")
    if document["synthetic"] is not True or document["adult"] is not True:
        raise ValueError("Only explicitly synthetic adult UPLs are authorized")
    facts = document["facts"]
    if type(facts) is not dict or "ageBand" not in facts or not set(facts).issubset(FACT_FIELDS):
        raise ValueError("UPL v2 requires ageBand and only supported optional facts")
    if type(facts["ageBand"]) is not str or facts["ageBand"] not in AGE_BANDS:
        raise ValueError("ageBand must be an allowed adult band")
    if facts.get("region") is not None and facts["region"] not in ("NA", "EU", "APAC", "OTHER"):
        raise ValueError("Unsupported region")
    for field in ("openness", "energy"):
        if facts.get(field) is not None and (type(facts[field]) not in (int, float) or not 0 <= facts[field] <= 1):
            raise ValueError(f"{field} must be a finite non-boolean number in [0,1]")
    for field in ("interests", "goals", "preferredStyles"):
        if facts.get(field) is not None:
            values = facts[field]
            if type(values) is not list or len(values) > 16:
                raise ValueError(f"{field} must have at most 16 text entries")
            for value in values:
                text(value, field, 80)
            if len(set(values)) != len(values):
                raise ValueError(f"{field} contains duplicate entries")
    if facts.get("preferredLanguage") is not None:
        text(facts["preferredLanguage"], "preferredLanguage", 80)
    encoder = document["encoder"]
    keys(encoder, {"id", "dimension", "modelId", "revision", "pooling", "normalization", "templateVersion"}, "encoder")
    expected = {"id": "qwen3-embedding-0.6b-v1", "modelId": "Qwen/Qwen3-Embedding-0.6B",
                "revision": QWEN_REVISION, "pooling": "last-token", "normalization": "l2",
                "templateVersion": "angelos.upl-text.v1"}
    if any(encoder[field] != value for field, value in expected.items()):
        raise ValueError("Unsupported or unconfigured pinned embedding encoder")
    dimension = encoder["dimension"]
    if type(dimension) is not int or dimension not in (128, 256, 512, 1024):
        raise ValueError("Embedding dimension must be 128, 256, 512 or 1024")
    vector = document["upl"]
    if type(vector) is not list or len(vector) != dimension:
        raise ValueError("Embedding length does not match declared dimension")
    if any(type(value) not in (int, float) or not -1 <= value <= 1 for value in vector):
        raise ValueError("Embedding coordinates must be finite signed non-boolean numbers in [-1,1]")
    if not math.isclose(math.sqrt(math.fsum(value * value for value in vector)), 1.0, rel_tol=0, abs_tol=1e-4):
        raise ValueError("Embedding must be L2 normalized within absolute tolerance 1e-4; no automatic normalization")
    context = document["context"]
    keys(context, {"campaignId", "factRevision", "eventRevision", "consentVersion", "asOf", "ingredientsHash"}, "context")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", text(context["campaignId"], "campaignId", 128)) is None:
        raise ValueError("Invalid context campaign identifier")
    for field in ("factRevision", "eventRevision", "consentVersion", "asOf"):
        integer(context[field], field)
    if context["consentVersion"] < 1:
        raise ValueError("consentVersion must be positive")
    if re.fullmatch(r"[0-9a-f]{64}", text(context["ingredientsHash"], "ingredientsHash", 64)) is None:
        raise ValueError("Invalid ingredients SHA-256")
    recent = document["recentIntent"]
    if type(recent) is not list or len(recent) > MAX_RECENT_INTENT:
        raise ValueError(f"recentIntent must contain at most {MAX_RECENT_INTENT} events")
    for event in recent:
        keys(event, {"kind", "topic", "occurredAt", "expiresAt"}, "recent intent")
        if event["kind"] not in ("view", "click", "save", "dismiss"):
            raise ValueError("Unsupported recent intent kind")
        text(event["topic"], "topic", 80)
        occurred = integer(event["occurredAt"], "occurredAt")
        expires = integer(event["expiresAt"], "expiresAt")
        if not occurred <= context["asOf"] < expires:
            raise ValueError("Recent intent must be active at context.asOf, not future or expired")
    payload = canonical(document)
    if len(payload) > MAX_BYTES:
        raise ValueError("UPL v2 bundle exceeds 65536 bytes")
    audience = json.loads(payload)
    audience["format"] = AUDIENCE_FORMAT
    audience["sha256"] = hashlib.sha256(canonical(audience)).hexdigest()
    return audience


def validate_audience(audience):
    keys(audience, WIRE_FIELDS | {"sha256"}, "UPL v2 audience")
    if audience["format"] != AUDIENCE_FORMAT:
        raise ValueError("Unsupported audience version")
    document = {key: value for key, value in audience.items() if key != "sha256"}
    document["format"] = WIRE_FORMAT
    expected = make_audience(document)
    if canonical(audience) != canonical(expected):
        raise ValueError("Full UPL v2 bundle hash mismatch")
    return expected
