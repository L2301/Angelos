"""Synthetic-adult UPL v2: pinned, cache-only semantic inference, never downloads."""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
ENCODER_ID = "qwen3-embedding-0.6b-v1"
DIMENSIONS = (128, 256, 512, 1024)
TEMPLATE_VERSION = "angelos.upl-text.v1"
INGREDIENTS_FORMAT = "angelos.upl-ingredients.v2"
OUTPUT_FORMAT = "angelos.upl-input.v2"
FACTS_SCHEMA = "angelos.facts.v2"
EVENTS_SCHEMA = "angelos.events.v1"
MAX_STDIN_BYTES = 262144
MAX_INPUT_TOKENS = 4096
MAX_SAFE_INTEGER = 2 ** 53 - 1
FACT_FIELDS = ("ageBand", "region", "openness", "energy", "interests", "goals", "preferredStyles", "preferredLanguage")
JS_TRIM_CHARACTERS = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"

def _object(value, fields, required=None):
    if type(value) is not dict or not set(value) <= set(fields) or not set(fields if required is None else required) <= set(value):
        raise ValueError("Object contains missing or unsupported fields")


def _integer(value, name, minimum=0):
    if type(value) not in (int, float) or not minimum <= value <= MAX_SAFE_INTEGER or int(value) != value:
        raise ValueError(f"{name} must be a JavaScript-safe integer >= {minimum}")
    return int(value)


def _text(value, name, maximum=80):
    if (not isinstance(value, str) or not value or value != value.strip(JS_TRIM_CHARACTERS)
            or any(ord(character) < 32 or ord(character) == 127 or 0xD800 <= ord(character) <= 0xDFFF for character in value)):
        raise ValueError(f"{name} must be trimmed nonempty text without controls, DEL or unpaired surrogates")
    if len(value.encode("utf-16-le")) // 2 > maximum:
        raise ValueError(f"{name} exceeds {maximum} UTF-16 code units")


def _identifier(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", value) is None:
        raise ValueError(f"Invalid {name}")


def validate_facts(facts):
    _object(facts, FACT_FIELDS, ("ageBand",))
    if facts["ageBand"] not in ("18-24", "25-34", "35-44", "45-54", "55-64", "65+"):
        raise ValueError("A declared adult ageBand is required")
    for name, value in facts.items():
        if value is None:
            continue
        if name == "region" and value not in ("NA", "EU", "APAC", "OTHER"):
            raise ValueError("Invalid coarse region")
        if name in ("openness", "energy"):
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite number in [0,1] or null")
        if name in ("interests", "goals", "preferredStyles"):
            if type(value) is not list or len(value) > 16:
                raise ValueError(f"{name} requires at most 16 strings")
            for entry in value:
                _text(entry, name)
            if len(set(value)) != len(value):
                raise ValueError(f"{name} contains duplicate strings")
        if name == "preferredLanguage":
            _text(value, name)
    return facts


def _times(record, as_of):
    occurred = _integer(record["occurredAt"], "occurredAt")
    received = _integer(record["receivedAt"], "receivedAt")
    if not occurred <= received <= as_of:
        raise ValueError("Times must satisfy occurredAt <= receivedAt <= asOf")


def validate_ingredients(ingredients):
    _object(ingredients, ("format", "synthetic", "adult", "factsSchema", "facts", "provenance", "eventsSchema",
                          "events", "factRevision", "eventRevision", "consentVersion", "campaignId", "asOf"))
    if ingredients["format"] != INGREDIENTS_FORMAT or ingredients["factsSchema"] != FACTS_SCHEMA or ingredients["eventsSchema"] != EVENTS_SCHEMA:
        raise ValueError("Unsupported ingredients/facts/events schema")
    if ingredients["synthetic"] is not True or ingredients["adult"] is not True:
        raise ValueError("Only explicitly synthetic adult ingredients are permitted")
    facts = validate_facts(ingredients["facts"])
    _identifier(ingredients["campaignId"], "campaignId")
    as_of = _integer(ingredients["asOf"], "asOf")
    for name in ("factRevision", "eventRevision", "consentVersion"):
        _integer(ingredients[name], name, 0 if name == "eventRevision" else 1)
    provenance = ingredients["provenance"]
    if type(provenance) is not list or len(provenance) != len(facts):
        raise ValueError("Exactly one latest provenance entry per present fact is required")
    fields = set()
    for record in provenance:
        _object(record, ("field", "source", "occurredAt", "receivedAt"))
        if not isinstance(record["field"], str) or record["field"] not in facts or record["field"] in fields:
            raise ValueError("Provenance must uniquely identify present facts")
        fields.add(record["field"])
        _identifier(record["source"], "provenance source")
        _times(record, as_of)
    events = ingredients["events"]
    if type(events) is not list or len(events) > 64:
        raise ValueError("At most 64 scoped events are permitted")
    identifiers = set()
    for record in events:
        _object(record, ("id", "kind", "topic", "occurredAt", "receivedAt", "expiresAt", "source", "campaignId"))
        _identifier(record["id"], "event id")
        _identifier(record["source"], "event source")
        if record["id"] in identifiers:
            raise ValueError("Duplicate event id")
        identifiers.add(record["id"])
        if record["kind"] not in ("view", "click", "save", "dismiss"):
            raise ValueError("Unsupported event kind")
        _text(record["topic"], "event topic")
        _times(record, as_of)
        if _integer(record["expiresAt"], "expiresAt") <= as_of or record["campaignId"] != ingredients["campaignId"]:
            raise ValueError("Only unexpired events in the same campaign are permitted")
    return ingredients


def canonical_json(value):
    """Python canonical serialization; not a promise of ECMAScript numeric bytes."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def strict_json(payload):
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")

    return json.loads(payload, object_pairs_hook=unique_fields, parse_constant=reject_constant)


def prepare_request(request):
    _object(request, ("ingredients", "dimension", "ingredientsJson"), ("ingredients", "dimension"))
    dimension = _integer(request["dimension"], "dimension", 1)
    if dimension not in DIMENSIONS:
        raise ValueError("dimension must be 128, 256, 512 or 1024")
    ingredients = validate_ingredients(request["ingredients"])
    serialized = request.get("ingredientsJson", canonical_json(ingredients))
    if not isinstance(serialized, str) or len(serialized.encode("utf-8")) > MAX_STDIN_BYTES:
        raise ValueError("ingredientsJson must be a bounded UTF-8 JSON string")
    verified = validate_ingredients(strict_json(serialized))
    if verified != ingredients:
        raise ValueError("ingredientsJson disagrees with ingredients")
    return {"ingredients": copy.deepcopy(ingredients), "dimension": dimension, "ingredientsJson": serialized}


def read_request(stream):
    payload = stream.read(MAX_STDIN_BYTES + 1)
    if len(payload) > MAX_STDIN_BYTES:
        raise ValueError(f"stdin exceeds {MAX_STDIN_BYTES} bytes")
    return prepare_request(strict_json(payload.decode("utf-8")))


def semantic_text(facts):
    """Document-style stable declared facts only; no summaries or retrieval query."""
    validate_facts(facts)
    rendered = dict(facts)
    for name in ("openness", "energy"):
        if rendered.get(name) is not None:
            rendered[name] = float(rendered[name]) or 0.0
    return TEMPLATE_VERSION + "\nDeclared profile facts:\n" + canonical_json(rendered)


def encoder_metadata(dimension):
    if _integer(dimension, "dimension", 1) not in DIMENSIONS:
        raise ValueError("Unsupported semantic dimension")
    return {"id": ENCODER_ID, "dimension": int(dimension), "modelId": MODEL_ID, "revision": REVISION,
            "pooling": "last-token", "normalization": "l2", "templateVersion": TEMPLATE_VERSION}


def last_token_pool(hidden, attention_mask):
    import torch

    if (not isinstance(hidden, torch.Tensor) or not hidden.is_floating_point() or hidden.ndim != 3
            or min(hidden.shape) < 1 or not torch.isfinite(hidden).all()
            or not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != hidden.shape[:2]
            or not ((attention_mask == 0) | (attention_mask == 1)).all()):
        raise ValueError("Invalid hidden states or attention mask")
    mask = attention_mask.to(device=hidden.device, dtype=torch.bool)
    if not mask.any(dim=1).all():
        raise ValueError("Cannot pool an empty token sequence")
    positions = torch.arange(hidden.shape[1], device=hidden.device).expand(mask.shape)
    last = positions.masked_fill(~mask, -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class LocalSemanticEncoder:
    """Frozen Qwen base-transformer inference with last-token MRL embeddings."""

    def __init__(self, model, tokenizer, device):
        if model.config.model_type != "qwen3" or model.config.hidden_size != 1024:
            raise ValueError("Expected the 1024-wide Qwen3 embedding architecture")
        self.model = model.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = device

    def encode(self, text, dimension):
        import torch

        encoder_metadata(dimension)
        with torch.inference_mode():
            encoded = self.tokenizer(text, padding=True, truncation=False, add_special_tokens=True, return_tensors="pt")
            input_ids, mask = encoded["input_ids"], encoded["attention_mask"]
            if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not 1 <= input_ids.shape[1] <= MAX_INPUT_TOKENS or mask.shape != input_ids.shape:
                raise ValueError("Semantic text exceeds 4096 tokens or has invalid shape; truncation is forbidden")
            output = self.model(input_ids=input_ids.to(self.device), attention_mask=mask.to(self.device),
                                use_cache=False, return_dict=True)
            hidden = output.last_hidden_state
            if hidden.shape != (*input_ids.shape, 1024):
                raise ValueError("Embedding checkpoint returned an unexpected hidden-state shape")
            vector = last_token_pool(hidden, mask)[0, :int(dimension)].float()
            norm = torch.linalg.vector_norm(vector)
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError("Embedding norm is zero or nonfinite; no substitute vector is allowed")
            return (vector / norm).cpu().tolist()


def load_local_encoder(device="cpu", cache_dir=None):
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("Choose an explicit cpu, cuda or mps device")
    from .models import default_cache_dir

    snapshot = Path(cache_dir or default_cache_dir()) / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots" / REVISION
    required = ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors")
    if any(not (snapshot / name).is_file() for name in required):
        raise RuntimeError(
            f"UPL checkpoint is missing or incomplete at {snapshot}. Stage the official {MODEL_ID}@{REVISION} "
            "snapshot through a separate, explicitly authorized provisioning step, then set --cache-dir to its Hub cache root. "
            "This runtime never downloads, uses another revision, or substitutes an embedding."
        )
    configuration = strict_json((snapshot / "config.json").read_text())
    tokenizer_configuration = strict_json((snapshot / "tokenizer_config.json").read_text())
    if (configuration.get("model_type") != "qwen3" or configuration.get("hidden_size") != 1024
            or configuration.get("num_hidden_layers") != 28 or "auto_map" in configuration or "auto_map" in tokenizer_configuration):
        raise ValueError("Pinned snapshot architecture is invalid or requires custom code")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no automatic CPU fallback")
    if device == "mps" and (not torch.backends.mps.is_available() or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0"):
        raise RuntimeError("MPS is unavailable or CPU fallback is enabled; no fallback is permitted")
    options = {"local_files_only": True, "trust_remote_code": False, "revision": REVISION}
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), **options, padding_side="left")
    model = AutoModel.from_pretrained(str(snapshot), **options, use_safetensors=True, dtype=torch.float32).to(device)
    return LocalSemanticEncoder(model, tokenizer, device)


def encode_request(request, *, device="cpu", cache_dir=None):
    prepared = prepare_request(request)
    ingredients, dimension = prepared["ingredients"], prepared["dimension"]
    encoder = load_local_encoder(device=device, cache_dir=cache_dir)
    vector = encoder.encode(semantic_text(ingredients["facts"]), dimension)
    if (type(vector) is not list or len(vector) != dimension
            or any(type(value) not in (int, float) or not -1 <= value <= 1 for value in vector)
            or not math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, rel_tol=0, abs_tol=1e-4)):
        raise ValueError("Encoder must return signed finite L2-normalized floats at the requested width")
    context = {name: ingredients[name] for name in ("campaignId", "factRevision", "eventRevision", "consentVersion", "asOf")}
    context["ingredientsHash"] = hashlib.sha256(prepared["ingredientsJson"].encode("utf-8")).hexdigest()
    return {"format": OUTPUT_FORMAT, "synthetic": True, "adult": True, "factsSchema": FACTS_SCHEMA,
            "facts": ingredients["facts"], "encoder": encoder_metadata(dimension), "upl": vector, "context": context,
            "recentIntent": [{name: record[name] for name in ("kind", "topic", "occurredAt", "expiresAt")} for record in ingredients["events"]]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdin", action="store_true", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--cache-dir")
    arguments = parser.parse_args(argv)
    try:
        request = read_request(sys.stdin.buffer)
        with contextlib.redirect_stdout(sys.stderr):
            result = encode_request(request, device=arguments.device, cache_dir=arguments.cache_dir)
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False)
    except Exception as error:
        parser.exit(2, f"UPL encoding failed ({type(error).__name__}): {error}\n")
    print(encoded)


if __name__ == "__main__":
    main()
