"""Strict synthetic UPL parsing and explicit typed ComfyUI conditioning nodes."""

import json
import math
from string import Formatter

from .nodes import bounded_text, json_bytes, make_audience, reject_constant, unique_fields, validate_audience
from . import upl_v2


WIRE_FORMAT = "angelos.upl-input.v1"
WIRE_FIELDS = {"format", "synthetic", "adult", "factsSchema", "facts", "encoder", "upl"}
MAX_SCALAR = 1000000


def parse_upl(upl_json):
    bounded_text(upl_json, "upl_json", upl_v2.MAX_BYTES)
    try:
        document = json.loads(upl_json, object_pairs_hook=unique_fields, parse_constant=reject_constant)
    except RecursionError as error:
        raise ValueError("UPL JSON is nested too deeply") from error
    json_bytes(document, upl_v2.MAX_BYTES)
    if type(document) is dict and document.get("format") == upl_v2.WIRE_FORMAT:
        return upl_v2.make_audience(document)
    bounded_text(upl_json, "upl_json", 16384)
    json_bytes(document, 16384)
    if type(document) is not dict or set(document) != WIRE_FIELDS:
        raise ValueError("UPL requires exactly format, synthetic, adult, factsSchema, facts, encoder and upl; no input hash")
    if document["format"] != WIRE_FORMAT or document["factsSchema"] != "angelos.facts.v1":
        raise ValueError("Unsupported UPL wire or facts version")
    if document["synthetic"] is not True or document["adult"] is not True:
        raise ValueError("Only explicitly synthetic adult UPLs are authorized")
    encoder = document["encoder"]
    if (type(encoder) is not dict or set(encoder) != {"id", "dimension"}
            or encoder["id"] != "structured-demo-v1" or type(encoder["dimension"]) is not int
            or encoder["dimension"] != 4):
        raise ValueError("Only structured-demo-v1 with dimension 4 is supported")
    vector = document["upl"]
    if type(vector) is not list or len(vector) != 4:
        raise ValueError("UPL must contain exactly four coordinates")
    for value in vector:
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("UPL coordinates must be finite non-boolean numbers in [0,1]")
    audience = make_audience("custom", json_bytes(document["facts"]).decode("utf-8"), True)
    if any(float(actual) != expected for actual, expected in zip(vector, audience["upl"])):
        raise ValueError("UPL does not match the declared facts under structured-demo-v1")
    return audience


def scalar(value, name):
    if type(value) not in (int, float) or abs(value) > MAX_SCALAR or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite non-boolean scalar within +/-{MAX_SCALAR}")
    return float(value)


def integer(value, name):
    if type(value) is not int or abs(value) > MAX_SCALAR:
        raise ValueError(f"{name} must be an integer within +/-{MAX_SCALAR}")
    return value


class AngelosParseUPL:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"upl_json": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ("ANGELOS_AUDIENCE", "STRING")
    RETURN_NAMES = ("audience", "canonical_audience_json")
    FUNCTION = "parse"
    CATEGORY = "Angelos/UPL"

    def parse(self, upl_json):
        audience = parse_upl(upl_json)
        return audience, json_bytes(audience).decode("utf-8")


class AngelosUPLFactString:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audience": ("ANGELOS_AUDIENCE", {"forceInput": True}),
                             "field": (["ageBand", "region", "preferredLanguage", "interests", "goals", "preferredStyles"],)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fact_text",)
    FUNCTION = "extract"
    CATEGORY = "Angelos/UPL"

    def extract(self, audience, field):
        audience = validate_audience(audience)
        if field not in ("ageBand", "region", "preferredLanguage", "interests", "goals", "preferredStyles"):
            raise ValueError("Only supported explicit text/list facts are available")
        if audience["facts"].get(field) is None:
            raise ValueError(f"Explicit fact {field} is absent; no inferred fallback")
        value = audience["facts"][field]
        return (", ".join(value) if type(value) is list else value,)


class AngelosUPLFactFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audience": ("ANGELOS_AUDIENCE", {"forceInput": True}),
                             "field": (["openness", "energy"],)}}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("fact_value",)
    FUNCTION = "extract"
    CATEGORY = "Angelos/UPL"

    def extract(self, audience, field):
        audience = validate_audience(audience)
        if field not in ("openness", "energy"):
            raise ValueError("Only explicit numeric facts openness and energy are supported")
        if audience["facts"].get(field) is None:
            raise ValueError(f"Explicit fact {field} is absent; no inferred fallback")
        return (float(audience["facts"][field]),)


class AngelosUPLVectorFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audience": ("ANGELOS_AUDIENCE", {"forceInput": True}),
                             "index": ("INT", {"default": 0, "min": 0, "max": 1023})}}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("opaque_coordinate",)
    FUNCTION = "extract"
    CATEGORY = "Angelos/UPL"

    def extract(self, audience, index):
        audience = validate_audience(audience)
        if type(index) is not int or not 0 <= index < len(audience["upl"]):
            raise ValueError("Opaque coordinate index must be within the declared vector dimension")
        return (float(audience["upl"][index]),)


class AngelosUPLEmbeddingBlock:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audience": ("ANGELOS_AUDIENCE", {"forceInput": True})}}

    RETURN_TYPES = ("ANGELOS_EMBEDDING",)
    RETURN_NAMES = ("opaque_embedding_bundle",)
    FUNCTION = "extract"
    CATEGORY = "Angelos/UPL"

    def extract(self, audience):
        audience = validate_audience(audience)
        if audience["format"] != upl_v2.AUDIENCE_FORMAT:
            raise ValueError("Opaque embedding blocks require a UPL v2 bundle, not structured v1")
        return (audience,)


class AngelosUPLRecentIntentText:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audience": ("ANGELOS_AUDIENCE", {"forceInput": True})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("recent_intent_text",)
    FUNCTION = "extract"
    CATEGORY = "Angelos/UPL"

    def extract(self, audience):
        audience = validate_audience(audience)
        if audience["format"] != upl_v2.AUDIENCE_FORMAT:
            raise ValueError("Recent intent requires an explicit UPL v2 bundle")
        content = "\n".join(f"{event['kind']}: {event['topic']}" for event in audience["recentIntent"])
        return (bounded_text(content, "recent intent text", 32768, empty=True),)


class AngelosScalarMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "value": ("FLOAT", {"forceInput": True}),
            "input_min": ("FLOAT", {"default": 0.0, "min": -MAX_SCALAR, "max": MAX_SCALAR}),
            "input_max": ("FLOAT", {"default": 1.0, "min": -MAX_SCALAR, "max": MAX_SCALAR}),
            "output_min": ("FLOAT", {"default": 2.0, "min": -MAX_SCALAR, "max": MAX_SCALAR}),
            "output_max": ("FLOAT", {"default": 7.5, "min": -MAX_SCALAR, "max": MAX_SCALAR}),
        }}

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("mapped_float",)
    FUNCTION = "map_value"
    CATEGORY = "Angelos/UPL"

    def map_value(self, value, input_min, input_max, output_min, output_max):
        value = scalar(value, "value")
        input_min, input_max = scalar(input_min, "input_min"), scalar(input_max, "input_max")
        output_min, output_max = scalar(output_min, "output_min"), scalar(output_max, "output_max")
        if input_min >= input_max or not input_min <= value <= input_max or output_min > output_max:
            raise ValueError("Mapping requires increasing input bounds, ordered output bounds and an in-range value")
        mapped = output_min + (value - input_min) / (input_max - input_min) * (output_max - output_min)
        return (scalar(mapped, "mapped_float"),)


class AngelosScalarToInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "value": ("FLOAT", {"forceInput": True}),
            "input_min": ("FLOAT", {"default": 0.0, "min": -MAX_SCALAR, "max": MAX_SCALAR}),
            "input_max": ("FLOAT", {"default": 1.0, "min": -MAX_SCALAR, "max": MAX_SCALAR}),
            "output_min": ("INT", {"default": 4, "min": -MAX_SCALAR, "max": MAX_SCALAR}),
            "output_max": ("INT", {"default": 8, "min": -MAX_SCALAR, "max": MAX_SCALAR}),
            "rounding": (["nearest", "floor", "ceil"],),
        }}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("mapped_integer",)
    FUNCTION = "map_value"
    CATEGORY = "Angelos/UPL"

    def map_value(self, value, input_min, input_max, output_min, output_max, rounding):
        value = scalar(value, "value")
        input_min, input_max = scalar(input_min, "input_min"), scalar(input_max, "input_max")
        output_min, output_max = integer(output_min, "output_min"), integer(output_max, "output_max")
        if input_min >= input_max or not input_min <= value <= input_max or output_min > output_max:
            raise ValueError("Mapping requires increasing input bounds, ordered output bounds and an in-range value")
        if rounding not in ("nearest", "floor", "ceil"):
            raise ValueError("Unknown integer rounding mode")
        mapped = output_min + (value - input_min) / (input_max - input_min) * (output_max - output_min)
        result = math.floor(mapped + 0.5) if rounding == "nearest" else math.floor(mapped) if rounding == "floor" else math.ceil(mapped)
        return (result,)


class AngelosScalarCompare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "value": ("FLOAT", {"forceInput": True}),
            "threshold": ("FLOAT", {"default": 0.5, "min": -MAX_SCALAR, "max": MAX_SCALAR}),
            "operator": ([">=", ">", "<=", "<", "=="],),
        }}

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("condition",)
    FUNCTION = "compare"
    CATEGORY = "Angelos/UPL"

    def compare(self, value, threshold, operator):
        value, threshold = scalar(value, "value"), scalar(threshold, "threshold")
        comparisons = {">=": value >= threshold, ">": value > threshold, "<=": value <= threshold,
                       "<": value < threshold, "==": value == threshold}
        if type(operator) is not str or operator not in comparisons:
            raise ValueError("Unknown comparison operator")
        return (comparisons[operator],)


class AngelosPromptTemplate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "template": ("STRING", {"default": "A fictional Aurora scene in {text} with energy {value}", "multiline": True}),
            "text": ("STRING", {"default": "NA"}),
            "value": ("FLOAT", {"default": 0.5, "min": -MAX_SCALAR, "max": MAX_SCALAR}),
            "integer_value": ("INT", {"default": 1, "min": -MAX_SCALAR, "max": MAX_SCALAR}),
            "enabled": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "render"
    CATEGORY = "Angelos/UPL"

    def render(self, template, text="", value=0.5, integer_value=1, enabled=True):
        bounded_text(template, "template", 4096)
        bounded_text(text, "text", 4096, empty=True)
        value, integer_value = scalar(value, "value"), integer(integer_value, "integer_value")
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        fields = {"text": text, "value": str(value), "integer": str(integer_value), "enabled": str(enabled).lower()}
        for literal, field, specification, conversion in Formatter().parse(template):
            if field is not None and (field not in fields or specification or conversion):
                raise ValueError("Only {text}, {value}, {integer}, {enabled} are allowed; no attribute access or format specs")
        prompt = template.format_map(fields)
        return (bounded_text(prompt, "formatted prompt", 4096),)


class AngelosChoosePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "condition": ("BOOLEAN", {"forceInput": True}),
            "when_true": ("STRING", {"default": "A bold fictional Aurora campaign", "multiline": True}),
            "when_false": ("STRING", {"default": "A quiet fictional Aurora campaign", "multiline": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "choose"
    CATEGORY = "Angelos/UPL"

    def choose(self, condition, when_true, when_false):
        if type(condition) is not bool:
            raise ValueError("condition must be a boolean, not a truthy string or number")
        bounded_text(when_true, "when_true", 4096)
        bounded_text(when_false, "when_false", 4096)
        return (when_true if condition else when_false,)
