"""Bounded text contracts shared by graph validation, adapters and the CLI."""

import math

from .models import model_spec


MAX_INPUT_TOKENS = 4096
MAX_TEXT_UNITS = 8192
MAX_STDIN_BYTES = 65536
ECMASCRIPT_WHITESPACE = "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def bounded_text(value, maximum, name):
    if not isinstance(value, str) or not value.strip(ECMASCRIPT_WHITESPACE):
        raise ValueError(f"{name} must be nonblank text")
    try:
        units = len(value.encode("utf-16-le")) // 2
    except UnicodeEncodeError as error:
        raise ValueError(f"{name} must contain valid Unicode") from error
    if units > maximum:
        raise ValueError(f"{name} exceeds {maximum} UTF-16 code units; truncation is forbidden")
    return value


def validate_text_parameters(model, system, max_tokens, temperature):
    if model_spec(model).kind != "text":
        raise ValueError("Text generation requires an allowlisted text model")
    bounded_text(system, 2048, "system")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, (int, float)) or not 1 <= max_tokens <= 512 or int(max_tokens) != max_tokens:
        raise ValueError("max_tokens must be an integer in [1,512]")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be finite and in [0,2]")


def validate_text_result(result, max_tokens):
    if not isinstance(result, dict) or set(result) != {"text", "usage", "finish_reason"}:
        raise ValueError("Text adapter must return exactly text, usage and finish_reason")
    bounded_text(result["text"], MAX_TEXT_UNITS, "Generated text")
    usage = result["usage"]
    if not isinstance(usage, dict) or set(usage) != {"input_tokens", "output_tokens"}:
        raise ValueError("Text adapter returned invalid usage")
    for name, maximum in (("input_tokens", MAX_INPUT_TOKENS), ("output_tokens", max_tokens)):
        if type(usage[name]) is not int or not 1 <= usage[name] <= maximum:
            raise ValueError(f"Text adapter returned invalid {name}")
    if result["finish_reason"] not in ("stop", "length") or result["finish_reason"] == "length" and usage["output_tokens"] != max_tokens:
        raise ValueError("Text adapter returned invalid finish_reason")
    return result
