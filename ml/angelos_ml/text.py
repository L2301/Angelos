"""Bounded JSON-in/JSON-out local text generation; never downloads weights."""

import argparse
import contextlib
import json
import sys

from .text_contract import MAX_STDIN_BYTES, MAX_TEXT_UNITS, bounded_text, validate_text_parameters


def read_request(stream):
    payload = stream.read(MAX_STDIN_BYTES + 1)
    if len(payload) > MAX_STDIN_BYTES:
        raise ValueError("stdin exceeds 65536 bytes")

    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")

    request = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_fields, parse_constant=reject_constant)
    if not isinstance(request, dict) or set(request) != {"model", "prompt", "system", "max_tokens", "temperature", "seed"}:
        raise ValueError("stdin requires exactly model, prompt, system, max_tokens, temperature and seed")
    validate_text_parameters(request["model"], request["system"], request["max_tokens"], request["temperature"])
    bounded_text(request["prompt"], MAX_TEXT_UNITS, "prompt")
    seed = request["seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, float)) or not 0 <= seed <= 2 ** 53 - 1 or int(seed) != seed:
        raise ValueError("seed must be a nonnegative JavaScript-safe integer")
    request["seed"] = int(seed)
    request["max_tokens"] = int(request["max_tokens"])
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdin", action="store_true", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--cache-dir")
    arguments = parser.parse_args(argv)
    try:
        request = read_request(sys.stdin.buffer)
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from .providers import PipelineRegistry, node_seed

            torch.set_num_threads(2)
            registry = PipelineRegistry(device=arguments.device, dtype=getattr(torch, arguments.dtype), cache_dir=arguments.cache_dir)
            parameters = {name: request[name] for name in ("system", "max_tokens", "temperature")}
            derived_seed = node_seed(request["seed"], "text")
            result = registry.generate_text(request["model"], request["prompt"], parameters, derived_seed)
            response = {
                **result, "model": registry.metadata(request["model"]), "seed": request["seed"],
                "node_seed": str(derived_seed), "parameters": parameters,
            }
        encoded = json.dumps(response, allow_nan=False)
    except Exception as error:
        parser.exit(2, f"Text generation failed ({type(error).__name__}): {error}\n")
    print(encoded)


if __name__ == "__main__":
    main()
