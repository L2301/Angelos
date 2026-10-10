"""Strict graph validation, ordering, and source identity."""

import copy
import hashlib
import heapq
import json
import math
import re
from pathlib import Path

from .upl import DIMENSIONS, encoder_metadata


IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
FORBIDDEN_IDENTIFIERS = {"__proto__", "prototype", "constructor", "campaign", "output"}
OPERATIONS = {
    "upl": ({}, set(), "upl"),
    "feature": ({"input": "upl"}, {"index"}, "scalar"),
    "constant": ({}, {"value"}, "scalar"),
    "prompt": ({"condition": "scalar"}, {"low", "high", "threshold"}, "prompt"),
    "generate_text": ({"prompt": ("prompt", "text")}, {"model", "system", "max_tokens", "temperature"}, "text"),
    "generate_image": ({"prompt": ("prompt", "text")}, {"model", "steps", "guidance", "width", "height"}, "image"),
    "generate_video": ({"prompt": ("prompt", "text")}, {"model", "steps", "guidance", "width", "height", "num_frames"}, "video"),
    "frame": ({"video": "video"}, {"index"}, "image"),
    "gradient": ({"condition": "scalar"}, {"low", "high", "angle"}, "image"),
    "disk": (
        {"image": "image", "condition": "scalar"},
        {"color", "center", "radius", "softness"},
        "image",
    ),
    "exposure": ({"image": "image", "condition": "scalar"}, {"amount"}, "image"),
    "vignette": ({"image": "image"}, {"strength"}, "image"),
    "brand": ({"image": "image"}, {"x", "y", "cell", "color", "background"}, "image"),
}


def exact_fields(value, expected, context):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{context} requires exactly {sorted(expected)}")


def finite_number(value, minimum, maximum, context):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number")
    if not minimum <= value <= maximum or not math.isfinite(value):
        raise ValueError(f"{context} must be finite and in [{minimum}, {maximum}]")


def vector(value, dimension, context):
    if not isinstance(value, list) or len(value) != dimension:
        raise ValueError(f"{context} requires {dimension} coordinates")
    for coordinate in value:
        finite_number(coordinate, 0, 1, context)


def validate_parameters(node, dimension=4):
    operation = node["op"]
    parameters = node["params"]
    context = node["id"]
    exact_fields(parameters, OPERATIONS[operation][1], f"{context}.params")
    if operation == "generate_text":
        from .text_contract import validate_text_parameters

        validate_text_parameters(**parameters)
        return
    if operation == "prompt":
        for name in ("low", "high"):
            text = parameters[name]
            if not isinstance(text, str) or not text.strip() or len(text.encode("utf-16-le")) // 2 > 2048 or re.search(r"[{}]", text.replace("{condition}", "")):
                raise ValueError(f"{context}.{name}: expected prompt text with only optional {{condition}} placeholders")
        finite_number(parameters["threshold"], 0, 1, f"{context}.threshold")
        return
    if operation in {"generate_image", "generate_video"}:
        from .models import model_spec

        specification = model_spec(parameters["model"])
        if specification.kind != OPERATIONS[operation][2]:
            raise ValueError(f"{context}.model has the wrong output type")
        for name, minimum, maximum in (("steps", 1, 100), ("width", 256, 512), ("height", 256, 512)):
            finite_number(parameters[name], minimum, maximum, f"{context}.{name}")
            if int(parameters[name]) != parameters[name]:
                raise ValueError(f"{context}.{name} must be an integer")
        if parameters["width"] % 32 or parameters["height"] % 32:
            raise ValueError("Generation dimensions must be multiples of 32")
        finite_number(parameters["guidance"], 0, 20, f"{context}.guidance")
        if operation == "generate_video":
            finite_number(parameters["num_frames"], 1, 121, f"{context}.num_frames")
            if (parameters["num_frames"] - 1) % 8:
                raise ValueError("LTX frame count must be 8n+1")
        return
    if operation == "frame":
        finite_number(parameters["index"], 0, 120, f"{context}.index")
        if int(parameters["index"]) != parameters["index"]:
            raise ValueError("Frame index must be an integer")
    if operation == "feature":
        index = parameters["index"]
        finite_number(index, 0, dimension - 1, f"{context}.index")
        if int(index) != index:
            raise ValueError(f"{context}.index must be an integer")
    for name in ("low", "high", "color", "background"):
        if name in parameters:
            vector(parameters[name], 3, f"{context}.{name}")
    if "center" in parameters:
        vector(parameters["center"], 2, f"{context}.center")
    bounds = {
        "value": (0, 1),
        "angle": (-1, 1),
        "radius": (0.02, 1),
        "softness": (0.001, 1),
        "amount": (-1, 1),
        "strength": (0, 1),
        "x": (0, 1),
        "y": (0, 1),
        "cell": (0.005, 0.05),
    }
    for name, (minimum, maximum) in bounds.items():
        if name in parameters:
            finite_number(parameters[name], minimum, maximum, f"{context}.{name}")
    if operation == "brand":
        if parameters["x"] + 7 * parameters["cell"] > 1:
            raise ValueError("Brand plate extends past the right edge")
        if parameters["y"] + 9 * parameters["cell"] > 1:
            raise ValueError("Brand plate extends past the bottom edge")


def validate_graph(graph):
    exact_fields(graph, {"version", "id", "encoder", "nodes", "output"}, "graph")
    if graph["version"] not in ("angelos.graph.v1", "angelos.graph.v2"):
        raise ValueError("Unsupported graph version")
    if not valid_identifier(graph["id"]):
        raise ValueError("Invalid campaign ID")
    if graph["version"] == "angelos.graph.v1":
        exact_fields(graph["encoder"], {"id", "dimension"}, "encoder")
        if graph["encoder"]["id"] != "structured-demo-v1":
            raise ValueError("Unsupported UPL encoder")
        dimension = graph["encoder"]["dimension"]
        if isinstance(dimension, bool) or dimension != 4:
            raise ValueError("UPL dimension must be four")
    else:
        encoder = graph["encoder"]
        if (not isinstance(encoder, dict) or type(encoder.get("dimension")) is not int
                or encoder["dimension"] not in DIMENSIONS or encoder != encoder_metadata(encoder["dimension"])):
            raise ValueError("Unsupported pinned semantic encoder contract")
        dimension = encoder["dimension"]
    nodes = graph["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 64:
        raise ValueError("Graph requires between one and 64 nodes")
    lookup = {}
    for node in nodes:
        exact_fields(node, {"id", "op", "inputs", "params"}, "node")
        identifier = node["id"]
        if not valid_identifier(identifier):
            raise ValueError("Invalid node ID")
        if identifier in lookup:
            raise ValueError(f"Duplicate node ID: {identifier}")
        operation = node["op"]
        if not isinstance(operation, str) or operation not in OPERATIONS:
            raise ValueError(f"Unsupported operation: {operation}")
        exact_fields(node["inputs"], OPERATIONS[operation][0], f"{identifier}.inputs")
        validate_parameters(node, dimension)
        lookup[identifier] = node
    if sum(node["op"] == "upl" for node in nodes) != 1:
        raise ValueError("Exactly one UPL source is required")
    output = graph["output"]
    if not isinstance(output, str) or output not in lookup:
        raise ValueError("Unknown graph output")
    if OPERATIONS[lookup[output]["op"]][2] != "image":
        raise ValueError("Graph output must be an image")
    brands = [node for node in nodes if node["op"] == "brand"]
    if len(brands) > 1 or brands and brands[0]["id"] != output:
        raise ValueError("At most one brand node is allowed, and it must be the output")
    dependencies = {}
    consumers = {identifier: [] for identifier in lookup}
    for identifier, node in lookup.items():
        dependencies[identifier] = set()
        for port, source in node["inputs"].items():
            if not isinstance(source, str) or source not in lookup:
                raise ValueError(f"Unknown source on {identifier}.{port}")
            expected = OPERATIONS[node["op"]][0][port]
            actual = OPERATIONS[lookup[source]["op"]][2]
            if actual not in (expected if isinstance(expected, tuple) else (expected,)):
                raise ValueError(f"Type mismatch on {identifier}.{port}: {actual} != {expected}")
            if node["op"] == "frame" and node["params"]["index"] >= lookup[source]["params"]["num_frames"]:
                raise ValueError("Frame index exceeds video frame count")
            dependencies[identifier].add(source)
        for source in dependencies[identifier]:
            consumers[source].append(identifier)
    pending = {identifier: len(sources) for identifier, sources in dependencies.items()}
    ready = [identifier for identifier, count in pending.items() if count == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        identifier = heapq.heappop(ready)
        ordered.append(lookup[identifier])
        for consumer in consumers[identifier]:
            pending[consumer] -= 1
            if pending[consumer] == 0:
                heapq.heappush(ready, consumer)
    if len(ordered) != len(nodes):
        raise ValueError("Graph contains a cycle")
    reachable = set()
    stack = [output]
    while stack:
        identifier = stack.pop()
        if identifier not in reachable:
            reachable.add(identifier)
            stack.extend(dependencies[identifier])
    if len(reachable) != len(nodes):
        raise ValueError("All nodes must be reachable from the output")
    return copy.deepcopy(ordered)


def valid_identifier(value):
    return isinstance(value, str) and len(value) <= 64 and value not in FORBIDDEN_IDENTIFIERS and IDENTIFIER.fullmatch(value) is not None


def javascript_number(value):
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Canonical JSON cannot contain non-finite numbers")
    if numeric == 0:
        return "0"
    representation = repr(numeric).lower()
    if "e" not in representation:
        return representation.removesuffix(".0")
    mantissa, exponent_text = representation.split("e")
    exponent = int(exponent_text)
    if -6 <= exponent < 21:
        negative = mantissa.startswith("-")
        unsigned = mantissa.lstrip("-")
        leading, _, trailing = unsigned.partition(".")
        digits = leading + trailing
        decimal_index = len(leading) + exponent
        if decimal_index <= 0:
            fixed = "0." + "0" * (-decimal_index) + digits
        elif decimal_index >= len(digits):
            fixed = digits + "0" * (decimal_index - len(digits))
        else:
            fixed = digits[:decimal_index] + "." + digits[decimal_index:]
        return ("-" if negative else "") + fixed
    sign = "+" if exponent >= 0 else "-"
    return mantissa.removesuffix(".0") + "e" + sign + str(abs(exponent))


def canonical_json(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, int)):
        return javascript_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(canonical_json(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(
            canonical_json(key) + ":" + canonical_json(value[key]) for key in sorted(value)
        ) + "}"
    raise ValueError("Unsupported canonical JSON value")


def graph_hash(graph):
    validate_graph(graph)
    normalized = copy.deepcopy(graph)
    normalized["nodes"] = sorted(normalized["nodes"], key=lambda node: node["id"])
    return hashlib.sha256(canonical_json(normalized).encode("utf-8")).hexdigest()


def verify_graph_hash(graph, expected, canonical_path=None):
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("--graph-hash must be a 64-character hexadecimal compiler hash")
    if canonical_path is None:
        computed = graph_hash(graph)
    else:
        canonical_graph = load_graph(canonical_path)
        normalized = copy.deepcopy(graph)
        normalized["nodes"] = sorted(normalized["nodes"], key=lambda node: node["id"])
        if canonical_graph != normalized:
            raise ValueError("Compiler canonical graph does not match the training graph")
        canonical = Path(canonical_path).read_text(encoding="utf-8").strip()
        computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if computed != expected.lower():
        raise ValueError(
            "Compiler graph hash does not match the graph; regenerate the sidecar or "
            "supply --graph-canonical for an authoritative JS numeric serialization"
        )
    return computed


def load_graph(path):
    def reject_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    graph = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )
    validate_graph(graph)
    return graph
