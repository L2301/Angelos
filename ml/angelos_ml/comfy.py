"""Validate and copy a local ComfyUI teacher pack without executing its workflow.

Public API: import_teacher_pack(pack_dir, output_dir, *, allow_test_fixture=False)
returns the import.json receipt. CLI: python -m angelos_ml.comfy import PACKDIR
--output NEWDIR [--allow-test-fixture]. Output's parent must already exist.
This importer requires macOS/Linux POSIX descriptor APIs, unlike the exporter
plugin, which can also be copied into ComfyUI on Windows. Unsupported platforms
fail explicitly. Supply physical paths (e.g. resolve existing parents first);
symlink aliases such as macOS /tmp or /var are deliberately rejected.

Exportmanifest v1 and v2 are accepted. Paths must have no symlink components or
parent traversals; manifests reference flat filenames. Unknown pack files,
hardlinks, nonregular files, incomplete packs and unverified hash bindings fail.
Limits: 2 MiB per JSON, 50,000 JSON values, depth 32, eight RGB8 PNGs, 4096 per
side, 4096**2 total pixels and 64 MiB total PNG bytes. Pillow is the existing
optional generative dependency; no torch, ComfyUI runtime or network is needed.

All input bytes are validated before creating output. Existing destinations are
never overwritten. Successful files are mode 0444 and the directory 0555, a
write-once application convention, not protection against owner chmod/tampering.
The receipt is written last. PNG bytes and pixels are copied without resizing.
Hashes establish consistency, not authenticity, upstream native generation,
hardware execution, checkpoint identity, quality, brand fidelity or CTR uplift.
Workflow node classes and checkpoint references are inert provenance only.
Export v1 requires synthetic adult facts with structured-demo-v1 four-dimensional
UPL. Export v2 retains the pinned Qwen embedding and full context/intent bundle;
validation does not verify encoder execution or recompute the ingredients hash.
Review profileId groups audience configurations, not people or ledger identities;
fact changes produce a new hash and cannot prove person-disjoint data splits.
"""

import argparse
from datetime import datetime, timedelta
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import zlib
from functools import lru_cache


PACK_FORMAT = "angelos.comfyui-teacher-pack.v1"
PACK_FORMAT_V2 = "angelos.comfyui-teacher-pack.v2"
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_PNG_BYTES = 64 * 1024 * 1024
MAX_PIXELS = 4096 ** 2
AGE_BANDS = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
REGIONS = ("NA", "EU", "APAC", "OTHER")
PRESETS = {
    "quiet": {"ageBand": "35-44", "region": "NA", "openness": 0.25, "energy": 0.2},
    "balanced": {"ageBand": "35-44", "region": "NA", "openness": 0.5, "energy": 0.5},
    "bold": {"ageBand": "35-44", "region": "NA", "openness": 0.85, "energy": 0.9},
}
CHECKPOINT_FIELDS = {
    "CheckpointLoaderSimple": ("ckpt_name",), "CheckpointLoader": ("ckpt_name", "config_name"),
    "UNETLoader": ("unet_name",), "VAELoader": ("vae_name",), "CLIPLoader": ("clip_name",),
    "DualCLIPLoader": ("clip_name1", "clip_name2"),
    "TripleCLIPLoader": ("clip_name1", "clip_name2", "clip_name3"),
    "LoraLoader": ("lora_name",), "LoraLoaderModelOnly": ("lora_name",),
}
VERIFICATION_FLAGS = ("modelHashesVerified", "workflowExecutionVerified", "upstreamNativeGenerationVerified",
                      "hardwareExecutionVerified", "allRandomnessCaptured")


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def _text(value, name, maximum, empty=False):
    if type(value) is not str or "\0" in value or (not empty and not value.strip()):
        raise ValueError(f"Invalid {name} text")
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} exceeds its text bound")
    return value


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _json(payload):
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError("JSON exceeds byte bound")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_fields, parse_constant=_reject_constant)
        budget = [50000]

        def check(item, depth):
            budget[0] -= 1
            if depth > 32 or budget[0] < 0:
                raise ValueError("JSON exceeds depth or item bound")
            if type(item) is dict:
                for key, child in item.items():
                    _text(key, "JSON key", 256)
                    check(child, depth + 1)
            elif type(item) is list:
                for child in item:
                    check(child, depth + 1)
            elif type(item) is str:
                _text(item, "JSON string", 65536, empty=True)
            elif type(item) is float and not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
            elif type(item) is int and item.bit_length() > 256:
                raise ValueError("JSON integer exceeds 256 bits")

        check(value, 0)
        return value
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid bounded UTF-8 JSON") from error


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"Invalid {name} fields")


def _hash(value, name):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {name} SHA-256")
    return value


def _bound_integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name} integer bound")
    return value


def _path(value):
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Parent path traversal is forbidden")
    return Path(os.path.abspath(path))


def _require_posix():
    descriptor_operations = (os.open, os.mkdir, os.stat, os.unlink, os.rmdir)
    if (os.name != "posix" or any(type(getattr(os, flag, None)) is not int for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK"))
            or not callable(getattr(os, "fchmod", None))
            or any(operation not in getattr(os, "supports_dir_fd", ()) for operation in descriptor_operations)
            or os.listdir not in getattr(os, "supports_fd", ())
            or os.stat not in getattr(os, "supports_follow_symlinks", ())):
        raise ValueError("Safe ComfyUI import requires macOS/Linux POSIX O_DIRECTORY/O_NOFOLLOW and dir_fd support; use physical paths")


def _open_directory(path):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_file(directory, name, maximum):
    if type(name) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", name) is None:
        raise ValueError("Pack filename must be flat, without traversal")
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        facts = os.fstat(descriptor)
        if not stat.S_ISREG(facts.st_mode) or facts.st_nlink != 1:
            raise ValueError("Pack entries must be regular files without links")
        if not 0 < facts.st_size <= maximum:
            raise ValueError("Pack file exceeds byte bound or is empty")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(maximum + 1)
        if len(payload) > maximum or not payload:
            raise ValueError("Pack file exceeds byte bound or is empty")
        return payload
    finally:
        os.close(descriptor)


@lru_cache(maxsize=1)
def _v2_contract():
    path = Path(__file__).resolve().parents[2] / "integrations" / "comfyui_angelos" / "upl_v2.py"
    specification = importlib.util.spec_from_file_location("angelos_comfy_v2_contract", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _audience(audience, version_two=False):
    if version_two:
        _v2_contract().validate_audience(audience)
        return
    _keys(audience, ("format", "synthetic", "adult", "preset", "facts", "factsSchema", "encoder", "upl", "sha256"), "audience")
    if audience["encoder"] != {"id": "structured-demo-v1", "dimension": 4}:
        raise ValueError("Export v1 requires structured-demo-v1 audience; opaque encoders need a new pack version")
    preset = audience["preset"]
    if type(preset) is not str or preset not in (*PRESETS, "custom"):
        raise ValueError("Unknown audience preset")
    facts = audience["facts"]
    _keys(facts, ("ageBand", "region", "openness", "energy"), "adult audience facts")
    if facts["ageBand"] not in AGE_BANDS or facts["region"] not in REGIONS:
        raise ValueError("Only supported synthetic adult audience facts are accepted")
    normalized = dict(facts)
    for field in ("openness", "energy"):
        if type(facts[field]) not in (int, float) or not 0 <= facts[field] <= 1:
            raise ValueError("Audience facts must be finite in [0,1]")
        normalized[field] = float(facts[field])
    if preset != "custom" and _canonical(normalized) != _canonical(PRESETS[preset]):
        raise ValueError("Audience preset does not match facts")
    expected = {
        "format": "angelos.comfyui-audience.v1", "synthetic": True, "adult": True,
        "preset": preset, "facts": normalized, "factsSchema": "angelos.facts.v1",
        "encoder": {"id": "structured-demo-v1", "dimension": 4},
        "upl": [AGE_BANDS.index(facts["ageBand"]) / 5, REGIONS.index(facts["region"]) / 3,
                normalized["openness"], normalized["energy"]],
    }
    expected["sha256"] = _digest(_canonical(expected))
    if _canonical(audience) != _canonical(expected):
        raise ValueError("Audience facts/UPL/hash binding mismatch")


def _campaign(manifest):
    campaign = manifest["campaign"]
    _keys(campaign, ("format", "id", "version", "referencePrompt", "audience", "bindingHash"), "campaign")
    version_two = manifest["format"] == PACK_FORMAT_V2
    _audience(campaign["audience"], version_two)
    if campaign["format"] != ("angelos.comfyui-campaign.v2" if version_two else "angelos.comfyui-campaign.v1"):
        raise ValueError("Unsupported campaign format")
    identifier_pattern = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}" if version_two else r"[A-Za-z][A-Za-z0-9_-]{0,63}"
    for field, pattern in (("id", identifier_pattern), ("version", r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")):
        if type(campaign[field]) is not str or re.fullmatch(pattern, campaign[field]) is None:
            raise ValueError(f"Invalid campaign {field}")
    if version_two and campaign["audience"]["context"]["campaignId"] != campaign["id"]:
        raise ValueError("Campaign does not match UPL context campaignId")
    _text(campaign["referencePrompt"], "reference prompt", 4096)
    unsigned = {key: value for key, value in campaign.items() if key != "bindingHash"}
    if campaign["bindingHash"] != _digest(_canonical(unsigned)):
        raise ValueError("Campaign binding hash mismatch")
    if _canonical(manifest["audience"]) != _canonical(campaign["audience"]):
        raise ValueError("Campaign/audience binding mismatch")


def _descriptor(value, filename):
    _keys(value, ("file", "sha256"), "file descriptor")
    if value["file"] != filename:
        raise ValueError("Unexpected filename or path traversal")
    return _hash(value["sha256"], filename)


def _workflow(directory, provenance, payloads):
    _keys(provenance, ("workflow", "apiPromptSha256", "uiWorkflow", "exportNodeId", "imageInput",
                       "seedFields", "checkpointReferences", *VERIFICATION_FLAGS), "provenance")
    if any(provenance[flag] is not False for flag in VERIFICATION_FLAGS):
        raise ValueError("Exporter v1 cannot attest workflow/model/hardware/native execution")
    workflow_hash = _descriptor(provenance["workflow"], "workflow.json")
    payload = _read_file(directory, "workflow.json", MAX_JSON_BYTES)
    if _digest(payload) != workflow_hash:
        raise ValueError("Workflow hash mismatch")
    payloads["workflow.json"] = payload
    workflow = _json(payload)
    _keys(workflow, ("apiPrompt", "extraPngInfo", "exportNodeId"), "workflow")
    prompt = workflow["apiPrompt"]
    if type(prompt) is not dict or not 1 <= len(prompt) <= 1024:
        raise ValueError("Expected 1-1024 API workflow nodes")
    seeds, checkpoints = [], []
    for identifier, node in prompt.items():
        _text(identifier, "node ID", 128)
        if type(node) is not dict or type(node.get("inputs")) is not dict:
            raise ValueError("Workflow node needs inputs")
        _text(node.get("class_type"), "node class", 128)
        for field, value in node["inputs"].items():
            if field.lower() == "seed" or field.lower().endswith("_seed"):
                seeds.append({"nodeId": identifier, "field": field,
                              "value": str(value) if type(value) is int else value,
                              "encoding": "decimal-integer" if type(value) is int else "unresolved-json"})
            if field in CHECKPOINT_FIELDS.get(node["class_type"], ()):
                checkpoints.append({"nodeId": identifier, "classType": node["class_type"],
                                    "field": field, "value": value, "verified": False})
    unique_id = _text(workflow["exportNodeId"], "export node ID", 128)
    exporter = prompt.get(unique_id)
    if exporter is None or exporter["class_type"] != "AngelosExportCreative":
        raise ValueError("Workflow must identify its Angelos exporter")
    image_input = exporter["inputs"].get("images")
    if (type(image_input) is not list or len(image_input) != 2 or type(image_input[0]) is not str
            or image_input[0] not in prompt or type(image_input[1]) is not int or image_input[1] < 0):
        raise ValueError("Invalid exporter upstream image connection")
    expected = {"exportNodeId": unique_id, "imageInput": image_input, "seedFields": seeds,
                "checkpointReferences": checkpoints, "apiPromptSha256": _digest(_canonical(prompt))}
    for field, value in expected.items():
        if _canonical(provenance[field]) != _canonical(value):
            raise ValueError(f"Workflow provenance mismatch: {field}")
    extra = workflow["extraPngInfo"]
    if extra is not None and type(extra) is not dict:
        raise ValueError("Invalid extra PNG info")
    ui_workflow = extra.get("workflow") if extra is not None else None
    if ui_workflow is None:
        if provenance["uiWorkflow"] is not None:
            raise ValueError("Unexpected UI workflow descriptor")
    else:
        if type(ui_workflow) is not dict:
            raise ValueError("UI workflow must be an object")
        ui_hash = _descriptor(provenance["uiWorkflow"], "workflow-ui.json")
        ui_bytes = _read_file(directory, "workflow-ui.json", MAX_JSON_BYTES)
        if _digest(ui_bytes) != ui_hash or _canonical(_json(ui_bytes)) != _canonical(ui_workflow):
            raise ValueError("UI workflow binding/hash mismatch")
        payloads["workflow-ui.json"] = ui_bytes


def _png(payload, width, height, pixel_hash):
    from PIL import Image

    if payload[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Expected native PNG bytes")
    offset, chunks = 8, []
    while offset < len(payload):
        if len(chunks) >= 4096:
            raise ValueError("PNG exceeds chunk count bound")
        if len(payload) - offset < 12:
            raise ValueError("Truncated PNG chunk")
        length = struct.unpack_from(">I", payload, offset)[0]
        end = offset + 12 + length
        kind = payload[offset + 4:offset + 8]
        if end > len(payload) or kind not in (b"IHDR", b"IDAT", b"IEND"):
            raise ValueError("Only bounded exporter RGB PNG chunks are accepted")
        expected_crc = struct.unpack_from(">I", payload, end - 4)[0]
        if zlib.crc32(memoryview(payload)[offset + 4:end - 4]) != expected_crc:
            raise ValueError("PNG chunk CRC mismatch")
        if kind == b"IHDR":
            if chunks or length != 13:
                raise ValueError("Invalid PNG header")
            header = struct.unpack_from(">IIBBBBB", payload, offset + 8)
            if header != (width, height, 8, 2, 0, 0, 0):
                raise ValueError("PNG dimensions/mode/depth differ from declared native RGB8")
        elif not chunks or chunks[0] != b"IHDR":
            raise ValueError("PNG needs an initial IHDR")
        if kind == b"IEND" and (length != 0 or end != len(payload)):
            raise ValueError("Invalid PNG end or trailing bytes")
        chunks.append(kind)
        offset = end
    if len(chunks) < 3 or chunks[-1] != b"IEND" or any(kind != b"IDAT" for kind in chunks[1:-1]):
        raise ValueError("Incomplete PNG")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (width, height) or getattr(image, "n_frames", 1) != 1:
                raise ValueError("PNG must be a single native RGB8 frame")
            image.load()
            if _digest(image.tobytes()) != pixel_hash:
                raise ValueError("Decoded PNG pixel hash mismatch")
    except (OSError, SyntaxError) as error:
        raise ValueError("Invalid PNG pixel stream") from error


def _validate_pack(directory, allow_test_fixture):
    manifest_bytes = _read_file(directory, "manifest.json", MAX_JSON_BYTES)
    manifest = _json(manifest_bytes)
    _keys(manifest, ("format", "kind", "sourceKind", "status", "testOnly", "packId", "createdAt", "campaign",
                     "audience", "imageContract", "provenance", "images", "training", "promotion"), "manifest")
    if manifest["format"] not in (PACK_FORMAT, PACK_FORMAT_V2) or manifest["status"] != "complete":
        raise ValueError("Expected a complete teacher pack v1 or v2")
    fixture = manifest["testOnly"]
    if type(fixture) is not bool or manifest["training"] is not False or manifest["promotion"] is not False:
        raise ValueError("Invalid pack evidence flags")
    if fixture and not allow_test_fixture:
        raise ValueError("Test fixture requires explicit --allow-test-fixture opt-in")
    if manifest["kind"] != ("test-fixture" if fixture else "hd-teacher-pack") or manifest["sourceKind"] != ("comfyui-test-fixture" if fixture else "comfyui-local-render"):
        raise ValueError("Inconsistent fixture provenance")
    if type(manifest["packId"]) is not str or re.fullmatch(r"[0-9a-f]{32}", manifest["packId"]) is None:
        raise ValueError("Invalid pack ID")
    created = datetime.fromisoformat(_text(manifest["createdAt"], "createdAt", 64))
    if created.utcoffset() != timedelta(0):
        raise ValueError("createdAt must be a UTC timestamp")
    _campaign(manifest)
    contract = manifest["imageContract"]
    _keys(contract, ("inputLayout", "inputDtype", "inputDevice", "pngMode", "pngBitDepth", "spatialTransform", "quantization", "hdSizeGatePassed"), "image contract")
    expected = {"inputLayout": "NHWC", "pngMode": "RGB", "pngBitDepth": 8, "spatialTransform": "none",
                "quantization": "float32, floor(value * 255 + 0.5), uint8; no clipping"}
    if any(_canonical(contract[field]) != _canonical(value) for field, value in expected.items()):
        raise ValueError("Unsupported image contract; resizing/conversion is forbidden")
    if contract["inputDtype"] not in ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64"):
        raise ValueError("Unsupported input floating dtype")
    _text(contract["inputDevice"], "input device provenance", 128)
    images = manifest["images"]
    if type(images) is not list or not 1 <= len(images) <= 8:
        raise ValueError("Pack must have 1-8 images")
    payloads = {"manifest.json": manifest_bytes}
    _workflow(directory, manifest["provenance"], payloads)
    total_pixels, total_bytes = 0, 0
    batch_size = None
    for index, entry in enumerate(images):
        _keys(entry, ("index", "file", "width", "height", "sha256", "pixelSha256"), "image entry")
        if type(entry["index"]) is not int or entry["index"] != index:
            raise ValueError("Image indices must be contiguous and unique")
        name = ("TEST_ONLY-" if fixture else "") + f"image-{index:03}.png"
        if entry["file"] != name:
            raise ValueError("Invalid image filename or path traversal")
        width = _bound_integer(entry["width"], "width", 1, 4096)
        height = _bound_integer(entry["height"], "height", 1, 4096)
        if batch_size is not None and batch_size != (width, height):
            raise ValueError("Export batch images must have equal native dimensions")
        batch_size = (width, height)
        total_pixels += width * height
        if total_pixels > MAX_PIXELS:
            raise ValueError("Pack exceeds total pixel bound")
        hd_size = (width >= 1920 and height >= 1080) or (width >= 1080 and height >= 1920)
        if contract["hdSizeGatePassed"] is not hd_size or (not fixture and not hd_size):
            raise ValueError("HD size gate requires actual 1920x1080 or portrait dimensions")
        image_hash = _hash(entry["sha256"], name)
        pixel_hash = _hash(entry["pixelSha256"], "pixels")
        payload = _read_file(directory, name, MAX_PNG_BYTES - total_bytes)
        total_bytes += len(payload)
        if _digest(payload) != image_hash:
            raise ValueError("PNG file hash mismatch")
        _png(payload, width, height, pixel_hash)
        payloads[name] = payload
    if set(os.listdir(directory)) != set(payloads):
        raise ValueError("Pack contains missing or unexpected files")
    return manifest, payloads


def _publish(path, payloads):
    parent = _open_directory(path.parent)
    directory = None
    created_files = []
    try:
        os.mkdir(path.name, mode=0o700, dir_fd=parent)
        directory = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            for name, payload in payloads.items():
                descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory)
                created_files.append(name)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                    os.fchmod(handle.fileno(), 0o444)
            os.fchmod(directory, 0o555)
        except BaseException:
            os.fchmod(directory, 0o700)
            for name in created_files:
                os.unlink(name, dir_fd=directory)
            if os.stat(path.name, dir_fd=parent, follow_symlinks=False).st_ino == os.fstat(directory).st_ino:
                os.rmdir(path.name, dir_fd=parent)
            raise
    finally:
        if directory is not None:
            os.close(directory)
        os.close(parent)


def import_teacher_pack(pack_dir, output_dir, *, allow_test_fixture=False):
    """Import complete, hash-bound local evidence into a new read-only directory."""
    _require_posix()
    if type(allow_test_fixture) is not bool:
        raise ValueError("allow_test_fixture must be an explicit bool")
    source, output = _path(pack_dir), _path(output_dir)
    if output == source or source in output.parents:
        raise ValueError("Output must not be inside the input pack")
    directory = _open_directory(source)
    try:
        manifest, payloads = _validate_pack(directory, allow_test_fixture)
    finally:
        os.close(directory)
    manifest_hash = _digest(payloads["manifest.json"])
    version_two = manifest["format"] == PACK_FORMAT_V2
    receipt = {
        "format": "angelos.comfyui-import.v2" if version_two else "angelos.comfyui-import.v1",
        "id": "comfy-" + manifest_hash,
        "status": "complete", "fixtureOnly": manifest["testOnly"],
        "manifest": {"file": "manifest.json", "sha256": manifest_hash},
        "campaignBindingHash": manifest["campaign"]["bindingHash"],
        "audienceSha256": manifest["audience"]["sha256"], "images": manifest["images"],
        "spatialTransform": "none", "workflowExecuted": False, "downloaded": False,
        "verification": {flag: False for flag in VERIFICATION_FLAGS},
        "limitations": ["Hashes verify consistency, not authenticity or upstream native generation.",
                        "No model/hardware execution, brand fidelity, quality or CTR uplift is certified.",
                        ("Export v2 preserves opaque embeddings; encoder execution and ingredients hash are not independently verified."
                         if version_two else "Export v1 accepts only synthetic adult structured-demo-v1 audience metadata, not opaque encoders."),
                        "Review profileId identifies a synthetic audience configuration, not a stable person or ledger identity.",
                        "Changed facts change the audience hash; configuration groups cannot prove person-disjoint train/holdout across updates.",
                        "Review templates are unscored drafts, not ledger records; no click labels or automatic training are supplied."],
    }
    reviews = {
        "format": "angelos.review-template.v2" if version_two else "angelos.review-template.v1",
        "status": "unreviewed", "fixtureOnly": manifest["testOnly"],
        "instructions": (
            "Copy this template before editing. Supply actual quality/brand scores and source; no click evidence or automatic training is supplied. "
            "profileId is a synthetic audience configuration group, not a stable person or ledger identity. "
            "Changed facts change its hash, so these groups cannot prove person-disjoint train/holdout across updates. "
            "Do not insert this draft directly into an exposure/click ledger."),
        "records": [{"imageFile": entry["file"], "imageSha256": entry["sha256"],
                     "profileId": "comfy-audience-" + manifest["audience"]["sha256"],
                     "creativeId": manifest["packId"] + f"-{entry['index']:03}",
                     "upl": manifest["audience"]["upl"], "campaignBindingHash": manifest["campaign"]["bindingHash"],
                     "source": None, "quality": None, "brand": None} for entry in manifest["images"]],
    }
    if version_two:
        receipt["embeddingExecutionVerified"] = False
        receipt["ingredientsHashVerified"] = False
        reviews["audience"] = manifest["audience"]
    payloads["review-template.json"] = _canonical(reviews) + b"\n"
    payloads["import.json"] = _canonical(receipt) + b"\n"
    _publish(output, payloads)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    importer = subcommands.add_parser("import", help="Validate and copy local exported pixels; never execute a workflow")
    importer.add_argument("pack_dir")
    importer.add_argument("--output", required=True)
    importer.add_argument("--allow-test-fixture", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        receipt = import_teacher_pack(arguments.pack_dir, arguments.output, allow_test_fixture=arguments.allow_test_fixture)
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"comfy import: {error}\n")
    print(json.dumps(receipt, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
