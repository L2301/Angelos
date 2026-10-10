"""Export existing ComfyUI pixels and bounded provenance; never run a teacher."""

import hashlib
import json
import math
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PACK_FORMAT = "angelos.comfyui-teacher-pack.v1"
PACK_FORMAT_V2 = "angelos.comfyui-teacher-pack.v2"
MAX_BATCH = 8
MAX_SIDE = 4096
MAX_PIXELS = 4096 ** 2
MAX_WORKFLOW_BYTES = 2 * 1024 * 1024
AGE_BANDS = ("18-24", "25-34", "35-44", "45-54", "55-64", "65+")
REGIONS = ("NA", "EU", "APAC", "OTHER")
PRESETS = {
    "quiet": {"ageBand": "35-44", "region": "NA", "openness": 0.25, "energy": 0.2},
    "balanced": {"ageBand": "35-44", "region": "NA", "openness": 0.5, "energy": 0.5},
    "bold": {"ageBand": "35-44", "region": "NA", "openness": 0.85, "energy": 0.9},
}
CHECKPOINT_FIELDS = {
    "CheckpointLoaderSimple": ("ckpt_name",),
    "CheckpointLoader": ("ckpt_name", "config_name"),
    "UNETLoader": ("unet_name",),
    "VAELoader": ("vae_name",),
    "CLIPLoader": ("clip_name",),
    "DualCLIPLoader": ("clip_name1", "clip_name2"),
    "TripleCLIPLoader": ("clip_name1", "clip_name2", "clip_name3"),
    "LoraLoader": ("lora_name",),
    "LoraLoaderModelOnly": ("lora_name",),
}


def bounded_text(value, name, maximum, *, empty=False):
    if type(value) is not str or "\0" in value or (not empty and not value.strip()):
        raise ValueError(f"{name} must be nonempty text without NUL characters")
    if len(value) > maximum or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{name} exceeds {maximum} UTF-8 bytes")
    return value


def json_bytes(value, maximum=MAX_WORKFLOW_BYTES):
    budget = [50000]

    def check(item, depth):
        budget[0] -= 1
        if depth > 32 or budget[0] < 0:
            raise ValueError("JSON exceeds depth or item limit")
        if item is None or type(item) in (bool, int):
            if type(item) is int and item.bit_length() > 256:
                raise ValueError("JSON integer exceeds 256 bits")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
        elif type(item) is str:
            bounded_text(item, "JSON string", 65536, empty=True)
        elif type(item) is list:
            if len(item) > budget[0]:
                raise ValueError("JSON exceeds item limit")
            for child in item:
                check(child, depth + 1)
        elif type(item) is dict:
            if len(item) > budget[0]:
                raise ValueError("JSON exceeds item limit")
            for key, child in item.items():
                bounded_text(key, "JSON key", 256)
                check(child, depth + 1)
        else:
            raise ValueError("Provenance must contain only plain JSON values")

    check(value, 0)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > maximum:
        raise ValueError(f"JSON exceeds {maximum} bytes")
    return encoded


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate facts field: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def make_audience(preset="quiet", facts_json="{}", synthetic=True):
    if synthetic is not True:
        raise ValueError("Only synthetic adult facts are authorized; synthetic must be true")
    if type(preset) is not str or preset not in (*PRESETS, "custom"):
        raise ValueError("Unknown synthetic audience preset")
    bounded_text(facts_json, "facts_json", 4096)
    try:
        facts = json.loads(facts_json, object_pairs_hook=unique_fields, parse_constant=reject_constant)
    except RecursionError as error:
        raise ValueError("Facts JSON is nested too deeply") from error
    if preset != "custom":
        if facts != {}:
            raise ValueError("Preset audiences require empty facts_json; choose custom for explicit facts")
        facts = dict(PRESETS[preset])
    if type(facts) is not dict or set(facts) != {"ageBand", "region", "openness", "energy"}:
        raise ValueError("Facts require exactly ageBand, region, openness and energy")
    if type(facts["ageBand"]) is not str or facts["ageBand"] not in AGE_BANDS:
        raise ValueError("ageBand must be an allowed adult band")
    if type(facts["region"]) is not str or facts["region"] not in REGIONS:
        raise ValueError("region must be an allowed coarse region")
    for field in ("openness", "energy"):
        value = facts[field]
        if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
            raise ValueError(f"{field} must be a finite number in [0,1]")
        facts[field] = float(value)
    audience = {
        "format": "angelos.comfyui-audience.v1", "synthetic": True, "adult": True,
        "preset": preset, "facts": facts, "factsSchema": "angelos.facts.v1",
        "encoder": {"id": "structured-demo-v1", "dimension": 4},
        "upl": [AGE_BANDS.index(facts["ageBand"]) / 5, REGIONS.index(facts["region"]) / 3,
                facts["openness"], facts["energy"]],
    }
    return {**audience, "sha256": digest(json_bytes(audience))}


def validate_audience(audience):
    if type(audience) is dict and audience.get("format") == "angelos.comfyui-audience.v2":
        from .upl_v2 import validate_audience as validate_v2
        json_bytes(audience, 65536)
        return validate_v2(audience)
    encoded = json_bytes(audience, 16384)
    if type(audience) is not dict or not {"preset", "facts", "synthetic"}.issubset(audience):
        raise ValueError("Expected Angelos audience metadata")
    facts_json = json_bytes(audience["facts"]).decode("utf-8") if audience["preset"] == "custom" else "{}"
    expected = make_audience(audience["preset"], facts_json, audience["synthetic"])
    if encoded != json_bytes(expected):
        raise ValueError("Audience facts, UPL, flags, schema or hash do not match")
    return expected


def make_campaign(audience, campaign_id="aurora", version="v1", reference_prompt="A fictional Aurora campaign"):
    audience = validate_audience(audience)
    version_two = audience["format"] == "angelos.comfyui-audience.v2"
    identifier_pattern = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}" if version_two else r"[A-Za-z][A-Za-z0-9_-]{0,63}"
    if type(campaign_id) is not str or not re.fullmatch(identifier_pattern, campaign_id):
        raise ValueError("campaign_id must be a bounded ASCII identifier, not a path")
    if type(version) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version):
        raise ValueError("version must be a bounded ASCII version, not a path")
    bounded_text(reference_prompt, "reference_prompt", 4096)
    if version_two and audience["context"]["campaignId"] != campaign_id:
        raise ValueError("Campaign does not match UPL context campaignId")
    campaign = {
        "format": "angelos.comfyui-campaign.v2" if version_two else "angelos.comfyui-campaign.v1",
        "id": campaign_id, "version": version,
        "referencePrompt": reference_prompt, "audience": audience,
    }
    return {**campaign, "bindingHash": digest(json_bytes(campaign))}


def validate_campaign(campaign):
    version_two = type(campaign) is dict and campaign.get("format") == "angelos.comfyui-campaign.v2"
    encoded = json_bytes(campaign, 131072 if version_two else 32768)
    if type(campaign) is not dict or not {"audience", "id", "version", "referencePrompt"}.issubset(campaign):
        raise ValueError("Expected Angelos campaign metadata")
    expected = make_campaign(campaign["audience"], campaign["id"], campaign["version"], campaign["referencePrompt"])
    if encoded != json_bytes(expected):
        raise ValueError("Campaign metadata or binding hash does not match")
    return expected


def capture_workflow(prompt, extra_pnginfo, unique_id):
    bounded_text(unique_id, "unique_id", 128)
    payload = json_bytes({"apiPrompt": prompt, "extraPngInfo": extra_pnginfo, "exportNodeId": unique_id})
    workflow = json.loads(payload)
    prompt = workflow["apiPrompt"]
    extra = workflow["extraPngInfo"]
    if type(prompt) is not dict or not 1 <= len(prompt) <= 1024:
        raise ValueError("A complete ComfyUI PROMPT with 1-1024 nodes is required")
    for identifier, node in prompt.items():
        bounded_text(identifier, "workflow node ID", 128)
        if type(node) is not dict or type(node.get("inputs")) is not dict:
            raise ValueError("Each API prompt node requires an inputs object")
        bounded_text(node.get("class_type"), "class_type", 128)
    exporter = prompt.get(unique_id)
    if exporter is None or exporter["class_type"] != "AngelosExportCreative":
        raise ValueError("unique_id must identify AngelosExportCreative in PROMPT")
    image_input = exporter["inputs"].get("images")
    if (type(image_input) is not list or len(image_input) != 2 or type(image_input[0]) is not str
            or image_input[0] not in prompt or type(image_input[1]) is not int or image_input[1] < 0):
        raise ValueError("Export images must have an upstream node connection in PROMPT")
    if extra is not None and type(extra) is not dict:
        raise ValueError("EXTRA_PNGINFO must be a JSON object or absent")
    ui_workflow = extra.get("workflow") if extra is not None else None
    if ui_workflow is not None and type(ui_workflow) is not dict:
        raise ValueError("EXTRA_PNGINFO.workflow must be a JSON object")
    seeds, checkpoints = [], []
    for identifier, node in prompt.items():
        for field, value in node["inputs"].items():
            if field.lower() == "seed" or field.lower().endswith("_seed"):
                seeds.append({"nodeId": identifier, "field": field,
                              "value": str(value) if type(value) is int else value,
                              "encoding": "decimal-integer" if type(value) is int else "unresolved-json"})
            if field in CHECKPOINT_FIELDS.get(node["class_type"], ()):
                checkpoints.append({"nodeId": identifier, "classType": node["class_type"],
                                    "field": field, "value": value, "verified": False})
    return payload, prompt, ui_workflow, image_input, seeds, checkpoints


def validate_images(images, test_only):
    if type(test_only) is not bool:
        raise ValueError("test_only must be an explicit bool")
    if (not isinstance(images, torch.Tensor) or images.layout != torch.strided
            or images.ndim != 4 or not images.is_floating_point() or images.device.type == "meta"):
        raise ValueError("IMAGE must be a dense floating torch.Tensor in NHWC layout")
    batch, height, width, channels = images.shape
    if not 1 <= batch <= MAX_BATCH or channels != 3 or not 1 <= height <= MAX_SIDE or not 1 <= width <= MAX_SIDE:
        raise ValueError("IMAGE requires 1-8 RGB images with each side in [1,4096]")
    if batch * height * width > MAX_PIXELS:
        raise ValueError("IMAGE batch exceeds the total 4096² pixel budget")
    hd_size = (width >= 1920 and height >= 1080) or (width >= 1080 and height >= 1920)
    if not hd_size and not test_only:
        raise ValueError("HD export requires at least 1920x1080 or 1080x1920; no resizing is performed")
    for frame in images:
        pixels = frame.detach()
        if not bool(torch.isfinite(pixels).all()) or bool((pixels < 0).any()) or bool((pixels > 1).any()):
            raise ValueError("IMAGE pixels must be finite and in [0,1]; clipping is forbidden")
    return batch, height, width, hd_size


def write_new(path, payload):
    with path.open("xb") as stream:
        stream.write(payload)


def export_teacher_pack(images, campaign, prompt, extra_pnginfo, unique_id, *, test_only=False):
    """Save supplied pixels; test_only bypasses only HD size, never other bounds."""
    import folder_paths

    campaign = validate_campaign(campaign)
    batch, height, width, hd_size = validate_images(images, test_only)
    workflow_bytes, api_prompt, ui_workflow, image_input, seeds, checkpoints = capture_workflow(prompt, extra_pnginfo, unique_id)
    output_root = Path(folder_paths.get_output_directory()).resolve(strict=True)
    if not output_root.is_dir():
        raise ValueError("ComfyUI output directory must exist")
    export_root = output_root / "angelos"
    if export_root.is_symlink():
        raise ValueError("Refusing a symlinked angelos export directory")
    export_root.mkdir(exist_ok=True)
    if not export_root.is_dir() or export_root.resolve() != export_root:
        raise ValueError("Angelos export directory must stay inside ComfyUI output")
    pack_id = uuid.uuid4().hex
    prefix = "TEST_ONLY-" if test_only else ""
    slug = re.sub(r"[^a-z0-9_-]", "-", campaign["id"].lower())[:64]
    directory = export_root / f"{prefix}{slug}-{pack_id}"
    directory.mkdir(exist_ok=False)
    try:
        write_new(directory / "workflow.json", workflow_bytes)
        ui_descriptor = None
        if ui_workflow is not None:
            ui_bytes = json_bytes(ui_workflow)
            write_new(directory / "workflow-ui.json", ui_bytes)
            ui_descriptor = {"file": "workflow-ui.json", "sha256": digest(ui_bytes)}
        entries, previews = [], []
        subfolder = directory.relative_to(output_root).as_posix()
        for index in range(batch):
            pixels = images[index].detach().to(device="cpu", dtype=torch.float32).numpy()
            quantized = np.floor(pixels * np.float32(255) + np.float32(0.5)).astype(np.uint8)
            filename = f"{prefix}image-{index:03}.png"
            with (directory / filename).open("xb") as stream:
                Image.fromarray(quantized).save(stream, format="PNG", compress_level=4)
            with (directory / filename).open("rb") as stream:
                image_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            entries.append({"index": index, "file": filename, "width": width, "height": height,
                            "sha256": image_hash, "pixelSha256": digest(quantized.tobytes())})
            previews.append({"filename": filename, "subfolder": subfolder, "type": "output"})
        manifest = {
            "format": PACK_FORMAT_V2 if campaign["format"] == "angelos.comfyui-campaign.v2" else PACK_FORMAT,
            "kind": "test-fixture" if test_only else "hd-teacher-pack",
            "sourceKind": "comfyui-test-fixture" if test_only else "comfyui-local-render",
            "status": "complete", "testOnly": test_only, "packId": pack_id,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "campaign": campaign, "audience": campaign["audience"],
            "imageContract": {
                "inputLayout": "NHWC", "inputDtype": str(images.dtype), "inputDevice": str(images.device),
                "pngMode": "RGB", "pngBitDepth": 8, "spatialTransform": "none",
                "quantization": "float32, floor(value * 255 + 0.5), uint8; no clipping",
                "hdSizeGatePassed": hd_size,
            },
            "provenance": {
                "workflow": {"file": "workflow.json", "sha256": digest(workflow_bytes)},
                "apiPromptSha256": digest(json_bytes(api_prompt)), "uiWorkflow": ui_descriptor,
                "exportNodeId": unique_id, "imageInput": image_input,
                "seedFields": seeds, "checkpointReferences": checkpoints,
                "modelHashesVerified": False, "workflowExecutionVerified": False,
                "upstreamNativeGenerationVerified": False, "hardwareExecutionVerified": False,
                "allRandomnessCaptured": False,
            },
            "images": entries, "training": False, "promotion": False,
        }
        write_new(directory / "manifest.json", json_bytes(manifest))
    except Exception:
        shutil.rmtree(directory)
        raise
    return {"manifest": manifest, "manifestPath": f"{subfolder}/manifest.json", "images": previews}


class AngelosAudience:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset": ([*PRESETS, "custom"],),
            "facts_json": ("STRING", {"default": "{}", "multiline": True}),
            "synthetic": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("ANGELOS_AUDIENCE", "STRING")
    RETURN_NAMES = ("audience", "audience_json")
    FUNCTION = "build"
    CATEGORY = "Angelos"

    def build(self, preset, facts_json, synthetic):
        audience = make_audience(preset, facts_json, synthetic)
        return audience, json_bytes(audience).decode("utf-8")


class AngelosCampaign:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audience": ("ANGELOS_AUDIENCE", {"forceInput": True}),
            "campaign_id": ("STRING", {"default": "aurora"}),
            "version": ("STRING", {"default": "v1"}),
            "reference_prompt": ("STRING", {"default": "A fictional Aurora campaign", "multiline": True}),
        }}

    RETURN_TYPES = ("ANGELOS_CAMPAIGN", "STRING")
    RETURN_NAMES = ("campaign", "prompt")
    FUNCTION = "build"
    CATEGORY = "Angelos"

    def build(self, audience, campaign_id, version, reference_prompt):
        campaign = make_campaign(audience, campaign_id, version, reference_prompt)
        return campaign, campaign["referencePrompt"]


class AngelosExportCreative:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "campaign": ("ANGELOS_CAMPAIGN", {"forceInput": True}),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest_path",)
    FUNCTION = "export"
    CATEGORY = "Angelos"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def export(self, images, campaign, prompt=None, extra_pnginfo=None, unique_id=None):
        exported = export_teacher_pack(images, campaign, prompt, extra_pnginfo, unique_id)
        return {"ui": {"images": exported["images"]}, "result": (exported["manifestPath"],)}
