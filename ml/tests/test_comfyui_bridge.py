"""Offline ComfyUI export fixtures; never model inference or GPU-quality evidence."""

import copy
import hashlib
import importlib.util
import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "integrations" / "comfyui_angelos"
SPEC = importlib.util.spec_from_file_location("angelos_comfyui_bridge_test_package", PACKAGE / "__init__.py",
                                            submodule_search_locations=[str(PACKAGE)])
BRIDGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BRIDGE
SPEC.loader.exec_module(BRIDGE)
NODES = sys.modules[f"{SPEC.name}.nodes"]
UPL = sys.modules[f"{SPEC.name}.upl"]


class ComfyUIBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        folder_paths = ModuleType("folder_paths")
        folder_paths.get_output_directory = lambda: str(self.directory)
        injected = patch.dict(sys.modules, {"folder_paths": folder_paths})
        injected.start()
        self.addCleanup(injected.stop)
        self.audience = NODES.make_audience()
        self.campaign = NODES.make_campaign(self.audience, "aurora", "v1", "Fictional Aurora, quiet composition")
        self.prompt = {
            "1": {"class_type": "LoadImage", "inputs": {"image": "fixture.png"}},
            "2": {"class_type": "AngelosAudience", "inputs": {"preset": "quiet", "facts_json": "{}", "synthetic": True}},
            "3": {"class_type": "AngelosCampaign", "inputs": {"audience": ["2", 0], "campaign_id": "aurora",
                  "version": "v1", "reference_prompt": "Fictional Aurora, quiet composition"}},
            "4": {"class_type": "KSampler", "inputs": {"seed": 2 ** 64 - 1, "steps": 4}},
            "5": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "../unverified/checkpoint.safetensors"}},
            "9": {"class_type": "AngelosExportCreative", "inputs": {"images": ["1", 0], "campaign": ["3", 0]}},
        }
        self.extra = {"workflow": {"version": 0.4, "nodes": [{"id": 9, "type": "AngelosExportCreative"}],
                                   "links": [], "extra": {"fixture": True}}, "operatorNote": "Synthetic fixture"}
        self.images = torch.tensor([0.0, 0.5, 1.0]).reshape(1, 1, 1, 3).expand(1, 4, 6, 3).clone()

    def export_fixture(self, **overrides):
        arguments = {"images": self.images, "campaign": self.campaign, "prompt": self.prompt,
                     "extra_pnginfo": self.extra, "unique_id": "9", "test_only": True}
        arguments.update(overrides)
        return NODES.export_teacher_pack(**arguments)

    def read_pack(self, result):
        path = self.directory / result["manifestPath"]
        return path.parent, json.loads(path.read_text())

    def test_node_registration_and_supported_comfy_contract(self):
        self.assertEqual(set(BRIDGE.NODE_CLASS_MAPPINGS), {
            "AngelosAudience", "AngelosCampaign", "AngelosExportCreative", "AngelosParseUPL",
            "AngelosUPLFactString", "AngelosUPLFactFloat", "AngelosUPLVectorFloat", "AngelosScalarToInt",
            "AngelosScalarCompare", "AngelosScalarMap", "AngelosPromptTemplate", "AngelosChoosePrompt",
            "AngelosUPLEmbeddingBlock", "AngelosUPLRecentIntentText",
        })
        self.assertEqual(set(BRIDGE.NODE_DISPLAY_NAME_MAPPINGS), set(BRIDGE.NODE_CLASS_MAPPINGS))
        exporter = BRIDGE.NODE_CLASS_MAPPINGS["AngelosExportCreative"]
        self.assertTrue(exporter.OUTPUT_NODE)
        self.assertEqual(exporter.INPUT_TYPES()["hidden"], {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"})
        self.assertEqual(set(exporter.INPUT_TYPES()["required"]), {"images", "campaign"})
        self.assertNotIn("test_only", inspect.signature(exporter.export).parameters)
        self.assertNotIn("output_dir", inspect.signature(NODES.export_teacher_pack).parameters)
        self.assertTrue(math.isnan(exporter.IS_CHANGED()))

    def test_audience_presets_custom_facts_and_encoder_order(self):
        for preset in NODES.PRESETS:
            audience, serialized = NODES.AngelosAudience().build(preset, "{}", True)
            self.assertEqual(audience, json.loads(serialized))
            self.assertTrue(audience["synthetic"])
            self.assertTrue(audience["adult"])
            self.assertEqual(audience["encoder"], {"id": "structured-demo-v1", "dimension": 4})
            self.assertEqual(audience["upl"][:2], [0.4, 0.0])
        facts = {"ageBand": "65+", "region": "OTHER", "openness": 0, "energy": 1}
        custom = NODES.make_audience("custom", json.dumps(facts))
        self.assertEqual(custom["upl"], [1.0, 1.0, 0.0, 1.0])
        self.assertEqual(custom, NODES.validate_audience(custom))

    def test_synthetic_false_and_nonboolean_declarations_are_rejected(self):
        for value in (False, 0, 1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                NODES.make_audience(synthetic=value)
        for field, value in (("synthetic", False), ("adult", False), ("sha256", "0" * 64), ("upl", [0, 0, 0, 0])):
            audience = copy.deepcopy(self.audience)
            audience[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                NODES.make_campaign(audience)

    def test_invalid_facts_and_unexpected_personal_fields(self):
        for changes in ({"ageBand": "under-18"}, {"region": "../../outside"}, {"energy": True},
                        {"openness": -0.1}, {"energy": 1.1}, {"name": "Real person"}):
            facts = {**self.audience["facts"], **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                NODES.make_audience("custom", json.dumps(facts))
        for text in ("null", "[]", "{}", '{"energy":0,"energy":1}', '{"energy":NaN}',
                     '{"energy":Infinity}', "{" * 100, " " * 4097):
            with self.subTest(text=text[:20]), self.assertRaises(ValueError):
                NODES.make_audience("custom", text)
        with self.assertRaises(ValueError):
            NODES.make_audience("quiet", json.dumps(self.audience["facts"]))

    def test_campaign_prompt_binding_and_version_changes(self):
        campaign, prompt = NODES.AngelosCampaign().build(self.audience, "Aurora_1", "v1.2", "Exact prompt\n第二行")
        self.assertEqual(prompt, "Exact prompt\n第二行")
        self.assertEqual(campaign, NODES.validate_campaign(campaign))
        changed = NODES.make_campaign(self.audience, "Aurora_1", "v2", prompt)
        self.assertNotEqual(campaign["bindingHash"], changed["bindingHash"])
        self.assertEqual(self.audience, NODES.make_audience())

    def test_campaign_path_and_prompt_injection_rejected(self):
        for identifier in ("../escape", "/tmp/escape", "C:\\escape", "a/b", "a\\b", "a%2fb", ".", "a" * 65, "", "a\0b"):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                NODES.make_campaign(self.audience, identifier)
        for version in ("../v1", "v/1", "v" * 65, ""):
            with self.assertRaises(ValueError):
                NODES.make_campaign(self.audience, version=version)
        for prompt in ("", "\0", "🙂" * 1025):
            with self.assertRaises(ValueError):
                NODES.make_campaign(self.audience, reference_prompt=prompt)
        for field, value in (("id", "../escape"), ("referencePrompt", "different"), ("bindingHash", "0" * 64), ("output_dir", "/tmp/escape")):
            campaign = copy.deepcopy(self.campaign)
            campaign[field] = value
            with self.assertRaises(ValueError):
                self.export_fixture(campaign=campaign)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_fixture_pack_roundtrip_pixels_hashes_and_workflow(self):
        before = self.images.clone()
        result = self.export_fixture()
        directory, manifest = self.read_pack(result)
        self.assertEqual(manifest, result["manifest"])
        self.assertEqual(manifest["format"], "angelos.comfyui-teacher-pack.v1")
        self.assertEqual(manifest["sourceKind"], "comfyui-test-fixture")
        self.assertEqual(manifest["kind"], "test-fixture")
        self.assertTrue(manifest["testOnly"])
        self.assertTrue(directory.name.startswith("TEST_ONLY-"))
        self.assertFalse(manifest["imageContract"]["hdSizeGatePassed"])
        self.assertEqual(manifest["imageContract"]["spatialTransform"], "none")
        entry = manifest["images"][0]
        self.assertTrue(entry["file"].startswith("TEST_ONLY-"))
        png = directory / entry["file"]
        self.assertEqual(entry["sha256"], hashlib.sha256(png.read_bytes()).hexdigest())
        with Image.open(png) as image:
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(image.size, (6, 4))
            expected = np.broadcast_to(np.array([0, 128, 255], dtype=np.uint8), (4, 6, 3))
            np.testing.assert_array_equal(np.asarray(image), expected)
            self.assertEqual(entry["pixelSha256"], hashlib.sha256(image.tobytes()).hexdigest())
            self.assertNotIn("workflow", image.info)
        workflow_bytes = (directory / "workflow.json").read_bytes()
        self.assertEqual(manifest["provenance"]["workflow"]["sha256"], hashlib.sha256(workflow_bytes).hexdigest())
        self.assertEqual(json.loads(workflow_bytes), {"apiPrompt": self.prompt, "extraPngInfo": self.extra, "exportNodeId": "9"})
        self.assertEqual(json.loads((directory / "workflow-ui.json").read_bytes()), self.extra["workflow"])
        self.assertEqual(manifest["audience"], manifest["campaign"]["audience"])
        torch.testing.assert_close(before, self.images, rtol=0, atol=0)

    def test_seed_and_checkpoint_metadata_are_not_execution_evidence(self):
        self.prompt["4"]["inputs"]["noise_seed"] = ["2", 1]
        result = self.export_fixture()
        provenance = result["manifest"]["provenance"]
        seed = next(item for item in provenance["seedFields"] if item["field"] == "seed")
        self.assertEqual(seed["value"], str(2 ** 64 - 1))
        self.assertEqual(seed["encoding"], "decimal-integer")
        self.assertIn({"nodeId": "4", "field": "noise_seed", "value": ["2", 1], "encoding": "unresolved-json"}, provenance["seedFields"])
        checkpoint = provenance["checkpointReferences"][0]
        self.assertEqual(checkpoint["value"], "../unverified/checkpoint.safetensors")
        self.assertFalse(checkpoint["verified"])
        for field in ("modelHashesVerified", "workflowExecutionVerified", "upstreamNativeGenerationVerified",
                      "hardwareExecutionVerified", "allRandomnessCaptured"):
            self.assertIs(provenance[field], False)

    def test_missing_ui_workflow_is_explicit_not_invented(self):
        result = self.export_fixture(extra_pnginfo=None)
        directory, manifest = self.read_pack(result)
        self.assertIsNone(manifest["provenance"]["uiWorkflow"])
        self.assertFalse((directory / "workflow-ui.json").exists())
        self.assertIsNone(json.loads((directory / "workflow.json").read_bytes())["extraPngInfo"])

    def test_workflow_hash_covers_all_nodes_and_is_key_order_independent(self):
        original = self.export_fixture()["manifest"]["provenance"]["workflow"]["sha256"]
        reordered = dict(reversed(list(self.prompt.items())))
        self.assertEqual(original, self.export_fixture(prompt=reordered)["manifest"]["provenance"]["workflow"]["sha256"])
        self.prompt["4"]["inputs"]["seed"] += 1
        changed = self.export_fixture()["manifest"]["provenance"]["workflow"]["sha256"]
        self.assertNotEqual(original, changed)

    def test_new_directory_per_export_and_no_overwrite(self):
        first = self.export_fixture()
        first_path = self.directory / first["manifestPath"]
        first_bytes = first_path.read_bytes()
        second = self.export_fixture()
        self.assertNotEqual(first["manifestPath"], second["manifestPath"])
        self.assertEqual(first_path.read_bytes(), first_bytes)
        identifier = first["manifest"]["packId"]
        with patch.object(NODES.uuid, "uuid4", return_value=SimpleNamespace(hex=identifier)):
            with self.assertRaises(FileExistsError):
                self.export_fixture()
        self.assertEqual(first_path.read_bytes(), first_bytes)

    def test_ui_export_has_hard_hd_gate_and_no_test_flag(self):
        exporter = NODES.AngelosExportCreative()
        with self.assertRaisesRegex(ValueError, "HD export"):
            exporter.export(self.images, self.campaign, self.prompt, self.extra, "9")
        with self.assertRaises(TypeError):
            exporter.export(self.images, self.campaign, self.prompt, self.extra, "9", test_only=True)
        for flag in (1, 0, "true", None):
            with self.assertRaises(ValueError):
                self.export_fixture(test_only=flag)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_native_landscape_and_portrait_ui_previews_without_resize(self):
        for height, width in ((1080, 1920), (1920, 1080), (1088, 1920)):
            with self.subTest(height=height, width=width):
                pixels = self.images[:, :1, :1, :].expand(1, height, width, 3)
                result = NODES.AngelosExportCreative().export(pixels, self.campaign, self.prompt, self.extra, "9")
                manifest_path = self.directory / result["result"][0]
                manifest = json.loads(manifest_path.read_bytes())
                self.assertEqual(manifest["sourceKind"], "comfyui-local-render")
                self.assertFalse(manifest["testOnly"])
                self.assertTrue(manifest["imageContract"]["hdSizeGatePassed"])
                preview = result["ui"]["images"][0]
                self.assertEqual(preview["type"], "output")
                with Image.open(self.directory / preview["subfolder"] / preview["filename"]) as image:
                    self.assertEqual(image.size, (width, height))
                    self.assertEqual(image.getpixel((width - 1, height - 1)), (0, 128, 255))

    def test_batch_limit_total_pixel_budget_and_axis_bounds(self):
        anchor = self.images[:, :1, :1, :]
        for shape in ((9, 2, 2, 3), (1, 4097, 1, 3), (1, 1, 4097, 3), (8, 1920, 1920, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                self.export_fixture(images=anchor.expand(shape))
        exported = self.export_fixture(images=self.images.expand(8, 4, 6, 3))
        self.assertEqual(len(exported["manifest"]["images"]), 8)
        self.assertEqual(len({entry["file"] for entry in exported["manifest"]["images"]}), 8)

    def test_invalid_tensor_layout_dtype_and_channels(self):
        for pixels in (np.zeros((1, 4, 6, 3)), torch.zeros(1, 3, 4, 6), torch.zeros(4, 6, 3),
                       torch.zeros(1, 4, 6, 4), torch.zeros(0, 4, 6, 3), torch.zeros(1, 0, 6, 3),
                       self.images.to(torch.uint8), self.images.to(torch.bool), self.images.to(torch.complex64),
                       torch.empty(1, 4, 6, 3, device="meta")):
            with self.subTest(kind=type(pixels)), self.assertRaises(ValueError):
                self.export_fixture(images=pixels)

    def test_nan_infinity_and_out_of_range_are_not_clipped(self):
        for value in (float("nan"), float("inf"), float("-inf"), -0.01, 1.01, 1.0000000001):
            pixels = self.images.to(torch.float64)
            pixels[0, 0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite.*\\[0,1\\]"):
                self.export_fixture(images=pixels)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_noncontiguous_and_bfloat_pixels_preserve_geometry(self):
        for pixels in (self.images.transpose(1, 2), self.images.to(torch.bfloat16)):
            result = self.export_fixture(images=pixels)
            entry = result["manifest"]["images"][0]
            self.assertEqual((entry["height"], entry["width"]), tuple(pixels.shape[1:3]))

    def test_workflow_required_and_export_connection_validated(self):
        for arguments in ({"prompt": None}, {"prompt": {}}, {"unique_id": "missing"}, {"unique_id": "4"},
                          {"unique_id": ""}, {"extra_pnginfo": []}, {"extra_pnginfo": {"workflow": []}}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.export_fixture(**arguments)
        for connection in (None, ["missing", 0], ["1", -1], ["1", True], "../image.png"):
            prompt = copy.deepcopy(self.prompt)
            prompt["9"]["inputs"]["images"] = connection
            with self.assertRaises(ValueError):
                self.export_fixture(prompt=prompt)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_provenance_size_depth_json_type_and_finite_bounds(self):
        nested = {}
        cursor = nested
        for index in range(40):
            cursor["child"] = {}
            cursor = cursor["child"]
        cycle = {}
        cycle["cycle"] = cycle
        for extra in ({"value": float("nan")}, {"value": torch.tensor(1)}, {1: "bad key"},
                      {"value": "a" * 65537}, nested, cycle,
                      {"items": ["a" * 60000] * 36}, {"items": [None] * 50001}, {"value": 1 << 257}):
            with self.subTest(kind=list(extra)[:1]), self.assertRaises(ValueError):
                self.export_fixture(extra_pnginfo=extra)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_symlinked_export_root_and_existing_nondirectory_rejected(self):
        outside = self.directory / "outside"
        outside.mkdir()
        export_root = self.directory / "angelos"
        export_root.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.export_fixture()
        self.assertEqual(list(outside.iterdir()), [])
        export_root.unlink()
        export_root.write_text("do not overwrite")
        with self.assertRaises(FileExistsError):
            self.export_fixture()
        self.assertEqual(export_root.read_text(), "do not overwrite")

    def test_failed_save_removes_only_its_new_incomplete_pack(self):
        existing = self.export_fixture()
        existing_path = self.directory / existing["manifestPath"]
        original = existing_path.read_bytes()
        with patch.object(NODES.Image.Image, "save", side_effect=OSError("fixture disk failure")):
            with self.assertRaisesRegex(OSError, "fixture disk failure"):
                self.export_fixture()
        self.assertEqual(existing_path.read_bytes(), original)
        self.assertEqual(len(list((self.directory / "angelos").iterdir())), 1)

    def wire_upl(self):
        return {"format": "angelos.upl-input.v1", "synthetic": True, "adult": True,
                "factsSchema": "angelos.facts.v1", "facts": self.audience["facts"],
                "encoder": self.audience["encoder"], "upl": self.audience["upl"]}

    def test_standalone_install_registers_without_angelos_imports(self):
        destination = self.directory / "custom_nodes" / "comfyui_angelos"
        destination.mkdir(parents=True)
        for filename in ("__init__.py", "nodes.py", "upl.py", "upl_v2.py"):
            (destination / filename).write_bytes((PACKAGE / filename).read_bytes())
        name = "angelos_standalone_install_test"
        spec = importlib.util.spec_from_file_location(name, destination / "__init__.py",
                                                    submodule_search_locations=[str(destination)])
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules):
            sys.modules[name] = module
            spec.loader.exec_module(module)
            self.assertEqual(set(module.NODE_CLASS_MAPPINGS), set(BRIDGE.NODE_CLASS_MAPPINGS))
            parser = module.NODE_CLASS_MAPPINGS["AngelosParseUPL"]()
            audience = parser.parse(json.dumps(self.wire_upl()))[0]
            self.assertEqual(audience["upl"], self.audience["upl"])

    def test_float_mapping_has_explicit_ranges_and_never_clips(self):
        mapper = UPL.AngelosScalarMap()
        self.assertEqual(mapper.RETURN_TYPES, ("FLOAT",))
        for value, expected in ((0, 2.0), (0.5, 4.75), (1, 7.5)):
            actual = mapper.map_value(value, 0, 1, 2, 7.5)[0]
            self.assertIs(type(actual), float)
            self.assertEqual(actual, expected)
        self.assertEqual(mapper.map_value(0.25, 0, 1, 0.2, 0.8), (0.35000000000000003,))
        self.assertEqual(mapper.map_value(0.5, 0, 1, 2, 2), (2.0,))
        for arguments in ((-0.1, 0, 1, 2, 7.5), (1.1, 0, 1, 2, 7.5),
                          (0.5, 1, 1, 2, 7.5), (0.5, 0, 1, 7.5, 2)):
            with self.assertRaises(ValueError):
                mapper.map_value(*arguments)
        for index in range(5):
            for invalid in (True, "1", float("nan"), float("inf"), -float("inf"), 1000001, 10 ** 1000):
                arguments = [0.5, 0, 1, 2, 7.5]
                arguments[index] = invalid
                with self.subTest(index=index, invalid=invalid), self.assertRaises(ValueError):
                    mapper.map_value(*arguments)

    def test_huge_fact_numbers_fail_validation_without_float_overflow(self):
        facts = {**self.audience["facts"], "energy": 10 ** 1000}
        with self.assertRaises(ValueError):
            NODES.make_audience("custom", json.dumps(facts))

    def test_unhashed_wire_parses_js_integer_numbers_and_computes_own_hash(self):
        document = self.wire_upl()
        document["upl"][1] = 0
        audience, serialized = UPL.AngelosParseUPL().parse(json.dumps(document))
        self.assertEqual(audience, NODES.make_audience("custom", json.dumps(document["facts"])))
        self.assertEqual(audience, json.loads(serialized))
        self.assertEqual(audience, NODES.validate_audience(audience))
        self.assertEqual(audience["upl"], [0.4, 0.0, 0.25, 0.2])
        document["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "no input hash"):
            UPL.parse_upl(json.dumps(document))

    def test_upl_versions_flags_facts_and_dimensions_are_checked(self):
        for field, value in (("format", "angelos.upl-input.v2"), ("factsSchema", "angelos.facts.v2"),
                             ("synthetic", False), ("synthetic", 1), ("adult", False), ("adult", "true"),
                             ("encoder", {"id": "learned-unknown-v1", "dimension": 4}),
                             ("encoder", {"id": "structured-demo-v1", "dimension": 4.0}),
                             ("encoder", {"id": "structured-demo-v1", "dimension": 5}),
                             ("upl", [0.4, 0, 0.25]), ("upl", [0.4, 0, 0.25, 0.2, 0]),
                             ("upl", [0.4, 0, 0.25, 0.9]), ("upl", [0.4, 0, True, 0.2]),
                             ("upl", [0.4, 0, float("nan"), 0.2]), ("upl", [0.4, 0, 1.1, 0.2]),
                             ("upl", [0.4, 0, "0.25", 0.2])):
            document = self.wire_upl()
            document[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                UPL.parse_upl(json.dumps(document))
        document = self.wire_upl()
        document["facts"] = {**document["facts"], "ageBand": "under-18"}
        with self.assertRaises(ValueError):
            UPL.parse_upl(json.dumps(document))
        for serialized in ('{"format":"one","format":"two"}', " " * 16385, "[" * 1100 + "]" * 1100):
            with self.assertRaises(ValueError):
                UPL.parse_upl(serialized)

    def test_all_structured_age_region_encodings_match_store_formula(self):
        for age_index, age_band in enumerate(NODES.AGE_BANDS):
            for region_index, region in enumerate(NODES.REGIONS):
                document = self.wire_upl()
                document["facts"] = {"ageBand": age_band, "region": region, "openness": 0, "energy": 1}
                document["upl"] = [age_index / 5, region_index / 3, 0, 1]
                parsed = UPL.parse_upl(json.dumps(document))
                self.assertEqual(parsed["upl"], document["upl"])

    def test_typed_extraction_and_opaque_index_are_not_json_only(self):
        audience = UPL.parse_upl(json.dumps(self.wire_upl()))
        self.assertEqual(UPL.AngelosUPLFactString().extract(audience, "region"), ("NA",))
        self.assertEqual(UPL.AngelosUPLFactString.RETURN_TYPES, ("STRING",))
        self.assertEqual(UPL.AngelosUPLFactFloat().extract(audience, "energy"), (0.2,))
        self.assertEqual(UPL.AngelosUPLFactFloat.RETURN_TYPES, ("FLOAT",))
        self.assertEqual(UPL.AngelosUPLVectorFloat().extract(audience, 2), (0.25,))
        for field in ("inferred_trait", "__class__", "energy"):
            with self.assertRaises(ValueError):
                UPL.AngelosUPLFactString().extract(audience, field)
        with self.assertRaises(ValueError):
            UPL.AngelosUPLFactFloat().extract(audience, "ageBand")
        for index in (-1, 4, True, 1.5):
            with self.assertRaises(ValueError):
                UPL.AngelosUPLVectorFloat().extract(audience, index)

    def test_explicit_integer_mapping_and_boolean_comparison(self):
        mapper = UPL.AngelosScalarToInt()
        self.assertEqual(mapper.RETURN_TYPES, ("INT",))
        self.assertEqual(mapper.map_value(0.25, 0, 1, 4, 8, "nearest"), (5,))
        self.assertEqual(mapper.map_value(0.2, 0, 1, 4, 8, "floor"), (4,))
        self.assertEqual(mapper.map_value(0.2, 0, 1, 4, 8, "ceil"), (5,))
        for arguments in ((True, 0, 1, 4, 8, "nearest"), (0.5, 1, 1, 4, 8, "nearest"),
                          (1.1, 0, 1, 4, 8, "nearest"), (0.5, 0, 1, 4, 8.0, "nearest"),
                          (0.5, 0, 1, 4, 8, "execute"), (0.5, 0, 1, 9, 8, "floor")):
            with self.assertRaises(ValueError):
                mapper.map_value(*arguments)
        comparer = UPL.AngelosScalarCompare()
        self.assertEqual(comparer.RETURN_TYPES, ("BOOLEAN",))
        for operator, expected in ((">=", True), (">", False), ("<=", True), ("<", False), ("==", True)):
            self.assertEqual(comparer.compare(0.5, 0.5, operator), (expected,))
        for value in (True, "0.5", float("inf"), float("nan"), 1000001):
            with self.assertRaises(ValueError):
                comparer.compare(value, 0.5, ">=")

    def test_safe_template_supports_only_explicit_bounded_placeholders(self):
        template = UPL.AngelosPromptTemplate()
        self.assertEqual(template.render("{text} / {value} / {integer} / {enabled} {{literal}}", "NA", 0.2, 5, False),
                         ("NA / 0.2 / 5 / false {literal}",))
        self.assertEqual(template.render("{text}", "{__class__}"), ("{__class__}",))
        for text in ("{text.__class__}", "{text[0]}", "{value:.2f}", "{text!r}", "{unknown}", "{0}", "{}", "{"):
            with self.subTest(template=text), self.assertRaises(ValueError):
                template.render(text, "NA")
        with self.assertRaises(ValueError):
            template.render("{text}{text}", "x" * 3000)
        with self.assertRaises(ValueError):
            template.render("{enabled}", enabled="false")

    def test_parse_fact_choose_campaign_chain_changes_actual_prompt_string(self):
        prompts = []
        for energy in (0.2, 0.9):
            document = self.wire_upl()
            document["facts"] = {**document["facts"], "energy": energy}
            document["upl"] = [0.4, 0, 0.25, energy]
            audience = UPL.AngelosParseUPL().parse(json.dumps(document))[0]
            value = UPL.AngelosUPLFactFloat().extract(audience, "energy")[0]
            condition = UPL.AngelosScalarCompare().compare(value, 0.5, ">=")[0]
            selected = UPL.AngelosChoosePrompt().choose(condition, "Bold Aurora", "Quiet Aurora")[0]
            campaign, actual_string = NODES.AngelosCampaign().build(audience, "aurora", "v1", selected)
            self.assertEqual(campaign["referencePrompt"], actual_string)
            result = self.export_fixture(campaign=campaign)
            self.assertEqual(result["manifest"]["campaign"]["referencePrompt"], actual_string)
            self.assertEqual(result["manifest"]["format"], "angelos.comfyui-teacher-pack.v1")
            prompts.append(actual_string)
        self.assertEqual(prompts, ["Quiet Aurora", "Bold Aurora"])
        for value in (0, 1, "false", None):
            with self.assertRaises(ValueError):
                UPL.AngelosChoosePrompt().choose(value, "Bold", "Quiet")


if __name__ == "__main__":
    unittest.main()
