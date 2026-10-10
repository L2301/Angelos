"""Synthetic embedding transport tests, never Qwen or ComfyUI inference evidence."""

import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from angelos_ml import comfy


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "integrations" / "comfyui_angelos"
SPEC = importlib.util.spec_from_file_location("angelos_comfy_v2_test", PACKAGE / "__init__.py",
                                            submodule_search_locations=[str(PACKAGE)])
BRIDGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BRIDGE
SPEC.loader.exec_module(BRIDGE)
NODES = sys.modules[f"{SPEC.name}.nodes"]
UPL = sys.modules[f"{SPEC.name}.upl"]
V2 = sys.modules[f"{SPEC.name}.upl_v2"]


def wire(dimension=128):
    vector = [math.sin(index + 1) for index in range(dimension)]
    norm = math.sqrt(math.fsum(value * value for value in vector))
    return {
        "format": "angelos.upl-input.v2", "synthetic": True, "adult": True,
        "factsSchema": "angelos.facts.v2",
        "facts": {"ageBand": "35-44", "region": "NA", "openness": 0.25, "energy": 0.2,
                  "interests": ["hiking", "fictional Aurora"], "goals": ["explore"],
                  "preferredStyles": ["quiet"], "preferredLanguage": "en"},
        "encoder": {"id": "qwen3-embedding-0.6b-v1", "dimension": dimension,
                    "modelId": "Qwen/Qwen3-Embedding-0.6B", "revision": V2.QWEN_REVISION,
                    "pooling": "last-token", "normalization": "l2", "templateVersion": "angelos.upl-text.v1"},
        "upl": [value / norm for value in vector],
        "context": {"campaignId": "aurora", "factRevision": 1, "eventRevision": 2,
                    "consentVersion": 1, "asOf": 1791417600000, "ingredientsHash": "a" * 64},
        "recentIntent": [{"kind": "save", "topic": "hiking", "occurredAt": 1791417500000,
                          "expiresAt": 1791427600000}],
    }


class ComfyUPLV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name).resolve()
        self.folder_paths = ModuleType("folder_paths")
        self.folder_paths.get_output_directory = lambda: str(self.directory)

    def tearDown(self):
        for directory, subdirectories, filenames in os.walk(self.directory, followlinks=False):
            if not Path(directory).is_symlink():
                os.chmod(directory, 0o700)
        self.temporary.cleanup()

    def parse(self, document):
        return UPL.parse_upl(json.dumps(document, ensure_ascii=False))

    def export(self, document):
        audience = self.parse(document)
        campaign = NODES.make_campaign(audience, document["context"]["campaignId"], "v2", "Fictional Aurora hiking")
        prompt = {
            "parse": {"class_type": "AngelosParseUPL", "inputs": {"upl_json": json.dumps(document)}},
            "decode": {"class_type": "VAEDecode", "inputs": {}},
            "export": {"class_type": "AngelosExportCreative", "inputs": {"images": ["decode", 0]}},
        }
        with patch.dict(sys.modules, {"folder_paths": self.folder_paths}):
            exported = NODES.export_teacher_pack(torch.zeros((1, 3, 5, 3)), campaign, prompt, None, "export", test_only=True)
        return self.directory / exported["manifestPath"], exported["manifest"]

    def resign(self, manifest):
        audience = manifest["audience"]
        audience["sha256"] = hashlib.sha256(V2.canonical({key: value for key, value in audience.items() if key != "sha256"})).hexdigest()
        manifest["campaign"]["audience"] = copy.deepcopy(audience)
        campaign = manifest["campaign"]
        campaign["bindingHash"] = hashlib.sha256(V2.canonical({key: value for key, value in campaign.items() if key != "bindingHash"})).hexdigest()

    def test_supported_signed_dimensions_preserved_without_normalization_or_downcast(self):
        self.assertEqual(V2.QWEN_REVISION, "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
        for dimension in (128, 256, 512, 1024):
            document = wire(dimension)
            original = copy.deepcopy(document)
            audience = self.parse(document)
            self.assertEqual(document, original)
            self.assertEqual(audience["upl"], original["upl"])
            self.assertTrue(any(value < 0 for value in audience["upl"]))
            self.assertEqual(audience["context"], original["context"])
            self.assertEqual(audience["recentIntent"], original["recentIntent"])
            self.assertEqual(NODES.validate_audience(audience), audience)
            self.assertEqual(comfy._v2_contract().validate_audience(audience), audience)
        self.assertGreater(len(json.dumps(wire(1024))), 16384)

    def test_encoder_and_bundle_versions_all_fields_fail_closed(self):
        original = wire()
        for key in original:
            document = copy.deepcopy(original)
            del document[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.parse(document)
        for key in original["encoder"]:
            document = copy.deepcopy(original)
            document["encoder"][key] = "unknown"
            with self.subTest(encoder=key), self.assertRaises(ValueError):
                self.parse(document)
        for key, value in (("format", "angelos.upl-input.v3"), ("factsSchema", "angelos.facts.v1"),
                           ("synthetic", False), ("adult", 1), ("sha256", "a" * 64)):
            document = copy.deepcopy(original)
            document[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.parse(document)
        for revision in ("main", "a" * 40, V2.QWEN_REVISION.upper()):
            document = wire()
            document["encoder"]["revision"] = revision
            with self.assertRaises(ValueError):
                self.parse(document)

    def test_vector_bounds_dimensions_and_normalization_fail_closed(self):
        for dimension in (4, 127, 129, 2048, 128.0, True):
            document = wire()
            document["encoder"]["dimension"] = dimension
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                self.parse(document)
        for vector in ([0] * 128, [0.5] * 128, [1] * 127, [0] * 129):
            document = wire()
            document["upl"] = vector
            with self.assertRaises(ValueError):
                self.parse(document)
        for value in (float("nan"), float("inf"), -float("inf"), True, "0", 1.1, -1.1, 10 ** 1000):
            document = wire()
            document["upl"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.parse(document)
        document = wire()
        document["upl"] = [0.99995] + [0] * 127
        self.assertEqual(self.parse(document)["upl"], document["upl"])
        document["upl"][0] = 0.9998
        with self.assertRaises(ValueError):
            self.parse(document)

    def test_richer_facts_are_explicit_optional_and_bounded(self):
        document = wire()
        document["facts"] = {"ageBand": "18-24"}
        audience = self.parse(document)
        for extractor, field in ((UPL.AngelosUPLFactString(), "region"), (UPL.AngelosUPLFactFloat(), "energy")):
            with self.assertRaisesRegex(ValueError, "absent"):
                extractor.extract(audience, field)
        audience = self.parse(wire())
        self.assertEqual(UPL.AngelosUPLFactString().extract(audience, "interests"), ("hiking, fictional Aurora",))
        self.assertEqual(UPL.AngelosUPLFactString().extract(audience, "preferredLanguage"), ("en",))
        self.assertEqual(UPL.AngelosUPLFactFloat().extract(audience, "energy"), (0.2,))
        for key, value in (("ageBand", "under-18"), ("region", "private"), ("email", "x@example.test"),
                           ("energy", True), ("openness", 1.1), ("interests", "hiking"),
                           ("interests", ["hiking"] * 2), ("goals", [str(index) for index in range(17)]),
                           ("preferredStyles", ["x" * 81]), ("preferredLanguage", " en"),
                           ("preferredLanguage", "en\n"), ("preferredLanguage", "en\x7f"), ("ageBand", None)):
            document = wire()
            document["facts"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.parse(document)

    def test_nullable_optional_facts_survive_without_typed_defaults(self):
        document = wire()
        document["facts"] = {field: None for field in V2.FACT_FIELDS}
        document["facts"]["ageBand"] = "35-44"
        audience = self.parse(document)
        self.assertEqual(audience["facts"], document["facts"])
        for field in V2.FACT_FIELDS - {"ageBand"}:
            extractor = UPL.AngelosUPLFactFloat() if field in ("energy", "openness") else UPL.AngelosUPLFactString()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "absent"):
                extractor.extract(audience, field)
        path, manifest = self.export(document)
        destination = self.directory / "null-facts"
        comfy.import_teacher_pack(path.parent, destination, allow_test_fixture=True)
        self.assertEqual(json.loads((destination / "manifest.json").read_bytes())["audience"], manifest["audience"])

    def test_utf16_bounds_match_js_for_bmp_and_astral_text(self):
        for text in ("界" * 80, "🎨" * 40):
            document = wire()
            document["facts"]["interests"] = [text]
            document["facts"]["preferredLanguage"] = text
            document["recentIntent"][0]["topic"] = text
            self.assertEqual(self.parse(document)["facts"]["interests"], [text])
        for text in ("界" * 81, "🎨" * 41, "\ud800"):
            document = wire()
            document["facts"]["preferredLanguage"] = text
            with self.assertRaises(ValueError):
                self.parse(document)

    @unittest.skipUnless(shutil.which("node"), "Node is needed only for cross-language validation")
    def test_actual_js_validator_copy_json_is_accepted_without_rehashing_ingredients(self):
        script = (
            "import {readFileSync} from 'node:fs'; "
            "import {validateSemanticBundle} from './src/core/upl.js'; "
            "process.stdout.write(JSON.stringify(validateSemanticBundle(JSON.parse(readFileSync(0,'utf8')))));"
        )
        document = wire(1024)
        document["facts"]["energy"] = None
        document["facts"]["interests"] = ["🎨" * 40]
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script], cwd=ROOT,
                                input=json.dumps(document), capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        audience = UPL.parse_upl(result.stdout)
        self.assertEqual(audience["context"]["ingredientsHash"], document["context"]["ingredientsHash"])
        self.assertEqual(audience["upl"], document["upl"])
        self.assertEqual(audience["facts"], document["facts"])

    def test_encoder_producer_fixture_metadata_and_output_parse_without_model_loading(self):
        from angelos_ml import upl as producer
        document = wire(256)
        document["facts"]["goals"] = None
        context = document["context"]
        ingredients = {
            "format": "angelos.upl-ingredients.v2", "synthetic": True, "adult": True,
            "factsSchema": document["factsSchema"], "facts": document["facts"],
            "provenance": [{"field": field, "source": "synthetic-fixture", "occurredAt": context["asOf"],
                            "receivedAt": context["asOf"]} for field in document["facts"]],
            "eventsSchema": "angelos.events.v1", "events": [],
            **{key: value for key, value in context.items() if key != "ingredientsHash"},
        }
        encoder = SimpleNamespace(encode=lambda text, dimension: document["upl"])
        with patch.object(producer, "load_local_encoder", return_value=encoder):
            produced = producer.encode_request({"ingredients": ingredients, "dimension": 256})
        audience = self.parse(produced)
        self.assertEqual(audience["encoder"]["revision"], V2.QWEN_REVISION)
        self.assertEqual(audience["facts"]["goals"], None)
        self.assertEqual(audience["upl"], document["upl"])

    def test_context_revisions_timestamps_and_recent_intent_are_bound(self):
        for key, value in (("factRevision", True), ("eventRevision", -1), ("consentVersion", 0),
                           ("asOf", "2026-10-08"), ("asOf", 9007199254740992),
                           ("ingredientsHash", "z" * 64), ("campaignId", "../escape")):
            document = wire()
            document["context"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.parse(document)
        for key, value in (("kind", "purchase"), ("topic", ""), ("topic", "x" * 81),
                           ("occurredAt", 1791417600001), ("occurredAt", -1),
                           ("expiresAt", 1791417600000), ("expiresAt", True), ("extra", 1)):
            document = wire()
            document["recentIntent"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.parse(document)
        document = wire()
        document["recentIntent"] *= 65
        with self.assertRaises(ValueError):
            self.parse(document)
        document["recentIntent"] = []
        self.assertEqual(UPL.AngelosUPLRecentIntentText().extract(self.parse(document)), ("",))

    def test_opaque_nodes_do_not_emit_clip_conditioning_or_infer_facts(self):
        audience = self.parse(wire(1024))
        node = UPL.AngelosUPLEmbeddingBlock()
        self.assertEqual(node.RETURN_TYPES, ("ANGELOS_EMBEDDING",))
        bundle = node.extract(audience)[0]
        self.assertEqual(bundle, audience)
        self.assertEqual(UPL.AngelosUPLRecentIntentText().extract(audience), ("save: hiking",))
        self.assertEqual(UPL.AngelosUPLVectorFloat().extract(audience, 1023), (audience["upl"][1023],))
        for index in (-1, True, 1024):
            with self.assertRaises(ValueError):
                UPL.AngelosUPLVectorFloat().extract(audience, index)
        for extractor in (node, UPL.AngelosUPLRecentIntentText()):
            with self.assertRaises(ValueError):
                extractor.extract(NODES.make_audience())
        bundle["upl"][0] = 0
        self.assertNotEqual(bundle, audience)

    def test_full_bundle_hash_covers_context_intent_encoder_facts_and_vector(self):
        audience = self.parse(wire())
        for field in ("context", "recentIntent", "encoder", "facts", "upl"):
            changed = copy.deepcopy(audience)
            if field == "context":
                changed[field]["ingredientsHash"] = "b" * 64
            elif field == "recentIntent":
                changed[field][0]["topic"] = "new topic"
            elif field == "encoder":
                changed[field]["revision"] = "b" * 40
            elif field == "facts":
                changed[field]["interests"] = ["new explicit fact"]
            else:
                changed[field][0] = -changed[field][0]
            for validator in (NODES.validate_audience, comfy._v2_contract().validate_audience):
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validator(changed)

    def test_v2_export_import_preserves_full_bundle_and_native_bytes(self):
        for dimension in (128, 256, 512, 1024):
            path, manifest = self.export(wire(dimension))
            self.assertEqual(manifest["format"], "angelos.comfyui-teacher-pack.v2")
            self.assertEqual(manifest["campaign"]["format"], "angelos.comfyui-campaign.v2")
            destination = self.directory / f"imported-{dimension}"
            receipt = comfy.import_teacher_pack(path.parent, destination, allow_test_fixture=True)
            self.assertEqual(receipt["format"], "angelos.comfyui-import.v2")
            self.assertFalse(receipt["embeddingExecutionVerified"])
            self.assertFalse(receipt["ingredientsHashVerified"])
            self.assertFalse(receipt["workflowExecuted"])
            reviews = json.loads((destination / "review-template.json").read_bytes())
            self.assertEqual(reviews["format"], "angelos.review-template.v2")
            self.assertEqual(reviews["audience"], manifest["audience"])
            self.assertEqual(reviews["records"][0]["upl"], wire(dimension)["upl"])
            self.assertIsNone(reviews["records"][0]["quality"])
            for source in path.parent.iterdir():
                self.assertEqual(source.read_bytes(), (destination / source.name).read_bytes())

    def test_mixed_pack_versions_and_resigned_invalid_bundles_fail_import(self):
        path, original = self.export(wire())
        for field, value in (("format", comfy.PACK_FORMAT),):
            manifest = copy.deepcopy(original)
            manifest[field] = value
            path.write_bytes(V2.canonical(manifest))
            with self.assertRaises(ValueError):
                comfy.import_teacher_pack(path.parent, self.directory / "rejected", allow_test_fixture=True)
        for change in ("campaign", "norm", "revision", "expired", "hash", "dimension"):
            manifest = copy.deepcopy(original)
            audience = manifest["audience"]
            if change == "campaign":
                manifest["campaign"]["id"] = "other"
            elif change == "norm":
                audience["upl"] = [0] * 128
            elif change == "revision":
                audience["encoder"]["revision"] = "a" * 40
            elif change == "expired":
                audience["recentIntent"][0]["expiresAt"] = audience["context"]["asOf"]
            elif change == "dimension":
                audience["encoder"]["dimension"] = 4
            self.resign(manifest)
            if change == "hash":
                manifest["audience"]["context"]["factRevision"] += 1
            path.write_bytes(V2.canonical(manifest))
            destination = self.directory / "rejected"
            with self.subTest(change=change), self.assertRaises(ValueError):
                comfy.import_teacher_pack(path.parent, destination, allow_test_fixture=True)
            self.assertFalse(destination.exists())

    def test_campaign_mismatch_rejected_and_identifier_slug_is_safe(self):
        audience = self.parse(wire())
        with self.assertRaisesRegex(ValueError, "campaignId"):
            NODES.make_campaign(audience, "other")
        document = wire()
        document["context"]["campaignId"] = "2026.Aurora:campaign"
        path, manifest = self.export(document)
        self.assertTrue(path.parent.name.startswith("TEST_ONLY-2026-aurora-campaign-"))
        self.assertEqual(manifest["campaign"]["id"], document["context"]["campaignId"])
        comfy.import_teacher_pack(path.parent, self.directory / "safe-slug", allow_test_fixture=True)

    def test_json_bounds_duplicates_and_unhashed_input(self):
        document = wire()
        serialized = json.dumps(document)
        for invalid in (serialized.replace('"adult": true', '"adult": true, "adult": true'),
                        " " * 65537, "[" * 1100 + "]" * 1100):
            with self.assertRaises(ValueError):
                UPL.parse_upl(invalid)
        document["sha256"] = "a" * 64
        with self.assertRaises(ValueError):
            self.parse(document)

    def test_importer_shared_contract_load_needs_no_torch_or_comfyui(self):
        script = "from angelos_ml import comfy; import sys; comfy._v2_contract(); assert 'torch' not in sys.modules; assert 'folder_paths' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "ml"), "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
