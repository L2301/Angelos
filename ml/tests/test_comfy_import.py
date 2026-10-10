"""Local exporter compatibility and adversarial pack tests; no generation claims."""

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from PIL import Image
import torch

from angelos_ml.comfy import import_teacher_pack
from angelos_ml import comfy


ROOT = Path(__file__).resolve().parents[2]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


class ComfyImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        specification = importlib.util.spec_from_file_location("angelos_export_contract", ROOT / "integrations/comfyui_angelos/nodes.py")
        cls.exporter = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(cls.exporter)
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="angelos-comfy-import-")
        self.directory = Path(self.temporary.name).resolve()
        self.pack = self.export()
        self.output = self.directory / "imported"

    def tearDown(self):
        for directory, subdirectories, filenames in os.walk(self.directory, followlinks=False):
            if not Path(directory).is_symlink():
                os.chmod(directory, 0o700)
        self.temporary.cleanup()

    def export(self, *, fixture=True, width=48, height=32, batch=1, ui=True, audience=None):
        audience = self.exporter.make_audience() if audience is None else audience
        campaign = self.exporter.make_campaign(audience, reference_prompt="Fictional café campaign")
        prompt = {
            "checkpoint": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "local-only.safetensors"}},
            "sampler": {"class_type": "KSampler", "inputs": {"seed": 2 ** 64 - 1, "noise_seed": ["seed-node", 0]}},
            "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sampler", 0]}},
            "export": {"class_type": "AngelosExportCreative", "inputs": {"images": ["decode", 0]}},
        }
        extra = {"workflow": {"nodes": [], "version": 0.4}} if ui else None
        folder_paths = types.SimpleNamespace(get_output_directory=lambda: str(self.directory))
        with mock.patch.dict(sys.modules, {"folder_paths": folder_paths}):
            result = self.exporter.export_teacher_pack(
                torch.full((batch, height, width, 3), 0.25), campaign, prompt, extra, "export", test_only=fixture)
        return self.directory / Path(result["manifestPath"]).parent

    def manifest(self, pack=None):
        return json.loads(((self.pack if pack is None else pack) / "manifest.json").read_bytes())

    def write_manifest(self, manifest, pack=None):
        ((self.pack if pack is None else pack) / "manifest.json").write_bytes(canonical(manifest))

    def rewrite_workflow(self, payload):
        (self.pack / "workflow.json").write_bytes(payload)
        manifest = self.manifest()
        manifest["provenance"]["workflow"]["sha256"] = digest(payload)
        self.write_manifest(manifest)

    def rewrite_png(self, payload):
        manifest = self.manifest()
        entry = manifest["images"][0]
        (self.pack / entry["file"]).write_bytes(payload)
        entry["sha256"] = digest(payload)
        self.write_manifest(manifest)

    def reject(self, *, pack=None, output=None, allow=True, pattern=None):
        destination = self.output if output is None else output
        context = self.assertRaisesRegex((ValueError, OSError), pattern) if pattern else self.assertRaises((ValueError, OSError))
        with context:
            import_teacher_pack(self.pack if pack is None else pack, destination, allow_test_fixture=allow)
        self.assertFalse(destination.exists())

    def test_actual_exporter_fixture_roundtrip_complete_bytes_readonly_and_blank_reviews(self):
        original = {path.name: path.read_bytes() for path in self.pack.iterdir()}
        receipt = import_teacher_pack(self.pack, self.output, allow_test_fixture=True)
        self.assertEqual(receipt, json.loads((self.output / "import.json").read_bytes()))
        self.assertEqual(receipt["id"], "comfy-" + digest(original["manifest.json"]))
        self.assertEqual(receipt["status"], "complete")
        self.assertTrue(receipt["fixtureOnly"])
        self.assertFalse(receipt["workflowExecuted"])
        self.assertFalse(receipt["downloaded"])
        self.assertTrue(all(value is False for value in receipt["verification"].values()))
        self.assertEqual(set(path.name for path in self.output.iterdir()), set(original) | {"import.json", "review-template.json"})
        for name, payload in original.items():
            self.assertEqual((self.output / name).read_bytes(), payload)
            self.assertEqual((self.pack / name).read_bytes(), payload)
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o555)
        self.assertTrue(all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in self.output.iterdir()))
        template = json.loads((self.output / "review-template.json").read_bytes())
        self.assertEqual(template["status"], "unreviewed")
        self.assertTrue(template["fixtureOnly"])
        record = template["records"][0]
        for field in ("quality", "brand", "source"):
            self.assertIsNone(record[field])
        self.assertNotIn("label", record)
        self.assertNotIn("click", record)
        self.assertEqual(record["upl"], self.manifest()["audience"]["upl"])
        self.assertEqual(record["campaignBindingHash"], receipt["campaignBindingHash"])
        self.assertEqual(record["profileId"], "comfy-audience-" + receipt["audienceSha256"])
        for description in (template["instructions"], " ".join(receipt["limitations"])):
            self.assertIn("not a stable person or ledger identity", description)
            self.assertIn("cannot prove person-disjoint train/holdout across updates", description)

    def test_real_hd_png_bytes_and_dimensions_without_resize_or_generation_claim(self):
        pack = self.export(fixture=False, width=1920, height=1080)
        receipt = import_teacher_pack(pack, self.output)
        self.assertFalse(receipt["fixtureOnly"])
        self.assertEqual(receipt["spatialTransform"], "none")
        entry = receipt["images"][0]
        self.assertEqual((entry["width"], entry["height"]), (1920, 1080))
        self.assertEqual((pack / entry["file"]).read_bytes(), (self.output / entry["file"]).read_bytes())
        with Image.open(self.output / entry["file"]) as image:
            self.assertEqual(image.size, (1920, 1080))
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(digest(image.tobytes()), entry["pixelSha256"])
        self.assertFalse(receipt["verification"]["upstreamNativeGenerationVerified"])

    def test_portrait_hd_no_ui_and_custom_audience_batch(self):
        facts = {"ageBand": "65+", "region": "APAC", "openness": 0.125, "energy": 0.75}
        audience = self.exporter.make_audience("custom", json.dumps(facts))
        pack = self.export(fixture=False, width=1080, height=1920, batch=2, ui=False, audience=audience)
        receipt = import_teacher_pack(pack, self.output)
        self.assertEqual(len(receipt["images"]), 2)
        self.assertEqual(receipt["audienceSha256"], audience["sha256"])
        self.assertFalse((self.output / "workflow-ui.json").exists())

    def test_fixture_requires_explicit_opt_in_even_when_hd_sized(self):
        self.reject(allow=False, pattern="opt-in")
        self.reject(allow=1, pattern="explicit bool")
        hd_fixture = self.export(width=1920, height=1080)
        self.reject(pack=hd_fixture, allow=False, pattern="opt-in")

    def test_reject_incomplete_unknown_or_false_evidence_manifest(self):
        original = self.manifest()
        for field, value in (("format", "unknown"), ("status", "partial"), ("kind", "hd-teacher-pack"),
                             ("sourceKind", "comfyui-local-render"), ("testOnly", 1), ("training", True),
                             ("promotion", True), ("packId", "../escape"), ("createdAt", "2026-01-01")):
            with self.subTest(field=field):
                self.write_manifest({**original, field: value})
                self.reject()
        self.write_manifest({**original, "unexpected": "ignored?"})
        self.reject(pattern="fields")

    def test_manifest_and_workflow_json_are_strict_and_bounded(self):
        valid = (self.pack / "manifest.json").read_bytes()
        invalid = (b'{"format":1,"format":2}', b'{"value":NaN}', b'{"value":1e999}',
                   canonical({"huge": 2 ** 257}), b'{"value":"\\ud800"}',
                   b'[' * 1100 + b'0' + b']' * 1100, valid + b' ' * comfy.MAX_JSON_BYTES)
        for payload in invalid:
            with self.subTest(length=len(payload)):
                (self.pack / "manifest.json").write_bytes(payload)
                self.reject()
        (self.pack / "manifest.json").write_bytes(valid)
        original_workflow = json.loads((self.pack / "workflow.json").read_bytes())
        for extra in ({"items": [0] * 50001}, {"text": "a" * 65537}):
            workflow = {**original_workflow, "extraPngInfo": extra}
            self.rewrite_workflow(canonical(workflow))
            self.reject(pattern="bound")
        nested = None
        for _ in range(33):
            nested = [nested]
        self.rewrite_workflow(canonical({**original_workflow, "extraPngInfo": {"nested": nested}}))
        self.reject(pattern="depth")

    def test_workflow_hash_and_provenance_bindings(self):
        workflow_path = self.pack / "workflow.json"
        workflow_path.write_bytes(workflow_path.read_bytes() + b" ")
        self.reject(pattern="Workflow hash")
        self.rewrite_workflow(workflow_path.read_bytes())
        original = self.manifest()
        changes = {"apiPromptSha256": "0" * 64, "exportNodeId": "other", "imageInput": ["other", 0],
                   "seedFields": [], "checkpointReferences": [], "modelHashesVerified": True,
                   "upstreamNativeGenerationVerified": True, "workflowExecutionVerified": True,
                   "hardwareExecutionVerified": True, "allRandomnessCaptured": True}
        for field, value in changes.items():
            with self.subTest(field=field):
                manifest = copy.deepcopy(original)
                manifest["provenance"][field] = value
                self.write_manifest(manifest)
                self.reject()

    def test_ui_workflow_hash_and_embedded_payload_binding(self):
        path = self.pack / "workflow-ui.json"
        path.write_bytes(b'{"different":true}')
        self.reject(pattern="UI workflow")
        manifest = self.manifest()
        manifest["provenance"]["uiWorkflow"]["sha256"] = digest(path.read_bytes())
        self.write_manifest(manifest)
        self.reject(pattern="UI workflow")

    def test_campaign_and_audience_hash_facts_upl_and_synthetic_binding(self):
        original = self.manifest()
        modifications = [
            lambda manifest: manifest.update({"audience": self.exporter.make_audience("bold")}),
            lambda manifest: manifest["campaign"].update({"bindingHash": "0" * 64}),
            lambda manifest: manifest["campaign"].update({"referencePrompt": "changed"}),
            lambda manifest: manifest["campaign"]["audience"].update({"synthetic": False}),
            lambda manifest: manifest["campaign"]["audience"].update({"adult": False}),
            lambda manifest: manifest["campaign"]["audience"].update({"upl": [0, 0, 0, 0]}),
            lambda manifest: manifest["campaign"]["audience"]["facts"].update({"ageBand": "under-18"}),
            lambda manifest: manifest["campaign"]["audience"]["facts"].update({"energy": 1.1}),
            lambda manifest: manifest["campaign"]["audience"]["encoder"].update({"id": "other"}),
        ]
        for modification in modifications:
            manifest = copy.deepcopy(original)
            modification(manifest)
            self.write_manifest(manifest)
            self.reject()

    def test_v1_rejects_opaque_encoder_even_with_recomputed_binding_hashes(self):
        manifest = self.manifest()
        audience = manifest["campaign"]["audience"]
        audience["encoder"] = {"id": "opaque-test-v1", "dimension": 4}
        audience["sha256"] = digest(canonical({key: value for key, value in audience.items() if key != "sha256"}))
        manifest["audience"] = copy.deepcopy(audience)
        campaign = manifest["campaign"]
        campaign["bindingHash"] = digest(canonical({key: value for key, value in campaign.items() if key != "bindingHash"}))
        self.write_manifest(manifest)
        self.reject(pattern="opaque encoders need a new pack version")

    def test_dimensions_contract_indices_and_total_pixel_bounds(self):
        original = self.manifest()
        changes = [
            lambda manifest: manifest["images"][0].update({"width": 49}),
            lambda manifest: manifest["images"][0].update({"height": 4097}),
            lambda manifest: manifest["images"][0].update({"height": True}),
            lambda manifest: manifest["images"][0].update({"index": 1}),
            lambda manifest: manifest["imageContract"].update({"hdSizeGatePassed": True}),
            lambda manifest: manifest["imageContract"].update({"spatialTransform": "upscaled"}),
            lambda manifest: manifest["imageContract"].update({"pngMode": "RGBA"}),
            lambda manifest: manifest["imageContract"].update({"pngBitDepth": 16}),
            lambda manifest: manifest.update({"images": manifest["images"] * 9}),
        ]
        for modification in changes:
            manifest = copy.deepcopy(original)
            modification(manifest)
            self.write_manifest(manifest)
            self.reject()
        self.write_manifest(original)
        with mock.patch.object(comfy, "MAX_PIXELS", 1):
            self.reject(pattern="pixel bound")
        with mock.patch.object(comfy, "MAX_PNG_BYTES", 10):
            self.reject(pattern="byte bound")

    def test_hd_manifest_cannot_pass_using_legacy_pixels(self):
        manifest = self.manifest()
        manifest.update({"testOnly": False, "kind": "hd-teacher-pack", "sourceKind": "comfyui-local-render"})
        old_name = manifest["images"][0]["file"]
        manifest["images"][0]["file"] = "image-000.png"
        (self.pack / old_name).rename(self.pack / "image-000.png")
        self.write_manifest(manifest)
        self.reject(allow=False, pattern="HD size")
        manifest["images"][0].update({"width": 1920, "height": 1080})
        manifest["imageContract"]["hdSizeGatePassed"] = True
        self.write_manifest(manifest)
        self.reject(allow=False, pattern="PNG dimensions")

    def test_png_hash_pixel_hash_crc_truncation_and_trailing_bytes(self):
        entry = self.manifest()["images"][0]
        original = (self.pack / entry["file"]).read_bytes()
        (self.pack / entry["file"]).write_bytes(original[:-4])
        self.reject(pattern="file hash")
        for payload in (original[:-4], original + b"trailing", b"not a png", original[:29] + b"xxxx" + original[33:]):
            self.rewrite_png(payload)
            self.reject()
        self.rewrite_png(original)
        manifest = self.manifest()
        manifest["images"][0]["pixelSha256"] = "0" * 64
        self.write_manifest(manifest)
        self.reject(pattern="pixel hash")

    def test_only_native_rgb8_png_not_jpeg_palette_rgba_or_apng(self):
        for mode, format_name in (("RGB", "JPEG"), ("RGBA", "PNG"), ("P", "PNG"), ("I;16", "PNG")):
            with self.subTest(mode=mode, format=format_name):
                buffer = io.BytesIO()
                Image.new(mode, (48, 32)).save(buffer, format=format_name)
                self.rewrite_png(buffer.getvalue())
                self.reject()
        buffer = io.BytesIO()
        Image.new("RGB", (48, 32), "red").save(buffer, format="PNG", save_all=True,
                                               append_images=[Image.new("RGB", (48, 32), "blue")])
        self.rewrite_png(buffer.getvalue())
        self.reject(pattern="chunks")

    def test_missing_extra_files_and_directories_are_not_complete_packs(self):
        path = self.pack / "workflow-ui.json"
        original = path.read_bytes()
        path.unlink()
        self.reject()
        path.write_bytes(original)
        for extra in ("unexpected.txt", ".hidden"):
            (self.pack / extra).write_text("not imported")
            self.reject(pattern="unexpected")
            (self.pack / extra).unlink()
        (self.pack / "extra-directory").mkdir()
        self.reject(pattern="unexpected")

    def test_descriptor_traversal_absolute_and_duplicate_image_names(self):
        original = self.manifest()
        for filename in ("../outside.png", "/etc/passwd", "sub/image.png", "..\\outside.png", "image-000.png\0"):
            manifest = copy.deepcopy(original)
            manifest["images"][0]["file"] = filename
            self.write_manifest(manifest)
            self.reject()
        for descriptor in ("workflow", "uiWorkflow"):
            manifest = copy.deepcopy(original)
            manifest["provenance"][descriptor]["file"] = "../outside.json"
            self.write_manifest(manifest)
            self.reject()
        manifest = copy.deepcopy(original)
        manifest["images"].append(copy.deepcopy(manifest["images"][0]))
        self.write_manifest(manifest)
        self.reject(pattern="indices")

    def test_no_symlink_components_source_entries_or_destination(self):
        alias = self.directory / "alias"
        alias.symlink_to(self.pack, target_is_directory=True)
        self.reject(pack=alias)
        ancestor = self.directory / "ancestor"
        ancestor.symlink_to(self.pack.parent, target_is_directory=True)
        self.reject(pack=ancestor / self.pack.name)
        for filename in ("manifest.json", "workflow.json", "workflow-ui.json", "TEST_ONLY-image-000.png"):
            path = self.pack / filename
            outside = self.directory / filename
            path.rename(outside)
            path.symlink_to(outside)
            self.reject()
            path.unlink()
            outside.rename(path)
        output_parent = self.directory / "output-link"
        output_parent.symlink_to(self.directory, target_is_directory=True)
        self.reject(output=output_parent / "via-link")
        self.assertFalse((self.directory / "via-link").exists())
        destination = self.directory / "existing-target"
        destination.mkdir()
        self.output.symlink_to(destination, target_is_directory=True)
        with self.assertRaises(FileExistsError):
            import_teacher_pack(self.pack, self.output, allow_test_fixture=True)
        self.assertEqual(list(destination.iterdir()), [])

    def test_reject_hardlinks_fifos_and_nonregular_files(self):
        path = self.pack / "workflow.json"
        original = path.read_bytes()
        outside = self.directory / "workflow-copy.json"
        os.link(path, outside)
        self.reject(pattern="without links")
        outside.unlink()
        path.unlink()
        os.mkfifo(path)
        self.reject(pattern="regular files")
        path.unlink()
        path.mkdir()
        self.reject(pattern="regular files")
        path.rmdir()
        path.write_bytes(original)

    def test_new_output_only_no_nested_output_or_parent_traversal(self):
        self.reject(output=self.pack / "nested", pattern="inside")
        self.reject(pack=self.pack / ".." / self.pack.name, pattern="traversal")
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("existing output")
        with self.assertRaises(FileExistsError):
            import_teacher_pack(self.pack, self.output, allow_test_fixture=True)
        self.assertEqual(sentinel.read_text(), "existing output")
        sentinel.unlink()
        self.output.rmdir()
        receipt = import_teacher_pack(self.pack, self.output, allow_test_fixture=True)
        with self.assertRaises(FileExistsError):
            import_teacher_pack(self.pack, self.output, allow_test_fixture=True)
        self.assertEqual(json.loads((self.output / "import.json").read_bytes()), receipt)

    def test_write_failure_cleans_only_new_output(self):
        original = (self.pack / "manifest.json").read_bytes()
        with mock.patch.object(comfy.os, "fsync", side_effect=OSError("simulated disk failure")):
            self.reject(pattern="disk failure")
        self.assertEqual((self.pack / "manifest.json").read_bytes(), original)

    def test_workflow_commands_and_checkpoint_references_are_never_executed(self):
        workflow = json.loads((self.pack / "workflow.json").read_bytes())
        workflow["apiPrompt"]["untrusted"] = {"class_type": "UnknownCommandNode",
                                             "inputs": {"command": "__import__('os').system('touch DO_NOT_CREATE')"}}
        self.rewrite_workflow(canonical(workflow))
        manifest = self.manifest()
        manifest["provenance"]["apiPromptSha256"] = digest(canonical(workflow["apiPrompt"]))
        self.write_manifest(manifest)
        with mock.patch("os.system", side_effect=AssertionError("Execution forbidden")), \
                mock.patch("subprocess.Popen", side_effect=AssertionError("Execution forbidden")), \
                mock.patch("urllib.request.urlopen", side_effect=AssertionError("Downloads forbidden")):
            receipt = import_teacher_pack(self.pack, self.output, allow_test_fixture=True)
        self.assertFalse(receipt["workflowExecuted"])
        self.assertFalse(receipt["downloaded"])
        self.assertEqual((self.output / "workflow.json").read_bytes(), canonical(workflow))

    def test_cli_requires_opt_in_and_emits_receipt(self):
        command = [sys.executable, "-m", "angelos_ml.comfy", "import", str(self.pack), "--output", str(self.output)]
        environment = {**os.environ, "PYTHONPATH": str(ROOT / "ml")}
        rejected = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=20)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("opt-in", rejected.stderr)
        self.assertFalse(self.output.exists())
        imported = subprocess.run(command + ["--allow-test-fixture"], cwd=ROOT, env=environment,
                                  capture_output=True, text=True, timeout=20)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        self.assertEqual(json.loads(imported.stdout), json.loads((self.output / "import.json").read_bytes()))

    def test_unsupported_platform_fails_clearly_before_path_or_file_access(self):
        for attribute, value in (("name", "nt"), ("O_NOFOLLOW", None), ("O_DIRECTORY", None),
                                 ("supports_dir_fd", set()), ("supports_fd", set())):
            with self.subTest(attribute=attribute), mock.patch.object(comfy.os, attribute, value):
                with self.assertRaisesRegex(ValueError, "macOS/Linux POSIX"):
                    import_teacher_pack(self.pack, self.output, allow_test_fixture=True)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
