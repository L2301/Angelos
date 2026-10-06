import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from angelos_ml.compile import METADATA_FILE, TeacherModule, audit_aten_graph, compile_teacher, main, write_compiled_teacher
from angelos_ml.graph import canonical_json, graph_hash
from angelos_ml.teacher import TensorTeacher, brand_probe_coordinates
from test_ml import ML_ROOT, example_graph, node_named


class CompileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = example_graph()
        cls.payload, cls.report = compile_teacher(cls.graph, graph_hash(cls.graph), samples=64, threads=1)

    def test_loaded_export_is_aten_with_constants_and_no_parameters(self):
        extra_files = {METADATA_FILE: ""}
        program = torch.export.load(io.BytesIO(self.payload), extra_files=extra_files)
        metadata = json.loads(extra_files[METADATA_FILE])
        self.assertEqual(metadata["graphHash"], graph_hash(self.graph))
        self.assertEqual(metadata["lowering"], "torch.export.export(strict=True)")
        self.assertEqual(metadata["fixedBatchSize"], 64)
        self.assertEqual(audit_aten_graph(program), self.report["metadata"]["graph"])
        self.assertTrue(program.constants)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in program.constants.values()))
        self.assertFalse(list(program.parameters()))
        self.assertFalse(metadata["training"])
        self.assertFalse(metadata["cudaTested"])
        self.assertFalse(metadata["interpreterFallback"])
        self.assertEqual(self.report["cpuParity"]["maxError"], 0)
        self.assertEqual(self.report["cpuParity"]["brandMaxError"], 0)
        self.assertGreaterEqual(self.report["cpuParity"]["brandChannelCount"], 63 * 3)

    def test_loaded_module_matches_full_teacher_without_interpreter(self):
        teacher = TensorTeacher(self.graph)
        coordinates = torch.cat((brand_probe_coordinates(teacher.brand), torch.tensor([[0.92, 0.73]])))
        upl = torch.full((64, 4), 0.5)
        expected = teacher(upl, coordinates)
        module = torch.export.load(io.BytesIO(self.payload)).module()
        with patch.object(TensorTeacher, "color_at", side_effect=AssertionError("Interpreter fallback")):
            actual = module(upl, coordinates)
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        with self.assertRaises((AssertionError, RuntimeError)):
            module(upl[:32], coordinates[:32])

    def test_disk_file_reload_and_nonoverwrite(self):
        with tempfile.TemporaryDirectory(dir=ML_ROOT) as directory:
            output = Path(directory) / "teacher.pt2"
            artifact_path, metrics_path = write_compiled_teacher(output, self.payload, self.report)
            self.assertEqual(hashlib.sha256(artifact_path.read_bytes()).hexdigest(), self.report["artifactSha256"])
            saved_report = json.loads(metrics_path.read_text())
            self.assertEqual(saved_report["cpuParity"]["maxError"], 0)
            module = torch.export.load(artifact_path).module()
            coordinates = torch.full((64, 2), 0.8)
            upl = torch.full((64, 4), 0.5)
            torch.testing.assert_close(module(upl, coordinates), TensorTeacher(self.graph)(upl, coordinates))
            with self.assertRaises(ValueError):
                write_compiled_teacher(output, self.payload, self.report)

    def test_source_changes_propagate_into_compiled_output(self):
        changed = copy.deepcopy(self.graph)
        node_named(changed, "field")["params"]["high"] = [0.85, 0.1, 0.2]
        changed_payload, changed_report = compile_teacher(changed, graph_hash(changed), samples=64, threads=1)
        self.assertNotEqual(self.report["graphHash"], changed_report["graphHash"])
        upl = torch.full((64, 4), 0.5)
        coordinates = torch.full((64, 2), 0.9)
        original = torch.export.load(io.BytesIO(self.payload)).module()(upl, coordinates)
        modified = torch.export.load(io.BytesIO(changed_payload)).module()(upl, coordinates)
        self.assertFalse(torch.allclose(original, modified))

    def test_invalid_ir_hash_and_export_failure_do_not_fallback(self):
        unsupported = copy.deepcopy(self.graph)
        node_named(unsupported, "field")["op"] = "remote"
        with patch("angelos_ml.compile.torch.export.export") as exporter:
            with self.assertRaises(ValueError):
                compile_teacher(unsupported, graph_hash(self.graph))
            with self.assertRaises(ValueError):
                compile_teacher(self.graph, "0" * 64)
            exporter.assert_not_called()
        with patch("angelos_ml.compile.torch.export.export", side_effect=RuntimeError("Unsupported tracing operation")):
            with self.assertRaisesRegex(RuntimeError, "Unsupported tracing"):
                compile_teacher(self.graph, graph_hash(self.graph))

    def test_non_aten_export_is_rejected(self):
        graph = torch.fx.Graph()
        upl = graph.placeholder("upl")
        output = graph.call_method("relu", (upl,))
        graph.output(output)
        class InvalidProgram:
            pass
        invalid = InvalidProgram()
        invalid.graph = graph
        with self.assertRaisesRegex(ValueError, "non-ATen"):
            audit_aten_graph(invalid)

    def test_cli_supports_canonical_sidecar_and_fixed_batch(self):
        graph = example_graph(include_brand=False)
        graph["nodes"].append({"id": "fixed", "op": "constant", "inputs": {}, "params": {"value": 0.8}})
        node_named(graph, "light")["inputs"]["condition"] = "fixed"
        normalized = copy.deepcopy(graph)
        normalized["nodes"] = sorted(normalized["nodes"], key=lambda node: node["id"])
        with tempfile.TemporaryDirectory(dir=ML_ROOT) as directory:
            source = Path(directory) / "graph.json"
            canonical = Path(directory) / "canonical.json"
            output = Path(directory) / "teacher.pt2"
            source.write_text(json.dumps(graph), encoding="utf-8")
            canonical.write_text(canonical_json(normalized), encoding="utf-8")
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                main(["--graph", str(source), "--graph-hash", graph_hash(graph), "--graph-canonical", str(canonical), "--output", str(output), "--samples", "8"])
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["samples"], 8)
            self.assertIsNone(summary["brandMaxError"])
            self.assertFalse(summary["cudaTested"])
            report = json.loads(output.with_suffix(".compile.metrics.json").read_text())
            self.assertEqual(report["metadata"]["graphHashVerification"], "compiler-canonical-file")
            self.assertEqual(torch.export.load(output).module()(torch.zeros(8, 4), torch.zeros(8, 2)).shape, (8, 3))

    def test_module_uses_source_teacher_and_validates_sample_limits(self):
        module = TeacherModule(self.graph)
        self.assertEqual(sum(parameter.numel() for parameter in module.parameters()), 0)
        for samples in (0, 4097, True):
            with self.assertRaises(ValueError):
                compile_teacher(self.graph, graph_hash(self.graph), samples=samples)

    def test_cli_export_failure_writes_no_artifact(self):
        with tempfile.TemporaryDirectory(dir=ML_ROOT) as directory:
            source = Path(directory) / "graph.json"
            output = Path(directory) / "teacher.pt2"
            source.write_text(json.dumps(self.graph), encoding="utf-8")
            with patch("angelos_ml.compile.torch.export.export", side_effect=RuntimeError("Unsupported tracing operation")):
                with patch("sys.stderr", new_callable=io.StringIO) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        main(["--graph", str(source), "--graph-hash", graph_hash(self.graph), "--output", str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("No interpreter fallback", stderr.getvalue())
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".compile.metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
