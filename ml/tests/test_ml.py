import copy
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from angelos_ml.data import profile_splits, sample_batch
from angelos_ml.graph import canonical_json, graph_hash, javascript_number, load_graph, validate_graph, verify_graph_hash
from angelos_ml.model import Student
from angelos_ml.teacher import GLYPH_ROWS, TensorTeacher, brand_masks, brand_probe_coordinates, pixel_coordinates
from angelos_ml.train import train_student, write_artifacts


ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = ROOT / "ml"


def example_graph(include_brand=True):
    nodes = [
        {"id": "person", "op": "upl", "inputs": {}, "params": {}},
        {"id": "energy", "op": "feature", "inputs": {"input": "person"}, "params": {"index": 3}},
        {"id": "field", "op": "gradient", "inputs": {"condition": "energy"}, "params": {"low": [0.02, 0.08, 0.12], "high": [0.2, 0.65, 0.55], "angle": 0.4}},
        {"id": "disk", "op": "disk", "inputs": {"image": "field", "condition": "energy"}, "params": {"color": [0.9, 0.7, 0.3], "center": [0.65, 0.4], "radius": 0.2, "softness": 0.1}},
        {"id": "shade", "op": "vignette", "inputs": {"image": "disk"}, "params": {"strength": 0.3}},
        {"id": "light", "op": "exposure", "inputs": {"image": "shade", "condition": "energy"}, "params": {"amount": 0.15}},
    ]
    if include_brand:
        nodes.append({"id": "mark", "op": "brand", "inputs": {"image": "light"}, "params": {"x": 0.0625, "y": 0.0625, "cell": 0.015625, "color": [1, 1, 1], "background": [0, 0, 0]}})
    return {
        "version": "angelos.graph.v1",
        "id": "test_campaign",
        "encoder": {"id": "structured-demo-v1", "dimension": 4},
        "nodes": nodes,
        "output": "mark" if include_brand else "light",
    }


def node_named(graph, identifier):
    return next(node for node in graph["nodes"] if node["id"] == identifier)


class GraphTests(unittest.TestCase):
    def test_reserved_graph_and_node_identifiers_rejected(self):
        for identifier in ("__proto__", "prototype", "constructor", "campaign", "output"):
            for location in ("campaign", "source", "terminal"):
                with self.subTest(identifier=identifier, location=location):
                    graph = example_graph()
                    if location == "campaign":
                        graph["id"] = identifier
                        message = "Invalid campaign ID"
                    elif location == "source":
                        node_named(graph, "person")["id"] = identifier
                        node_named(graph, "energy")["inputs"]["input"] = identifier
                        message = "Invalid node ID"
                    else:
                        node_named(graph, "mark")["id"] = identifier
                        graph["output"] = identifier
                        message = "Invalid node ID"
                    with self.assertRaisesRegex(ValueError, message):
                        validate_graph(graph)
                    with self.assertRaisesRegex(ValueError, message):
                        graph_hash(graph)

    def test_reserved_name_prefixes_and_case_variants_remain_valid(self):
        for identifier in ("campaign_demo", "output_image", "Campaign", "Output"):
            with self.subTest(identifier=identifier):
                graph = example_graph()
                graph["id"] = identifier
                node_named(graph, "mark")["id"] = identifier
                graph["output"] = identifier
                self.assertEqual(validate_graph(graph)[-1]["id"], identifier)

    def test_forward_references_and_source_order_independent_hash(self):
        graph = example_graph()
        expected = graph_hash(graph)
        graph["nodes"].reverse()
        self.assertEqual(graph_hash(graph), expected)
        ordered = validate_graph(graph)
        observed = set()
        for node in ordered:
            self.assertTrue(set(node["inputs"].values()).issubset(observed))
            observed.add(node["id"])

    def test_unknown_parameters_operations_and_fields_rejected(self):
        for mutation in ("params", "operation", "field", "input"):
            with self.subTest(mutation=mutation):
                graph = example_graph()
                target = node_named(graph, "field")
                if mutation == "params":
                    target["params"]["url"] = "https://example.invalid/model"
                elif mutation == "operation":
                    target["op"] = "remote_model"
                elif mutation == "field":
                    target["position"] = [0, 0]
                else:
                    target["inputs"]["extra"] = "energy"
                with self.assertRaises(ValueError):
                    validate_graph(graph)

    def test_cycle_orphan_type_and_multiple_sources_rejected(self):
        for mutation in ("cycle", "orphan", "type", "source", "number", "plate"):
            with self.subTest(mutation=mutation):
                graph = example_graph()
                if mutation == "cycle":
                    node_named(graph, "shade")["inputs"]["image"] = "light"
                elif mutation == "orphan":
                    graph["nodes"].append({"id": "unused", "op": "constant", "inputs": {}, "params": {"value": 0}})
                elif mutation == "type":
                    node_named(graph, "field")["inputs"]["condition"] = "person"
                elif mutation == "source":
                    graph["nodes"].append({"id": "other", "op": "upl", "inputs": {}, "params": {}})
                elif mutation == "number":
                    node_named(graph, "field")["params"]["angle"] = float("nan")
                else:
                    node_named(graph, "mark")["params"]["x"] = 1
                with self.assertRaises(ValueError):
                    validate_graph(graph)

    def test_normal_number_serialization(self):
        cases = ((1.0, "1"), (-0.0, "0"), (1e-6, "0.000001"), (1e-7, "1e-7"), (1e20, "100000000000000000000"), (1e21, "1e+21"))
        for value, expected in cases:
            self.assertEqual(javascript_number(value), expected)

    @unittest.skipUnless(shutil.which("node") and (ROOT / "src/core/graph.js").exists(), "Node compiler is not available")
    def test_hash_matches_actual_js_compiler(self):
        graph = example_graph()
        node_named(graph, "field")["params"]["low"] = [-0.0, 1e-6, 1e-7]
        graph["nodes"].reverse()
        script = "import { graphHash } from './src/core/graph.js'; let source = ''; for await (const chunk of process.stdin) source += chunk; console.log(await graphHash(JSON.parse(source)));"
        result = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(graph), text=True, capture_output=True, cwd=ROOT, check=True)
        self.assertEqual(graph_hash(graph), result.stdout.strip())

    def test_hash_authority_and_canonical_sidecar(self):
        graph = example_graph()
        expected = graph_hash(graph)
        self.assertEqual(verify_graph_hash(graph, expected), expected)
        with self.assertRaises(ValueError):
            verify_graph_hash(graph, "0" * 64)
        with self.assertRaises(ValueError):
            verify_graph_hash(graph, "not-a-hash")
        normalized = copy.deepcopy(graph)
        normalized["nodes"] = sorted(normalized["nodes"], key=lambda node: node["id"])
        with tempfile.TemporaryDirectory(dir=ML_ROOT) as directory:
            sidecar = Path(directory) / "canonical.json"
            sidecar.write_text(canonical_json(normalized), encoding="utf-8")
            self.assertEqual(verify_graph_hash(graph, expected, sidecar), expected)
            node_named(normalized, "field")["params"]["angle"] = 0.8
            sidecar.write_text(canonical_json(normalized), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_graph_hash(graph, expected, sidecar)

    def test_json_duplicate_keys_rejected(self):
        with tempfile.TemporaryDirectory(dir=ML_ROOT) as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"id":"one","id":"two"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_graph(path)


class TeacherTests(unittest.TestCase):
    def test_gradient_disk_vignette_and_exposure_match_scalar_math(self):
        graph = example_graph(include_brand=False)
        teacher = TensorTeacher(graph)
        upl = torch.tensor([[0.2, 0.3, 0.4, 0.8]], dtype=torch.float64)
        coordinates = torch.tensor([[0.82, 0.48]], dtype=torch.float64)
        amount = min(1, max(0, 0.65 * 0.82 + 0.4 * (0.48 - 0.5) + 0.35 * 0.8))
        field = node_named(graph, "field")["params"]
        gradient = [low + amount * (high - low) for low, high in zip(field["low"], field["high"])]
        distance = math.hypot(0.82 - (0.65 + 0.12 * (0.8 - 0.5)), 0.48 - 0.4)
        alpha = min(1, max(0, (0.2 - distance) / 0.1 + 0.5))
        disk = [channel + alpha * (target - channel) for channel, target in zip(gradient, [0.9, 0.7, 0.3])]
        attenuation = 1 - 0.3 * min(1, 2 * ((0.82 - 0.5) ** 2 + (0.48 - 0.5) ** 2))
        expected = [min(1, max(0, channel * attenuation * (1 + 0.15 * (2 * 0.8 - 1)))) for channel in disk]
        torch.testing.assert_close(teacher(upl, coordinates), torch.tensor([expected], dtype=torch.float64))

    def test_constant_and_gradient_clamping(self):
        graph = example_graph(include_brand=False)
        graph["nodes"].append({"id": "fixed", "op": "constant", "inputs": {}, "params": {"value": 1}})
        node_named(graph, "light")["inputs"]["condition"] = "fixed"
        node_named(graph, "light")["params"]["amount"] = 1
        node_named(graph, "field")["params"].update({"low": [1, 1, 1], "high": [1, 1, 1]})
        node_named(graph, "shade")["params"]["strength"] = 0
        result = TensorTeacher(graph)(torch.zeros(1, 4), torch.tensor([[0.0, 0.0]]))
        torch.testing.assert_close(result, torch.ones(1, 3))

    def test_all_brand_cells_and_half_open_plate(self):
        graph = example_graph()
        teacher = TensorTeacher(graph)
        coordinates = brand_probe_coordinates(teacher.brand)
        output = teacher(torch.zeros(len(coordinates), 4), coordinates)
        expected = []
        for row in range(9):
            for column in range(7):
                foreground = 1 <= row <= 7 and 1 <= column <= 5 and ((GLYPH_ROWS[row - 1] >> (5 - column)) & 1)
                expected.append([float(bool(foreground))] * 3)
        torch.testing.assert_close(output, torch.tensor(expected))
        boundaries = torch.tensor([[0.0625, 0.0625], [0.0625 + 7 / 64, 0.0625], [0.0625, 0.0625 + 9 / 64]])
        inside, _ = brand_masks(boundaries, teacher.brand)
        self.assertEqual(inside.tolist(), [True, False, False])

    def test_source_changes_change_hash_and_targets(self):
        graph = example_graph()
        changed = copy.deepcopy(graph)
        node_named(changed, "field")["params"]["high"] = [0.8, 0.1, 0.2]
        coordinates = torch.tensor([[0.94, 0.8]])
        upl = torch.full((1, 4), 0.5)
        self.assertNotEqual(graph_hash(graph), graph_hash(changed))
        self.assertFalse(torch.allclose(TensorTeacher(graph)(upl, coordinates), TensorTeacher(changed)(upl, coordinates)))
        changed = copy.deepcopy(graph)
        node_named(changed, "mark")["params"]["background"] = [0.25, 0.5, 0.75]
        coordinates = brand_probe_coordinates(TensorTeacher(changed).brand)[:1]
        self.assertFalse(torch.allclose(TensorTeacher(graph)(upl, coordinates), TensorTeacher(changed)(upl, coordinates)))

    def test_gradients_flow_through_tensor_teacher_and_student(self):
        teacher = TensorTeacher(example_graph(include_brand=False))
        upl = torch.full((2, 4), 0.5, requires_grad=True)
        coordinates = torch.tensor([[0.1, 0.7], [0.9, 0.8]], requires_grad=True)
        teacher(upl, coordinates).sum().backward()
        self.assertTrue(torch.isfinite(upl.grad).all())
        self.assertTrue(torch.isfinite(coordinates.grad).all())
        self.assertGreater(upl.grad.abs().sum().item(), 0)
        self.assertGreater(coordinates.grad.abs().sum().item(), 0)
        model = Student()
        with torch.no_grad():
            targets = teacher(upl.detach(), coordinates.detach())
        loss = (model(torch.cat((coordinates.detach(), upl.detach()), dim=1)) - targets).square().mean()
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in model.parameters()), 0)

    def test_inputs_and_pixel_centers(self):
        torch.testing.assert_close(pixel_coordinates(2, 1), torch.tensor([[0.25, 0.5], [0.75, 0.5]]))
        teacher = TensorTeacher(example_graph())
        for invalid in (torch.zeros(1, 3), torch.full((1, 4), float("nan")), torch.full((1, 4), 1.1)):
            with self.assertRaises(ValueError):
                teacher(invalid, torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            pixel_coordinates(513, 1)

    def test_profile_splits_and_logo_oversampling(self):
        training, heldout = profile_splits(12, 32, 16)
        self.assertFalse(set(map(tuple, training.tolist())) & set(map(tuple, heldout.tolist())))
        repeated, _ = profile_splits(12, 32, 16)
        torch.testing.assert_close(training, repeated)
        brand = TensorTeacher(example_graph()).brand
        upl, coordinates = sample_batch(training, 128, torch.Generator().manual_seed(9), brand)
        inside, _ = brand_masks(coordinates, brand)
        self.assertTrue(inside[:32].all())
        self.assertTrue(set(map(tuple, upl.tolist())).issubset(set(map(tuple, training.tolist()))))

    @unittest.skipUnless(shutil.which("node") and (ROOT / "src/core/graph.js").exists(), "Node teacher is not available")
    def test_tensor_teacher_matches_actual_js_reference(self):
        graph = example_graph()
        teacher = TensorTeacher(graph)
        generator = torch.Generator().manual_seed(912)
        coordinates = torch.cat((torch.rand((128, 2), generator=generator), brand_probe_coordinates(teacher.brand)))
        profiles = torch.rand((len(coordinates), 4), generator=generator)
        expected = teacher(profiles, coordinates).tolist()
        script = "import {createReference} from './src/core/graph.js'; let source = ''; for await (const chunk of process.stdin) source += chunk; const payload = JSON.parse(source); const reference = createReference(payload.graph); console.log(JSON.stringify(payload.xy.map((coordinate,index)=>reference(payload.upl[index],...coordinate))));"
        payload = json.dumps({"graph": graph, "xy": coordinates.tolist(), "upl": profiles.tolist()})
        result = subprocess.run(["node", "--input-type=module", "-e", script], input=payload, text=True, capture_output=True, cwd=ROOT, check=True)
        np.testing.assert_allclose(json.loads(result.stdout), expected, atol=1e-6, rtol=1e-5)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = example_graph()
        cls.artifact, cls.report, cls.fixture = train_student(
            cls.graph, graph_hash(cls.graph), steps=100, batch_size=256,
            train_count=32, heldout_count=16, width=24, height=18, threads=1, max_seconds=60,
        )

    def test_genuine_training_and_measured_brand_fallback(self):
        metrics = self.artifact["metrics"]
        self.assertGreater(metrics["parameterDeltaL2"], 0)
        self.assertLess(metrics["heldoutMse"], metrics["baselineMse"])
        self.assertEqual(self.artifact["training"]["steps"], 100)
        self.assertTrue(self.artifact["training"]["teacherIncludesBrand"])
        self.assertFalse(self.artifact["training"]["pretrained"])
        self.assertFalse(self.artifact["training"]["promoted"])
        self.assertGreater(metrics["brandMaxError"], 0.06)
        self.assertEqual(self.artifact["brand"]["mode"], "composite")
        self.assertEqual(metrics["compositeBrandMaxError"], 0)
        self.assertLess(metrics["heldoutCompositeMse"], metrics["heldoutMse"])
        self.assertFalse(set(map(tuple, self.report["trainProfiles"])) & set(map(tuple, self.report["heldoutProfiles"])))

    def test_row_major_weights_match_exported_point_fixture(self):
        point = self.fixture["pointCase"]
        coordinates = np.asarray(point["xy"], dtype=np.float32)
        upl = np.repeat(np.asarray([point["upl"]], dtype=np.float32), len(coordinates), axis=0)
        hidden = np.concatenate((coordinates, upl), axis=1)
        parameter_count = 0
        for layer in self.artifact["layers"]:
            weights = np.asarray(layer["weights"], dtype=np.float32).reshape(layer["output"], layer["input"])
            bias = np.asarray(layer["bias"], dtype=np.float32)
            hidden = hidden @ weights.T + bias
            hidden = np.maximum(hidden, 0) if layer["activation"] == "relu" else 1 / (1 + np.exp(-hidden))
            parameter_count += weights.size + bias.size
        np.testing.assert_allclose(hidden, point["rawStudentRgb"], atol=1e-6, rtol=1e-5)
        self.assertEqual(parameter_count, 4803)

    def test_json_artifacts_sidecars_and_nonoverwrite(self):
        with tempfile.TemporaryDirectory(dir=ML_ROOT) as directory:
            output = Path(directory) / "student.json"
            paths = write_artifacts(output, self.artifact, copy.deepcopy(self.report), copy.deepcopy(self.fixture))
            saved = [json.loads(path.read_text()) for path in paths]
            self.assertEqual(saved[0]["format"], "angelos.student.v1")
            expected = hashlib.sha256(output.read_bytes()).hexdigest()
            self.assertEqual(saved[1]["artifactSha256"], expected)
            self.assertEqual(saved[2]["artifactSha256"], expected)
            with self.assertRaises(ValueError):
                write_artifacts(output, self.artifact, self.report, self.fixture)

    def test_wrong_compiler_hash_rejected_before_training(self):
        with self.assertRaises(ValueError):
            train_student(self.graph, "0" * 64, steps=1)

    def test_no_brand_export_and_safe_seed(self):
        graph = example_graph(include_brand=False)
        artifact, report, fixture = train_student(
            graph, graph_hash(graph), steps=2, batch_size=16, train_count=4,
            heldout_count=2, width=8, height=8, threads=1,
        )
        self.assertIsNone(artifact["brand"])
        self.assertEqual(artifact["metrics"]["brandMaxError"], 0)
        self.assertFalse(artifact["metrics"]["hasBrand"])
        self.assertIsNone(report["rawStudentHeldout"]["brandMaxError"])
        self.assertEqual(fixture["rasterCases"][0]["rawStudentRgba"], fixture["rasterCases"][0]["studentRgba"])
        for invalid_seed in (-1, 2 ** 53):
            with self.assertRaises(ValueError):
                train_student(graph, graph_hash(graph), steps=1, seed=invalid_seed)

    @unittest.skipUnless(shutil.which("node") and (ROOT / "src/browser/student.js").exists(), "Node student runtime is not available")
    def test_export_validates_and_matches_browser_cpu_runtime(self):
        script = "import {validateStudent,renderCpu} from './src/browser/student.js'; let source = ''; for await (const chunk of process.stdin) source += chunk; const payload = JSON.parse(source); validateStudent(payload.artifact); const cases = payload.fixture.rasterCases.map(item => ({raw:Array.from(renderCpu(payload.artifact,item.upl,item.width,item.height,{composite:false})),composite:Array.from(renderCpu(payload.artifact,item.upl,item.width,item.height))})); console.log(JSON.stringify(cases));"
        payload = json.dumps({"artifact": self.artifact, "fixture": self.fixture})
        result = subprocess.run(["node", "--input-type=module", "-e", script], input=payload, text=True, capture_output=True, cwd=ROOT, check=True)
        actual_cases = json.loads(result.stdout)
        for actual, expected in zip(actual_cases, self.fixture["rasterCases"]):
            np.testing.assert_allclose(actual["raw"], expected["rawStudentRgba"], atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(actual["composite"], expected["studentRgba"], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
