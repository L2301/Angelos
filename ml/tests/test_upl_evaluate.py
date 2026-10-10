"""Injected numeric fixtures only: these tests never load or evaluate real weights."""

import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from angelos_ml.upl import REVISION, semantic_text
from angelos_ml.upl_evaluate import DEFAULT_FIXTURE, MAX_CORPUS_BYTES, dimension_report, evaluate, main, nearest_neighbors, normalized_prefix, read_corpus, validate_corpus, vector_statistics, write_report


class InjectedNumericEncoder:
    def __init__(self, vectors, device="cpu"):
        self.vectors = vectors
        self.device = device
        self.calls = []

    def encode(self, text, dimension):
        index = len(self.calls)
        self.calls.append((text, dimension))
        return copy.deepcopy(self.vectors[index])


def numeric_fixture(corpus):
    groups = sorted({profile["expectedGroup"] for profile in corpus["profiles"]})
    vectors = []
    for profile in corpus["profiles"]:
        vector = [0.0] * 1024
        vector[groups.index(profile["expectedGroup"])] = -1.0
        vectors.append(vector)
    return InjectedNumericEncoder(vectors)


class UplEvaluateTests(unittest.TestCase):
    def setUp(self):
        self.corpus, self.corpus_hash = read_corpus(DEFAULT_FIXTURE)

    def evaluate_fixture(self, encoder=None):
        encoder = encoder or numeric_fixture(self.corpus)
        loader = Mock(return_value=encoder)
        ticks = iter(range(1000))
        report = evaluate(encoder_loader=loader, clock=lambda: next(ticks))
        return report, loader, encoder

    def test_bounded_corpus_has_explicit_fictional_identities_and_shared_baseline(self):
        profiles = self.corpus["profiles"]
        self.assertEqual(len(profiles), 12)
        self.assertEqual(len({profile["syntheticPersonId"] for profile in profiles}), 12)
        self.assertEqual(len({profile["expectedGroup"] for profile in profiles}), 4)
        baseline = [{field: profile["facts"][field] for field in ("ageBand", "region", "openness", "energy")} for profile in profiles]
        self.assertTrue(all(facts == baseline[0] for facts in baseline))
        self.assertEqual(self.corpus_hash, hashlib.sha256(DEFAULT_FIXTURE.read_bytes()).hexdigest())

    def test_load_once_encode_once_at_1024_and_keep_labels_out_of_model_input(self):
        report, loader, encoder = self.evaluate_fixture()
        loader.assert_called_once_with(device="cpu", cache_dir=None)
        self.assertEqual(encoder.calls, [(semantic_text(profile["facts"]), 1024) for profile in self.corpus["profiles"]])
        self.assertEqual(report["encoder"]["revision"], REVISION)
        self.assertEqual(report["corpus"]["sha256"], self.corpus_hash)
        self.assertEqual(report["execution"]["mode"], "injected-unit-fixture-not-real-model-output")
        self.assertFalse(report["execution"]["checkpointLoaded"])
        self.assertEqual(report["execution"]["modelLoads"], 0)
        self.assertEqual(report["execution"]["encoderInstances"], 1)
        self.assertEqual(report["execution"]["modelDevice"], "cpu")
        self.assertEqual(report["latency"]["modelOrFixtureLoad"], 1)
        self.assertEqual(report["latency"]["encodeTotal"], 12)
        self.assertGreater(report["latency"]["evaluationWall"], report["latency"]["encodeTotal"])
        self.assertEqual([result["dimension"] for result in report["dimensions"]], [128, 256, 512, 1024])
        self.assertTrue(report["syntheticSemanticSmokeOnly"])
        self.assertIn("not CTR", " ".join(report["limitations"]))
        self.assertIn("dimension-selection", " ".join(report["limitations"]))

    def test_nested_prefixes_renormalize_actual_leading_coordinates(self):
        vector = [-3.0, 4.0] + [0.0] * 1022
        vector[128] = 12.0
        vector[256] = 84.0
        vector[512] = 132.0
        for dimension, norm in ((128, 5.0), (256, 13.0), (512, 85.0), (1024, math.hypot(85, 132))):
            with self.subTest(dimension=dimension):
                prefix, statistics = normalized_prefix(vector, dimension)
                self.assertEqual(len(prefix), dimension)
                self.assertAlmostEqual(statistics["l2Norm"], norm)
                self.assertAlmostEqual(prefix[0], -3 / norm)
                self.assertAlmostEqual(prefix[1], 4 / norm)
                self.assertAlmostEqual(math.hypot(*prefix), 1)
        self.assertEqual(vector[0], -3.0)
        self.assertEqual(vector[512], 132.0)

    def test_numeric_norms_payload_bytes_and_leave_self_out_neighbors(self):
        report, _, _ = self.evaluate_fixture()
        for result in report["dimensions"]:
            dimension = result["dimension"]
            self.assertEqual(result["status"], "valid")
            self.assertEqual(result["payload"]["float32EquivalentBytesPerProfile"], dimension * 4)
            self.assertEqual(result["payload"]["float32EquivalentBytesForCorpus"], dimension * 4 * 12)
            expected_json_bytes = len(json.dumps([-1.0] + [0.0] * (dimension - 1), separators=(",", ":")).encode())
            self.assertEqual(result["payload"]["compactVectorJsonBytesForCorpus"], expected_json_bytes * 12)
            for row in result["statistics"]:
                self.assertEqual(row["prefixStatistics"]["nonfiniteCount"], 0)
                self.assertEqual(row["normalizedL2Norm"], 1)
            smoke = result["semanticSmoke"]
            self.assertEqual(smoke["meanFractionalDeclaredGroupPurity"], 1)
            for row in smoke["profiles"]:
                self.assertEqual(row["nearestTieCount"], 2)
                self.assertNotIn(row["syntheticPersonId"], [neighbor["syntheticPersonId"] for neighbor in row["nearestNeighbors"]])
                self.assertTrue(all(neighbor["expectedGroup"] == row["expectedGroup"] for neighbor in row["nearestNeighbors"]))

    def test_frozen_four_dimensional_baseline_counts_all_ties_without_optimistic_argmax(self):
        report, _, _ = self.evaluate_fixture()
        baseline = report["frozen4dBaseline"]
        self.assertEqual(baseline["rawVectorForEveryProfile"], [0.4, 0, 0.6, 0.5])
        self.assertTrue(baseline["allProfilesIdentical"])
        smoke = baseline["semanticSmoke"]
        self.assertEqual(smoke["profilesWithTies"], 12)
        self.assertAlmostEqual(smoke["meanFractionalDeclaredGroupPurity"], 2 / 11)
        for row in smoke["profiles"]:
            self.assertEqual(row["nearestTieCount"], 11)
            self.assertAlmostEqual(row["fractionalDeclaredGroupPurity"], 2 / 11)
        reordered = nearest_neighbors(list(reversed(self.corpus["profiles"])), [[1.0, 0.0]] * 12)
        self.assertEqual(reordered["meanFractionalDeclaredGroupPurity"], smoke["meanFractionalDeclaredGroupPurity"])

    def test_nonfinite_and_zero_vectors_are_diagnostics_not_successful_scores(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=str(value)):
                encoder = numeric_fixture(self.corpus)
                encoder.vectors[0][900] = value
                report, _, _ = self.evaluate_fixture(encoder)
                self.assertEqual(report["status"], "invalid-vectors")
                self.assertEqual(report["profiles"][0]["source1024Statistics"]["nonfiniteCount"], 1)
                self.assertIsNone(report["profiles"][0]["source1024Statistics"]["l2Norm"])
                self.assertTrue(all(result["semanticSmoke"] is None for result in report["dimensions"]))
                json.dumps(report, allow_nan=False)
        encoder = numeric_fixture(self.corpus)
        encoder.vectors[0] = [0.0] * 1024
        report, _, _ = self.evaluate_fixture(encoder)
        self.assertEqual(report["status"], "invalid-vectors")
        self.assertEqual(report["profiles"][0]["source1024Statistics"]["l2Norm"], 0)
        self.assertIsNone(report["dimensions"][0]["semanticSmoke"])

    def test_zero_prefix_is_not_replaced_with_a_random_vector(self):
        vector = [0.0] * 1024
        vector[900] = 1.0
        result = dimension_report(self.corpus["profiles"], [vector] * 12, 128)
        self.assertEqual(result["status"], "invalid-vectors")
        self.assertIsNone(result["semanticSmoke"])
        self.assertIsNone(result["payload"]["compactVectorJsonBytesForCorpus"])
        self.assertTrue(vector_statistics(vector)["valid"])

    def test_refuse_wrong_width_boolean_vectors_and_device_fallback(self):
        for vector in ([0.0] * 512, [True] + [0.0] * 1023):
            with self.subTest(length=len(vector)):
                encoder = numeric_fixture(self.corpus)
                encoder.vectors[0] = vector
                with self.assertRaisesRegex(ValueError, "1024 numeric"):
                    self.evaluate_fixture(encoder)
        encoder = numeric_fixture(self.corpus)
        encoder.device = "mps"
        with self.assertRaisesRegex(ValueError, "no silent fallback"):
            self.evaluate_fixture(encoder)

    def test_validate_bounds_adult_flags_identifiers_facts_and_groups(self):
        changes = [
            lambda corpus: corpus.update(synthetic=False),
            lambda corpus: corpus.update(adult=1),
            lambda corpus: corpus.update(extra="unsupported"),
            lambda corpus: corpus["profiles"].pop(),
            lambda corpus: corpus["profiles"][0].update(syntheticPersonId=corpus["profiles"][1]["syntheticPersonId"]),
            lambda corpus: corpus["profiles"][0].update(syntheticPersonId="a real name"),
            lambda corpus: corpus["profiles"][0].update(expectedGroup="unbounded " * 100),
            lambda corpus: corpus["profiles"][0]["facts"].update(ageBand="under-18"),
            lambda corpus: corpus["profiles"][0]["facts"].update(energy=True),
            lambda corpus: corpus["profiles"][0]["facts"].update(openness=float("nan")),
            lambda corpus: corpus["profiles"][0]["facts"].update(region="EU"),
            lambda corpus: corpus["profiles"][0]["facts"].update(interests=[]),
            lambda corpus: corpus["profiles"][0]["facts"].update(interests=["phrase"] * 5),
            lambda corpus: corpus["profiles"][0]["facts"].update(preferredLanguage=None),
            lambda corpus: [profile.update(expectedGroup="one-group") for profile in corpus["profiles"]],
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                corpus = copy.deepcopy(self.corpus)
                change(corpus)
                with self.assertRaises(ValueError):
                    validate_corpus(corpus)

    def test_invalid_or_oversized_fixture_fails_before_loading_any_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.json"
            for payload in ('{"format":1,"format":2}', ' ' * (MAX_CORPUS_BYTES + 1), '{"value":NaN}'):
                fixture.write_text(payload)
                loader = Mock()
                with self.assertRaises(ValueError):
                    evaluate(fixture, encoder_loader=loader)
                loader.assert_not_called()

    def test_report_write_is_exclusive_and_cli_checks_existing_output_before_loading(self):
        report, _, _ = self.evaluate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            write_report(report, output)
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                write_report({"replacement": True}, output)
            self.assertEqual(output.read_bytes(), before)
            with patch("angelos_ml.upl_evaluate.evaluate") as evaluator, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as failure:
                    main(["--device", "cpu", "--output", str(output)])
                self.assertEqual(failure.exception.code, 2)
                evaluator.assert_not_called()
            self.assertEqual(output.read_bytes(), before)

    def test_cli_writes_labeled_injected_report_or_reports_missing_cache_without_substitution(self):
        report, _, _ = self.evaluate_fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke.json"
            with patch("angelos_ml.upl_evaluate.evaluate", return_value=report) as evaluator, contextlib.redirect_stdout(io.StringIO()):
                main(["--device", "cpu", "--output", str(output)])
                evaluator.assert_called_once_with(DEFAULT_FIXTURE, device="cpu", cache_dir=None)
            self.assertEqual(json.loads(output.read_text())["execution"]["mode"], "injected-unit-fixture-not-real-model-output")
            missing_output = Path(directory) / "missing.json"
            with patch("angelos_ml.upl_evaluate.evaluate", side_effect=RuntimeError("checkpoint missing; cache only")), contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as failure:
                    main(["--output", str(missing_output)])
                self.assertEqual(failure.exception.code, 2)
                self.assertIn("checkpoint missing; cache only", errors.getvalue())
            self.assertFalse(missing_output.exists())


if __name__ == "__main__":
    unittest.main()
