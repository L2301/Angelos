"""Synthetic vector/weight contracts only; no semantic model or HD quality evidence."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from angelos_ml.dataset import build_dataset
from angelos_ml.diffusion import ARCHITECTURE, LEGACY_ENCODER, DiffusionStudent, conditioning_encoder, conditioning_tensor, cosine_schedule, render_cpu, sample_ddim
from angelos_ml.feedback import frozen_upl, materialize_feedback, validate_public_bundle
from angelos_ml.hd import validate_source
from angelos_ml.hd_alignment import AlignmentConfig, _cases
from angelos_ml.rewards import _features, export_bundle, fit_reward_bundle, load_bundle
from angelos_ml.rl import synthetic_evidence
from angelos_ml.rl_policy import PolicyConfig, evaluate_candidate, optimize_policy, profiles_tensor, rollout
from angelos_ml.train_diffusion import generate_procedural_fixture, load_dataset
from angelos_ml.upl import DIMENSIONS, encoder_metadata
from tests.test_ml import example_graph


def synthetic_vector(dimension, seed=12):
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(dimension, generator=generator)
    return (vector / vector.norm()).tolist()


def synthetic_bundle(dimension=128, seed=12):
    return {
        "format": "angelos.upl-input.v2", "synthetic": True, "adult": True,
        "factsSchema": "angelos.facts.v2",
        "facts": {"ageBand": "25-34", "interests": ["synthetic hiking"], "goals": ["fixture exercise"]},
        "encoder": encoder_metadata(dimension), "upl": synthetic_vector(dimension, seed),
        "context": {"campaignId": "aurora", "factRevision": 1, "eventRevision": 1,
                    "consentVersion": 1, "asOf": 100, "ingredientsHash": "a" * 64},
        "recentIntent": [{"kind": "save", "topic": "synthetic trail", "occurredAt": 80, "expiresAt": 2000}],
    }


def synthetic_artifact(model):
    return {"format": model.artifact_format, "id": "synthetic-student", "campaignId": "aurora",
            "graphHash": "b" * 64, "encoder": copy.deepcopy(model.encoder), "architecture": copy.deepcopy(model.architecture),
            "layers": model.export_layers(), "schedule": cosine_schedule(), "brand": None}


def feedback_fixture():
    bundle = synthetic_bundle()
    artifact = synthetic_artifact(DiffusionStudent(bundle["encoder"]))
    digest = hashlib.sha256(json.dumps(artifact).encode()).hexdigest()
    descriptor = {name: copy.deepcopy(artifact[name]) for name in ("id", "campaignId", "graphHash", "encoder")}
    descriptor["hash"] = digest
    identity = json.dumps([artifact["id"], "snapshot-1", 7], separators=(",", ":"))
    creative = f"creative:{hashlib.sha256(identity.encode()).hexdigest()}:7"
    shared = {"artifact": descriptor, "campaignId": "aurora", "encoder": bundle["encoder"],
              "upl": bundle["upl"], "uplBundle": bundle, "profileId": "fixture-profile", "snapshotId": "snapshot-1"}
    document = {"format": "angelos.feedback.v1", "campaignId": "aurora", "asOf": 1000,
                "outcome": "click", "population": "local-synthetic-adults",
                "clicks": [{**copy.deepcopy(shared), "impressionId": "impression-1", "label": 1, "occurredAt": 200,
                            "maturesAt": 900, "context": {"creativeId": creative}}],
                "reviews": [{**copy.deepcopy(shared), "id": "review-1", "receivedAt": 500, "source": "operator",
                             "creativeId": creative, "qualityRating": 4, "brandPass": True}]}
    bank = {artifact["id"]: {"artifact": artifact, "model": None, "hash": digest}}
    return document, bank, artifact


class UPLTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_all_widths_have_signed_normalized_contracts_and_exact_pin(self):
        for dimension in DIMENSIONS:
            with self.subTest(dimension=dimension):
                encoder = encoder_metadata(dimension)
                vector = synthetic_vector(dimension)
                self.assertLess(min(vector), 0)
                self.assertEqual(conditioning_tensor(vector, encoder).shape, (dimension,))
                self.assertEqual(profiles_tensor([vector], "cpu", encoder).shape, (1, dimension))
                for invalid in (vector[:4], [0.0] * dimension, [value * 2 for value in vector], [True] + [0.0] * (dimension - 1)):
                    with self.assertRaises(ValueError):
                        conditioning_tensor(invalid, encoder)
                for field, value in (("revision", "a" * 40), ("normalization", "none"), ("pooling", "mean"),
                                     ("modelId", "unknown/model"), ("templateVersion", "other")):
                    with self.assertRaises(ValueError):
                        conditioning_encoder({**encoder, field: value})
        with self.assertRaises(ValueError):
            conditioning_tensor(synthetic_vector(128))
        with self.assertRaises(ValueError):
            conditioning_tensor([-0.1, 0, 0, 0], LEGACY_ENCODER)

    def test_native32_forward_uses_every_width_and_backpropagates_beyond_four(self):
        for dimension in DIMENSIONS:
            with self.subTest(dimension=dimension), torch.random.fork_rng():
                torch.manual_seed(31)
                model = DiffusionStudent(encoder_metadata(dimension))
                profiles = torch.tensor([synthetic_vector(dimension)], requires_grad=True)
                result = model(torch.zeros(1, 3, 32, 32), profiles, torch.tensor([12]))
                self.assertEqual(result.shape, (1, 3, 32, 32))
                self.assertEqual(model.layers[0].in_channels, dimension + 6)
                result.square().mean().backward()
                self.assertGreater(profiles.grad[:, 4:].abs().sum().item(), 0)

    def test_python_v2_checkpoint_roundtrip_and_legacy_export_refusal(self):
        model = DiffusionStudent(encoder_metadata(128))
        artifact = synthetic_artifact(model)
        restored = DiffusionStudent.from_artifact(artifact)
        self.assertEqual(restored.encoder, model.encoder)
        self.assertEqual(restored.artifact_format, "angelos.diffusion.v2")
        for original, loaded in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(original, loaded))
        self.assertEqual(sample_ddim(restored, synthetic_vector(128), 12).shape, (1, 3, 32, 32))
        with self.assertRaisesRegex(ValueError, "Legacy browser"):
            render_cpu(artifact, synthetic_vector(128))
        with self.assertRaises(ValueError):
            DiffusionStudent.from_artifact({**artifact, "format": "angelos.diffusion.v1"})
        browser_artifact = {**artifact, "metrics": {}, "training": {}}
        result = subprocess.run(
            ["node", "--input-type=module", "-e",
             "import assert from 'node:assert/strict'; import {readFileSync} from 'node:fs'; "
             "import {validateDiffusion} from './src/browser/diffusion.js'; "
             "assert.throws(() => validateDiffusion(JSON.parse(readFileSync(0, 'utf8'))), /Unsupported diffusion format/);"],
            cwd=Path(__file__).resolve().parents[2], input=json.dumps(browser_artifact), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        legacy = DiffusionStudent()
        self.assertEqual(legacy.architecture, ARCHITECTURE)
        self.assertEqual(legacy.layers[0].in_channels, 10)
        self.assertEqual(DiffusionStudent.from_artifact(synthetic_artifact(legacy)).artifact_format, "angelos.diffusion.v1")

    def test_reward_features_do_not_recenter_signed_embeddings(self):
        encoder = encoder_metadata(128)
        profiles = torch.tensor([synthetic_vector(128)])
        features = _features(torch.full((1, 3, 16, 16), 0.75), profiles, encoder)
        self.assertEqual(features.shape, (1, 12 + 13 * 128))
        torch.testing.assert_close(features[:, 12:140], profiles)

    def test_reward_learning_roundtrip_and_image_gradients_at_each_width(self):
        for dimension in DIMENSIONS:
            with self.subTest(dimension=dimension):
                encoder = encoder_metadata(dimension)
                clicks, reviews = synthetic_evidence(2026, encoder)
                rewards = fit_reward_bundle(clicks, reviews, encoder=encoder, allow_synthetic=True, steps=3)
                payload = export_bundle(rewards)
                self.assertEqual(payload["format"], "angelos.rewards.v2")
                restored = load_bundle(payload)
                self.assertEqual(restored.encoder, encoder)
                self.assertEqual(restored.id, rewards.id)
                image = torch.full((1, 3, 16, 16), 0.5, requires_grad=True)
                scores = restored.predict(image, [synthetic_vector(dimension)])
                scores["click"].sum().backward()
                self.assertGreater(image.grad.abs().sum().item(), 0)
                self.assertTrue(all(not parameter.requires_grad for parameter in restored.click_model.parameters()))
                with self.assertRaises(ValueError):
                    restored.predict(image, [[0.1] * 4])

    def test_reward_record_encoder_mismatch_fails_before_fitting(self):
        encoder = encoder_metadata(128)
        clicks, reviews = synthetic_evidence(2026, encoder)
        reviews[-1]["encoder"]["revision"] = "0" * 40
        with patch("angelos_ml.rewards._train") as train:
            with self.assertRaisesRegex(ValueError, "encoder"):
                fit_reward_bundle(clicks, reviews, encoder=encoder, allow_synthetic=True, steps=1)
            train.assert_not_called()

    def test_nonfixture_reward_records_require_public_bundle(self):
        encoder = encoder_metadata(128)
        clicks, reviews = synthetic_evidence(2026, encoder)
        for record in clicks:
            record["source"] = "local-ledger"
        for record in reviews:
            record["source"] = "operator"
        with self.assertRaisesRegex(ValueError, "bundle"):
            fit_reward_bundle(clicks, reviews, encoder=encoder, steps=1)

    def test_reward_checkpoint_identity_binds_recent_intent(self):
        encoder = encoder_metadata(128)
        clicks, reviews = synthetic_evidence(2026, encoder)
        for record in clicks + reviews:
            bundle = synthetic_bundle()
            bundle["upl"] = copy.deepcopy(record["upl"])
            record["uplBundle"] = bundle
        original = fit_reward_bundle(clicks, reviews, encoder=encoder, allow_synthetic=True, steps=1)
        for record in clicks + reviews:
            record["uplBundle"]["recentIntent"][0]["topic"] = "different valid frozen intent"
        changed = fit_reward_bundle(clicks, reviews, encoder=encoder, allow_synthetic=True, steps=1)
        self.assertNotEqual(original.id, changed.id)
        self.assertTrue(torch.equal(original.click_model.weight, changed.click_model.weight))
        self.assertIn("semantic-vector-only", changed.metadata["conditioning"])
        self.assertEqual(load_bundle(export_bundle(changed)).metadata["uplProvenance"][0]["uplBundle"]["recentIntent"][0]["topic"],
                         "different valid frozen intent")

    def test_feedback_preserves_full_frozen_bundle_and_replays_once(self):
        document, bank, artifact = feedback_fixture()
        with patch("angelos_ml.feedback.sample_ddim", return_value=torch.full((1, 3, 32, 32), 0.5)) as replay:
            clicks, reviews = materialize_feedback(document, bank, parent=artifact)
        replay.assert_called_once()
        self.assertEqual(clicks[0]["uplBundle"], document["clicks"][0]["uplBundle"])
        self.assertEqual(reviews[0]["encoder"], artifact["encoder"])
        document["reviews"][0]["uplBundle"]["recentIntent"][0]["topic"] = "mutated after materialization"
        self.assertEqual(reviews[0]["uplBundle"]["recentIntent"][0]["topic"], "synthetic trail")

    def test_feedback_missing_or_conflicting_intent_rejected_before_any_replay(self):
        for fault in ("missing", "intent", "encoder", "vector", "future", "campaign", "artifact-encoder"):
            with self.subTest(fault=fault):
                document, bank, artifact = feedback_fixture()
                row = document["reviews"][0]
                if fault == "missing":
                    del row["uplBundle"]
                elif fault == "intent":
                    row["uplBundle"]["recentIntent"][0]["topic"] = "different frozen topic"
                elif fault == "encoder":
                    row["uplBundle"]["encoder"]["revision"] = "0" * 40
                elif fault == "vector":
                    row["upl"] = synthetic_vector(128, 44)
                elif fault == "future":
                    row["uplBundle"]["context"]["asOf"] = 900
                elif fault == "campaign":
                    row["uplBundle"]["context"]["campaignId"] = "other"
                else:
                    row["artifact"]["encoder"]["revision"] = "0" * 40
                with patch("angelos_ml.feedback.sample_ddim") as replay:
                    with self.assertRaises(ValueError):
                        materialize_feedback(document, bank, parent=artifact)
                    replay.assert_not_called()

    def test_recent_intent_validity_uses_snapshot_time_not_current_time(self):
        bundle = synthetic_bundle()
        validate_public_bundle(bundle)
        row = {"upl": bundle["upl"], "encoder": bundle["encoder"], "uplBundle": bundle, "occurredAt": 200}
        self.assertEqual(frozen_upl(row, "aurora")["uplBundle"], bundle)
        for change in ({"expiresAt": 100}, {"occurredAt": 101}):
            invalid = copy.deepcopy(bundle)
            invalid["recentIntent"][0].update(change)
            with self.assertRaises(ValueError):
                validate_public_bundle(invalid)

    def test_wide_rl_updates_fixture_weights_without_changing_parent(self):
        encoder = encoder_metadata(128)
        model = DiffusionStudent(encoder)
        parent_weights = copy.deepcopy(model.state_dict())
        clicks, reviews = synthetic_evidence(2026, encoder)
        rewards = fit_reward_bundle(clicks, reviews, encoder=encoder, allow_synthetic=True, steps=3)
        candidate, report = optimize_policy(model, [synthetic_vector(128)], rewards, schedule=cosine_schedule(),
                                           config=PolicyConfig(rounds=1, epochs=1, group_size=2), seed=21)
        self.assertGreater(report["weightDeltaL2"], 0)
        self.assertTrue(all(torch.equal(value, model.state_dict()[name]) for name, value in parent_weights.items()))
        audit = evaluate_candidate(model, candidate, [synthetic_vector(128, 25)], rewards,
                                   schedule=cosine_schedule(), brand=None, config=PolicyConfig(), seed=42)
        self.assertIsNone(audit["observedCtrLift"])
        with self.assertRaisesRegex(ValueError, "encoder"):
            optimize_policy(DiffusionStudent(), [[0.1] * 4], rewards, schedule=cosine_schedule())
        with self.assertRaisesRegex(ValueError, "encoder"):
            rollout(model, [synthetic_vector(128)], cosine_schedule(), 42, terminal_model=DiffusionStudent())

    def test_dataset_width_and_frozen_input_provenance(self):
        graph = copy.deepcopy(example_graph())
        graph["version"] = "angelos.graph.v2"
        graph["encoder"] = encoder_metadata(128)
        bundles = [synthetic_bundle(seed=seed) for seed in (1, 2, 3, 4)]
        for bundle in bundles:
            bundle["context"]["campaignId"] = graph["id"]
        arrays = {"images": np.full((4, 3, 32, 32), 0.5, dtype=np.float32),
                  "upl": np.array([bundle["upl"] for bundle in bundles], dtype=np.float32),
                  "split": np.array([0, 0, 1, 1], dtype=np.int64), "teacher_seed": np.array([1, 2, 3, 4], dtype=np.int64)}
        manifest = {"graphHash": "a" * 64, "graph": graph, "encoder": graph["encoder"], "models": [],
                    "source": "procedural-fixture-explicit", "synthetic": True,
                    "samples": [{"index": index, "profileId": f"synthetic-person-{index}", "upl": bundle["upl"], "encoder": bundle["encoder"],
                                 "uplBundle": bundle, "split": int(arrays["split"][index]), "teacherSeed": index + 1}
                                for index, bundle in enumerate(bundles)]}
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "fixture.npz"
            np.savez(dataset, **arrays)
            sidecar = dataset.with_suffix(".manifest.json")
            sidecar.write_text(json.dumps(manifest))
            loaded, provenance = load_dataset(dataset, graph, "a" * 64, True)
            self.assertEqual(loaded["upl"].shape, (4, 128))
            self.assertEqual(provenance["manifest"]["samples"][0]["uplBundle"], bundles[0])
            del manifest["samples"][-1]["uplBundle"]
            sidecar.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "bundle"):
                load_dataset(dataset, graph, "a" * 64, True)

    def test_unmigrated_generators_fail_before_provider_or_encoder_calls(self):
        graph = copy.deepcopy(example_graph())
        graph["encoder"] = encoder_metadata(128)
        with patch("angelos_ml.dataset.RasterTeacher") as teacher:
            with self.assertRaisesRegex(ValueError, "legacy-only"):
                build_dataset(graph)
            teacher.assert_not_called()
        with self.assertRaisesRegex(ValueError, "legacy-only"):
            generate_procedural_fixture("unused.npz", graph, "a" * 64, "unused.json")
        with self.assertRaisesRegex(ValueError, "prompt-only"):
            validate_source({"upl": synthetic_vector(128)})

    def test_hd_reward_cases_bind_bundles_without_claiming_native_upl_conditioning(self):
        encoder = encoder_metadata(128)
        bundle = synthetic_bundle()
        training = {"id": "train", "profileId": "train-profile", "prompt": "synthetic fixture", "seed": 1,
                    "encoder": encoder, "upl": bundle["upl"], "uplBundle": bundle}
        audit = {**copy.deepcopy(training), "id": "audit", "profileId": "audit-profile", "seed": 2}
        rewards = SimpleNamespace(encoder=encoder, split_profiles={"train": [], "validation": []})
        prepared, heldout = _cases([training], [audit], AlignmentConfig(fixture_only=True), rewards)
        self.assertEqual(prepared[0]["uplBundle"]["recentIntent"], bundle["recentIntent"])
        self.assertEqual(heldout[0]["encoder"], encoder)
        del audit["uplBundle"]
        with self.assertRaisesRegex(ValueError, "bundle"):
            _cases([training], [audit], AlignmentConfig(fixture_only=True), rewards)


if __name__ == "__main__":
    unittest.main()
