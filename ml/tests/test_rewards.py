"""CPU reward contract tests using fabricated evidence, not quality milestones."""

import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
from torch.nn import functional

from angelos_ml.rewards import RewardBundle, export_bundle, fit_reward_bundle, load_bundle
from angelos_ml.rewards import _features


def fixture_records(profile_count=12):
    clicks, reviews = [], []
    for profile in range(profile_count):
        for creative in range(4):
            label = creative // 2
            brightness = 0.1 + 0.7 * label + 0.025 * (profile % 3 + creative % 2)
            shared = {"image": torch.full((3, 32, 32), brightness),
                      "upl": [profile / profile_count, 0.2, 0.6, 0.8],
                      "profileId": f"profile-{profile:02d}", "creativeId": f"creative-{creative}",
                      "source": "synthetic"}
            clicks.append({**shared, "label": label})
            reviews.append({**shared, "quality": 0.15 + 0.7 * label, "brand": float(label)})
    return clicks, reviews


def reidentify(artifact):
    payload = {key: value for key, value in artifact.items() if key != "id"}
    serialized = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    artifact["id"] = "reward-" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return artifact


class RewardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.clicks, cls.reviews = fixture_records()
        cls.bundle = fit_reward_bundle(cls.clicks, cls.reviews, allow_synthetic=True)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def fit(self, clicks=None, reviews=None, **kwargs):
        options = {"allow_synthetic": True, "steps": 20, **kwargs}
        return fit_reward_bundle(self.clicks if clicks is None else clicks,
                                 self.reviews if reviews is None else reviews, **options)

    def test_actual_learning_and_training_intercept_baselines(self):
        metrics = self.bundle.metadata["metrics"]
        for task, loss in (("click", "bce"), ("click", "brier"), ("quality", "mse"), ("brand", "bce")):
            self.assertLess(metrics[task][loss], metrics[task]["baseline_" + loss] * 0.5)
        self.assertAlmostEqual(metrics["click"]["baseline_probability"], 0.5)
        self.assertAlmostEqual(metrics["click"]["baseline_brier"], 0.25)
        self.assertAlmostEqual(metrics["quality"]["baseline_mean"], 0.5)
        self.assertAlmostEqual(metrics["quality"]["baseline_mse"], 0.35 ** 2)
        heldout = [record for record in self.clicks if record["profileId"] in self.bundle.split_profiles["validation"]]
        predictions = self.bundle.predict(torch.stack([record["image"] for record in heldout]),
                                          torch.tensor([record["upl"] for record in heldout]))
        labels = torch.tensor([record["label"] for record in heldout], dtype=torch.float32)
        self.assertAlmostEqual(metrics["click"]["brier"], (predictions["click"] - labels).square().mean().item())
        self.assertAlmostEqual(metrics["click"]["bce"], functional.binary_cross_entropy(
            predictions["click"].clamp(1e-6, 1 - 1e-6), labels).item())
        self.assertGreater(self.bundle.click_model.weight.abs().sum().item(), 0)
        self.assertGreater(self.bundle.review_model.weight.abs().sum().item(), 0)

    def test_single_disjoint_union_partition_and_counts(self):
        clicks = [record for record in self.clicks if record["profileId"] < "profile-09"]
        reviews = [record for record in self.reviews if record["profileId"] >= "profile-03"]
        bundle = self.fit(clicks, reviews, validation_fraction=0.5)
        split = bundle.split_profiles
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertEqual(set(split["train"] + split["validation"]),
                         {record["profileId"] for record in clicks + reviews})
        for task, records in (("click", clicks), ("review", reviews)):
            for part in ("train", "validation"):
                selected = [record for record in records if record["profileId"] in split[part]]
                counts = bundle.metadata["counts"][task][part]
                self.assertEqual(counts["records"], len(selected))
                self.assertEqual(counts["profiles"], len({record["profileId"] for record in selected}))
                self.assertEqual(bundle.metadata["sources"][task][part], {"synthetic": len(selected)})

    def test_heldout_records_never_train_either_model(self):
        first = self.fit()
        validation = first.split_profiles["validation"]
        clicks, reviews = copy.deepcopy((self.clicks, self.reviews))
        for record in clicks + reviews:
            if record["profileId"] in validation:
                record["image"] = torch.full((3, 16, 48), 0.5)
                record["upl"] = [0.1] * 4
                if "label" in record:
                    record["label"] = 1 - record["label"]
                else:
                    record["quality"] = 1 - record["quality"]
                    record["brand"] = 1 - record["brand"]
        second = self.fit(clicks, reviews)
        self.assertEqual(first.split_profiles, second.split_profiles)
        for before, after in ((first.click_model, second.click_model), (first.review_model, second.review_model)):
            for key, weight in before.state_dict().items():
                self.assertTrue(torch.equal(weight, after.state_dict()[key]))
        self.assertNotEqual(first.metadata["metrics"], second.metadata["metrics"])

    def test_deterministic_local_rng_does_not_touch_global_state(self):
        torch_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        first, second = self.fit(), self.fit()
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertEqual(export_bundle(first), export_bundle(second))
        self.assertNotEqual(first.id, self.fit(seed=17).id)

    def test_frozen_separate_models_and_prediction_contract(self):
        self.assertIsInstance(self.bundle, RewardBundle)
        self.assertIsNot(self.bundle.click_model, self.bundle.review_model)
        for model in (self.bundle.click_model, self.bundle.review_model):
            self.assertFalse(model.training)
            for parameter in model.parameters():
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)
                self.assertEqual(parameter.device.type, "cpu")
        images = torch.ones((3, 3, 32, 32))
        upl = torch.full((3, 4), 0.5)
        result = self.bundle.predict(images, upl)
        self.assertEqual(set(result), {"click", "quality", "brand", "click_std", "quality_std", "brand_std"})
        for name, value in result.items():
            self.assertEqual(tuple(value.shape), (3,))
            self.assertEqual(value.dtype, torch.float32)
            self.assertEqual(value.device.type, "cpu")
            self.assertFalse(value.requires_grad)
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(((value >= 0) & (value <= (0.5 if name.endswith("_std") else 1))).all())
        self.assertIsNone(images.grad)
        self.assertIsNone(upl.grad)
        self.assertTrue(any(result[name].max() > 0 for name in result if name.endswith("_std")))

    def test_image_backprop_through_pooling_and_float32_conversion_including_1080p(self):
        names = ("click", "quality", "brand", "click_std", "quality_std", "brand_std")
        for height, width, dtype in ((16, 24, torch.float64), (1080, 1920, torch.float32)):
            with self.subTest(shape=(height, width), dtype=dtype):
                logits = torch.zeros((1, 3, height, width), dtype=dtype, requires_grad=True)
                images = logits.sigmoid()
                upl = torch.full((1, 4), 0.5)
                result = self.bundle.predict(images, upl)
                for index, name in enumerate(names):
                    self.assertTrue(result[name].requires_grad)
                    self.assertEqual(result[name].dtype, torch.float32)
                    gradient = torch.autograd.grad(result[name].sum(), logits,
                                                   retain_graph=index < len(names) - 1)[0]
                    self.assertEqual(gradient.shape, logits.shape)
                    self.assertEqual(gradient.dtype, dtype)
                    self.assertTrue(torch.isfinite(gradient).all())
                    self.assertGreater(gradient.abs().sum().item(), 0)
        for model in (self.bundle.click_model, self.bundle.review_model):
            self.assertFalse(model.training)
            for parameter in model.parameters():
                self.assertFalse(parameter.requires_grad)
                self.assertIsNone(parameter.grad)

    def test_loaded_bundle_backprop_and_caller_controlled_no_grad(self):
        bundle = load_bundle(export_bundle(self.bundle))
        images = torch.full((1, 3, 32, 48), 0.5, requires_grad=True)
        upl = torch.full((1, 4), 0.5, requires_grad=True)
        predictions = bundle.predict(images, upl)
        sum(predictions[name].sum() for name in ("click", "quality", "brand")).backward()
        for tensor in (images, upl):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        with torch.no_grad():
            inference = bundle.predict(images, upl)
        detached = bundle.predict(images.detach(), upl.detach())
        for name, prediction in predictions.items():
            for result in (inference, detached):
                self.assertFalse(result[name].requires_grad)
                self.assertIsNone(result[name].grad_fn)
                self.assertTrue(torch.equal(result[name], prediction.detach()))
        for model in (bundle.click_model, bundle.review_model):
            for parameter in model.parameters():
                self.assertIsNone(parameter.grad)
                self.assertFalse(parameter.requires_grad)
        self.assertEqual(export_bundle(bundle), export_bundle(self.bundle))

    def test_training_evidence_stays_detached(self):
        clicks, reviews = copy.deepcopy((self.clicks, self.reviews))
        image_logits = torch.zeros((3, 32, 32), requires_grad=True)
        upl_logits = torch.zeros(4, requires_grad=True)
        for records in (clicks, reviews):
            records[0]["image"] = image_logits.sigmoid()
            records[0]["upl"] = upl_logits.sigmoid()
        self.fit(clicks, reviews)
        self.assertIsNone(image_logits.grad)
        self.assertIsNone(upl_logits.grad)

    def test_1080p_and_dimension_boundaries_without_fidelity_claim(self):
        conditioning = torch.full((1, 4), 0.5)
        reference = self.bundle.predict(torch.full((1, 3, 32, 32), 0.25), conditioning)
        for height, width in ((1080, 1920), (16, 4096), (4096, 16), (17, 51)):
            with self.subTest(shape=(height, width)):
                result = self.bundle.predict(torch.full((1, 3, height, width), 0.25), conditioning)
                for name in result:
                    torch.testing.assert_close(reference[name], result[name])
        limitations = " ".join(self.bundle.metadata["limitations"])
        self.assertIn("no high-resolution brand fidelity claim", limitations)
        self.assertIn("not a genuine 1080p generation quality milestone", limitations)

    def test_mixed_resolution_training_and_list_images(self):
        clicks, reviews = copy.deepcopy((self.clicks, self.reviews))
        for index, record in enumerate(clicks + reviews):
            height, width = ((16, 48), (40, 32))[index % 2]
            record["image"] = torch.full((3, height, width), record["image"][0, 0, 0].item())
        clicks[0]["image"] = clicks[0]["image"].tolist()
        reviews[0]["image"] = torch.full((3, 1080, 1920), 0.1)
        bundle = self.fit(clicks, reviews)
        self.assertTrue(torch.isfinite(bundle.predict([clicks[0]["image"]], [clicks[0]["upl"]])["click"]).all())

    def test_bounded_low_capacity_features_include_upl_interactions(self):
        images = torch.ones(2, 3, 32, 32)
        upl = torch.tensor([[0.0] * 4, [1.0] * 4])
        features = _features(images, upl)
        self.assertEqual(tuple(features.shape), (2, 64))
        self.assertTrue((features.abs() <= 1).all())
        torch.testing.assert_close(features[0, :12], features[1, :12])
        torch.testing.assert_close(features[0, 16:], -features[1, 16:])
        count = sum(parameter.numel() for model in (self.bundle.click_model, self.bundle.review_model)
                    for parameter in model.parameters())
        self.assertEqual(count, 585)

    def test_synthetic_opt_in_and_honest_human_provenance(self):
        with self.assertRaisesRegex(ValueError, "opt-in"):
            self.fit(allow_synthetic=False)
        self.assertTrue(self.bundle.metadata["fixtureOnly"])
        self.assertEqual(self.bundle.metadata["humanReviewCount"], 0)
        self.assertEqual(self.bundle.metadata["humanReviewTarget"], [100, 1000])
        clicks = [{**record, "source": "local-ledger"} for record in self.clicks]
        reviews = [{**record, "source": "operator"} for record in self.reviews]
        bundle = self.fit(clicks, reviews, allow_synthetic=False)
        self.assertFalse(bundle.metadata["fixtureOnly"])
        self.assertEqual(bundle.metadata["humanReviewCount"], len(reviews))
        for task in ("click", "review"):
            mixed_clicks, mixed_reviews = copy.deepcopy((clicks, reviews))
            (mixed_clicks if task == "click" else mixed_reviews)[0]["source"] = "synthetic"
            with self.assertRaisesRegex(ValueError, "opt-in"):
                self.fit(mixed_clicks, mixed_reviews, allow_synthetic=False)
            mixed = self.fit(mixed_clicks, mixed_reviews)
            self.assertTrue(mixed.metadata["fixtureOnly"])
            self.assertEqual(load_bundle(export_bundle(mixed)).id, mixed.id)

    def test_evidence_thresholds_and_missing_class_variation_fail(self):
        with self.assertRaisesRegex(ValueError, "groups"):
            self.fit(*fixture_records(3))
        with self.assertRaisesRegex(ValueError, "class variation"):
            self.fit([{**record, "label": 0} for record in self.clicks])
        heldout = self.bundle.split_profiles["validation"]
        clicks = [{**record, "label": 0} if record["profileId"] in heldout else record for record in self.clicks]
        with self.assertRaisesRegex(ValueError, "validation.*class variation"):
            self.fit(clicks)
        for field in ("quality", "brand"):
            with self.assertRaisesRegex(ValueError, "score variation"):
                self.fit(reviews=[{**record, field: 0.5} for record in self.reviews])
        for option in ("min_train_records", "min_validation_records", "min_train_profiles", "min_validation_profiles"):
            with self.subTest(option=option), self.assertRaises(ValueError):
                self.fit(**{option: 1000})
        sparse_clicks = self.clicks[::2]
        sparse_reviews = self.reviews[::2]
        bundle = self.fit(sparse_clicks, sparse_reviews, min_train_records=2, min_validation_records=2,
                          validation_fraction=0.5)
        self.assertEqual(len(bundle.split_profiles["validation"]), 6)

    def test_invalid_hyperparameters(self):
        for options in ({"seed": True}, {"seed": -1}, {"seed": 2 ** 32}, {"steps": 0},
                        {"steps": 1.5}, {"allow_synthetic": 1}, {"min_train_profiles": 1},
                        {"validation_fraction": float("nan")}, {"validation_fraction": 1},
                        {"validation_fraction": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.fit(**options)

    def test_invalid_record_contracts(self):
        changes = (("profileId", ""), ("creativeId", None), ("source", "operator"),
                   ("label", 0.3), ("label", True), ("label", float("nan")),
                   ("upl", [0.1] * 3), ("upl", [0, 0, 0, 1.1]),
                   ("image", torch.zeros(3, 15, 32)), ("image", torch.zeros(3, 16, 4097)),
                   ("image", torch.full((3, 16, 16), float("inf"))))
        for field, value in changes:
            with self.subTest(field=field, value=str(value)[:40]), self.assertRaises(ValueError):
                self.fit([{**self.clicks[0], field: value}, *self.clicks[1:]])
        for field, value in (("quality", -0.1), ("brand", 1.1), ("quality", None), ("source", "local-ledger")):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.fit(reviews=[{**self.reviews[0], field: value}, *self.reviews[1:]])
        for records in ([], None, [None]):
            with self.assertRaises(ValueError):
                fit_reward_bundle(records, self.reviews, allow_synthetic=True, steps=1)

    def test_invalid_prediction_contracts(self):
        for images in (torch.zeros(0, 3, 32, 32), torch.zeros(1, 4, 32, 32),
                       torch.zeros(3, 32, 32), torch.zeros(1, 3, 32, 15),
                       torch.zeros(1, 3, 4097, 16), torch.ones(1, 3, 16, 16) * 1.1,
                       torch.full((1, 3, 16, 16), float("nan")), None,
                       torch.zeros(1, 3, 16, 16, dtype=torch.bool),
                       torch.zeros(1, 3, 16, 16, dtype=torch.complex64)):
            with self.subTest(shape=getattr(images, "shape", None)), self.assertRaises(ValueError):
                self.bundle.predict(images, torch.zeros(1, 4))
        for upl in ([[0, 0, 0]], [[0, 0, 0, -0.1]], [[0, 0, 0, float("inf")]]):
            with self.assertRaises(ValueError):
                self.bundle.predict(torch.zeros(1, 3, 32, 32), upl)

    def test_json_roundtrip_exact_predictions_and_no_overwrite(self):
        images = torch.stack([record["image"] for record in self.clicks[:4]])
        upl = torch.tensor([record["upl"] for record in self.clicks[:4]])
        expected = self.bundle.predict(images, upl)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rewards.json"
            artifact = export_bundle(self.bundle, path)
            self.assertEqual(json.loads(path.read_text()), artifact)
            self.assertLess(path.stat().st_size, 30_000)
            with self.assertRaises(FileExistsError):
                export_bundle(self.bundle, path)
            for restored in (load_bundle(path), load_bundle(artifact)):
                self.assertEqual(restored.id, self.bundle.id)
                self.assertEqual(restored.metadata, self.bundle.metadata)
                self.assertEqual(restored.split_profiles, self.bundle.split_profiles)
                actual = restored.predict(images, upl)
                for name in expected:
                    self.assertTrue(torch.equal(expected[name], actual[name]))
                self.assertFalse(restored.click_model.training)
                self.assertFalse(restored.review_model.weight.requires_grad)

    def test_reject_corrupt_or_forged_artifacts(self):
        artifact = export_bundle(self.bundle)
        changed = copy.deepcopy(artifact)
        changed["weights"]["click"]["bias"][0][0] += 0.1
        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            load_bundle(changed)
        for invalid in ([], {**artifact, "extra": 1}, {**artifact, "format": "unknown"}):
            with self.assertRaises(ValueError):
                load_bundle(invalid)
        for value in (9.0, float("inf"), float("nan")):
            changed = copy.deepcopy(artifact)
            changed["weights"]["click"]["bias"][0][0] = value
            if torch.isfinite(torch.tensor(value)):
                reidentify(changed)
            with self.assertRaises(ValueError):
                load_bundle(changed)
        mutations = [
            lambda artifact: artifact["weights"]["review"]["weight"].pop(),
            lambda artifact: artifact["weights"]["click"].update({"extra": []}),
            lambda artifact: artifact["metadata"].update({"fixtureOnly": False}),
            lambda artifact: artifact["metadata"].update({"humanReviewCount": 999}),
            lambda artifact: artifact["metadata"].update({"criticType": "rl-value"}),
            lambda artifact: artifact["metadata"]["splitProfiles"].update({"validation": artifact["metadata"]["splitProfiles"]["train"]}),
            lambda artifact: artifact["metadata"]["metrics"]["quality"].update({"mse": -1}),
            lambda artifact: artifact["metadata"]["sources"]["click"].update({"train": {"operator": 36}}),
            lambda artifact: artifact["metadata"]["counts"]["click"]["train"].update({"records": 999}),
        ]
        for mutation in mutations:
            changed = copy.deepcopy(artifact)
            mutation(changed)
            with self.assertRaises(ValueError):
                load_bundle(reidentify(changed))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for payload in ('{"id": 1, "id": 2}', "not json", "[1,2]"):
                path.write_text(payload)
                with self.assertRaises(ValueError):
                    load_bundle(path)

    def test_metadata_is_copied_and_weight_mutation_blocks_export(self):
        bundle = load_bundle(export_bundle(self.bundle))
        bundle.split_profiles["train"].clear()
        bundle.metadata["fixtureOnly"] = False
        self.assertEqual(bundle.metadata, self.bundle.metadata)
        with torch.no_grad():
            bundle.click_model.bias.add_(0.1)
        with self.assertRaisesRegex(ValueError, "changed"):
            export_bundle(bundle)


if __name__ == "__main__":
    unittest.main()
