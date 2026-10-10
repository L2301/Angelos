import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from angelos_ml.diffusion import ARCHITECTURE, DiffusionStudent, cosine_schedule
from angelos_ml.feedback import artifact_bank, materialize_feedback


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        torch.set_num_threads(2)
        artifact = {"format": "angelos.diffusion.v1", "id": "student-1", "campaignId": "aurora", "graphHash": "b" * 64,
                    "encoder": {"id": "structured-demo-v1", "dimension": 4}, "architecture": ARCHITECTURE,
                    "layers": DiffusionStudent().export_layers(), "schedule": cosine_schedule(), "brand": None}
        path = Path(self.directory.name) / "parent.json"
        payload = json.dumps(artifact).encode()
        path.write_bytes(payload)
        self.bank = artifact_bank([path])
        descriptor = {key: artifact[key] for key in ("id", "campaignId", "graphHash", "encoder")}
        descriptor["hash"] = hashlib.sha256(payload).hexdigest()
        seed = 17
        identity = json.dumps([artifact["id"], "snapshot-1", seed], separators=(",", ":"))
        creative = f"creative:{hashlib.sha256(identity.encode()).hexdigest()}:{seed}"
        shared = {"artifact": descriptor, "campaignId": "aurora", "encoder": artifact["encoder"], "upl": [0.2, 0.3, 0.4, 0.5], "profileId": "profile-1", "snapshotId": "snapshot-1"}
        self.document = {"format": "angelos.feedback.v1", "campaignId": "aurora", "asOf": 1000, "outcome": "click", "population": "local-synthetic-adults",
                         "clicks": [{**shared, "impressionId": "impression-1", "label": 1, "maturesAt": 999, "context": {"creativeId": creative}}],
                         "reviews": [{**shared, "id": "review-1", "receivedAt": 800, "source": "operator", "creativeId": creative, "qualityRating": 4, "brandPass": True}]}

    def test_exact_frozen_creative_is_replayed_once(self):
        with patch("angelos_ml.feedback.sample_ddim", return_value=torch.full((1, 3, 32, 32), 0.5)) as render:
            clicks, reviews = materialize_feedback(self.document, self.bank)
        self.assertEqual(render.call_count, 1)
        self.assertEqual(render.call_args.args[1:3], ([0.2, 0.3, 0.4, 0.5], 17))
        self.assertEqual(clicks[0]["label"], 1)
        self.assertEqual(reviews[0]["quality"], 0.75)
        self.assertEqual(reviews[0]["source"], "operator")

    def test_tampered_or_immature_evidence_fails(self):
        faults = ("hash", "seed", "maturity", "campaign", "duplicate", "future-review", "empty")
        for fault in faults:
            document = copy.deepcopy(self.document)
            if fault == "hash": document["clicks"][0]["artifact"]["hash"] = "0" * 64
            if fault == "seed": document["clicks"][0]["context"]["creativeId"] += "1"
            if fault == "maturity": document["clicks"][0]["maturesAt"] = 1001
            if fault == "campaign": document["campaignId"] = "elsewhere"
            if fault == "duplicate": document["clicks"].append(document["clicks"][0])
            if fault == "future-review": document["reviews"][0]["receivedAt"] = 1001
            if fault == "empty": document["reviews"] = []
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                materialize_feedback(document, self.bank)

    def test_valid_feedback_cannot_train_a_different_campaign_or_encoder(self):
        parent = self.bank["student-1"]["artifact"]
        for change in ({"campaignId": "another-campaign"}, {"encoder": {"id": "other", "dimension": 4}}):
            with self.subTest(change=change), patch("angelos_ml.feedback.sample_ddim") as render:
                with self.assertRaisesRegex(ValueError, "does not match candidate parent"):
                    materialize_feedback(self.document, self.bank, parent={**parent, **change})
                render.assert_not_called()
        with patch("angelos_ml.feedback.sample_ddim", return_value=torch.full((1, 3, 32, 32), 0.5)):
            clicks, reviews = materialize_feedback(self.document, self.bank, parent=parent)
        self.assertEqual(len(clicks), 1)
        self.assertEqual(len(reviews), 1)


if __name__ == "__main__":
    unittest.main()
