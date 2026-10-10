"""Offline SCM/LoRA regression with tiny real Diffusers modules, not HD evidence."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from diffusers import AutoencoderDC, DDIMScheduler, SanaSprintPipeline, SanaTransformer2DModel, SCMScheduler
from safetensors.torch import load_file

from angelos_ml import hd_alignment
from angelos_ml.hd_alignment import AlignmentConfig, adapter_scope, align_pipeline
from angelos_ml.rewards import RewardBundle, fit_reward_bundle


def tiny_pipeline():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(84)
        transformer = SanaTransformer2DModel(
            in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=8,
            num_layers=1, num_cross_attention_heads=2, cross_attention_head_dim=8,
            cross_attention_dim=16, caption_channels=16, mlp_ratio=1, sample_size=16,
            guidance_embeds=True,
        )
        decoder = AutoencoderDC(
            latent_channels=4, encoder_block_out_channels=(8, 8), decoder_block_out_channels=(8, 8),
            encoder_layers_per_block=(1, 1), decoder_layers_per_block=(1, 1),
            encoder_qkv_multiscales=((), ()), decoder_qkv_multiscales=((), ()),
        )
    pipeline = SanaSprintPipeline(tokenizer=None, text_encoder=None, transformer=transformer,
                                  vae=decoder, scheduler=SCMScheduler())
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def cases():
    generator = torch.Generator(device="cpu").manual_seed(912)
    result = []
    for index, part in enumerate(("train", "audit")):
        result.append({"id": part, "profileId": f"new-{part}", "prompt": f"fictional {part} landscape",
                       "upl": [0.2, 0.3, 0.4, 0.5], "seed": 100 + index,
                       "prompt_embeds": torch.randn(1, 4, 16, generator=generator),
                       "prompt_attention_mask": torch.ones(1, 4, dtype=torch.long)})
    return [result[0]], [result[1]]


def reward_fixture():
    clicks, reviews = [], []
    for profile in range(12):
        for creative in range(4):
            label = creative // 2
            shared = {"profileId": f"reward-{profile}", "creativeId": f"creative-{profile}-{creative}",
                      "upl": [profile / 12, 0.2, 0.6, 0.8], "source": "synthetic",
                      "image": torch.full((3, 32, 32), 0.1 + 0.7 * label + 0.01 * (profile % 3))}
            clicks.append({**shared, "label": label})
            reviews.append({**shared, "quality": 0.15 + 0.7 * label, "brand": float(label)})
    return fit_reward_bundle(clicks, reviews, steps=200, allow_synthetic=True)


class HDAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.rewards = reward_fixture()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.pipeline = tiny_pipeline()
        self.training, self.audit = cases()
        self.config = AlignmentConfig(steps=2, blocks=1, fixture_only=True)

    def test_unwrapped_sampler_and_decode_match_official_pipeline(self):
        case = self.training[0]
        prepared = {**case, "embeddings": (case["prompt_embeds"], case["prompt_attention_mask"])}
        for sampling_steps in (1, 2, 4):
            config = replace(self.config, sampling_steps=sampling_steps)
            with torch.no_grad():
                expected = self.pipeline(
                    prompt_embeds=case["prompt_embeds"], prompt_attention_mask=case["prompt_attention_mask"],
                    width=32, height=32, use_resolution_binning=False, num_inference_steps=sampling_steps,
                    max_timesteps=1.5708, intermediate_timesteps=1.3 if sampling_steps == 2 else None,
                    guidance_scale=4.5, complex_human_instruction=None, clean_caption=False,
                    output_type="pt", generator=torch.Generator(device="cpu").manual_seed(case["seed"]),
                ).images
                _, actual = hd_alignment._sample(self.pipeline, prepared, config)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_scm_decoder_gradient_agrees_with_finite_difference(self):
        case = self.training[0]
        prepared = {**case, "embeddings": (case["prompt_embeds"], case["prompt_attention_mask"])}
        self.pipeline.vae.eval().requires_grad_(False)
        with adapter_scope(self.pipeline.transformer, self.config, seed=15) as adapters:
            parameter = next(iter(adapters.values())).up
            _, image = hd_alignment._sample(self.pipeline, prepared, self.config)
            gradient = torch.autograd.grad(image.mean(), parameter)[0]
            offset = gradient.abs().argmax().item()
            analytic = gradient.flatten()[offset].item()
            self.assertNotEqual(analytic, 0)
            epsilon = 0.002
            with torch.no_grad():
                parameter.flatten()[offset] += epsilon
                _, positive = hd_alignment._sample(self.pipeline, prepared, self.config)
                parameter.flatten()[offset] -= 2 * epsilon
                _, negative = hd_alignment._sample(self.pipeline, prepared, self.config)
                parameter.flatten()[offset] += epsilon
            measured = (positive.mean() - negative.mean()).item() / (2 * epsilon)
            self.assertAlmostEqual(analytic, measured, delta=max(abs(analytic) * 0.08, 1e-5))

    def test_adapter_changes_base_is_unchanged_and_audit_has_no_gradients(self):
        before = {name: value.clone() for name, value in self.pipeline.transformer.state_dict().items()}
        decoder_before = {name: value.clone() for name, value in self.pipeline.vae.state_dict().items()}
        flags = [parameter.requires_grad for parameter in self.pipeline.transformer.parameters()]
        calls = []
        original = hd_alignment._sample

        def traced(pipeline, case, config):
            calls.append((case["id"], torch.is_grad_enabled()))
            return original(pipeline, case, config)

        with patch.object(hd_alignment, "_sample", side_effect=traced):
            result = align_pipeline(self.pipeline, self.training, self.audit, self.rewards, config=self.config)
        self.assertGreater(result["report"]["adapterDeltaL2"], 0)
        self.assertTrue(any(value.abs().sum() > 0 for name, value in result["adapter_state"].items() if name.endswith(".up")))
        self.assertTrue(all(torch.equal(value, self.pipeline.transformer.state_dict()[name]) for name, value in before.items()))
        self.assertTrue(all(torch.equal(value, self.pipeline.vae.state_dict()[name]) for name, value in decoder_before.items()))
        self.assertEqual(flags, [parameter.requires_grad for parameter in self.pipeline.transformer.parameters()])
        self.assertTrue(all(parameter.grad is None for parameter in self.pipeline.transformer.parameters()))
        self.assertEqual(calls, [("train", False), ("train", True)] * 2 + [("audit", False)] * 2)
        self.assertIs(type(self.pipeline.transformer.transformer_blocks[0].attn1.to_q), nn.Linear)
        self.assertTrue(result["report"]["fixtureOnly"])
        self.assertFalse(result["report"]["autoPromoted"])
        self.assertFalse(result["report"]["hdQualityEstablished"])
        self.assertFalse(result["report"]["deploymentEligible"])
        self.assertFalse(any(result["report"]["deploymentGates"].values()))
        self.assertEqual(result["report"]["rewardResolutionCoverage"], "unverified")
        self.assertIsNone(result["report"]["observedCtrLift"])

    def test_official_tiled_decoder_preserves_adapter_gradients(self):
        config = replace(self.config, fixture_size=64)
        self.pipeline.vae.eval().requires_grad_(False)
        self.pipeline.vae.enable_tiling(tile_sample_min_height=32, tile_sample_min_width=32,
                                        tile_sample_stride_height=28, tile_sample_stride_width=28)
        case = self.training[0]
        prepared = {**case, "embeddings": (case["prompt_embeds"], case["prompt_attention_mask"])}
        with adapter_scope(self.pipeline.transformer, config) as adapters:
            _, image = hd_alignment._sample(self.pipeline, prepared, config)
            self.assertEqual(image.shape, (1, 3, 64, 64))
            gradient = torch.autograd.grad(image.mean(), next(iter(adapters.values())).up)[0]
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_audit_changes_do_not_select_or_change_trained_weights(self):
        first = align_pipeline(self.pipeline, self.training, self.audit, self.rewards, config=self.config)
        changed = [{**self.audit[0], "prompt": "different heldout scene", "seed": 707,
                    "prompt_embeds": -self.audit[0]["prompt_embeds"]}]
        second = align_pipeline(self.pipeline, self.training, changed, self.rewards, config=self.config)
        for name in first["adapter_state"]:
            torch.testing.assert_close(first["adapter_state"][name], second["adapter_state"][name], atol=0, rtol=0)

    def test_quality_regression_rejects_candidate(self):
        original = hd_alignment._scores
        audited = []

        def declining(rewards, image, case, config):
            values = original(rewards, image, case, config)
            if case["id"] == "audit":
                values["quality"] = torch.tensor([0.9 if not audited else 0.0])
                audited.append(case["id"])
            return values

        with patch.object(hd_alignment, "_scores", side_effect=declining):
            result = align_pipeline(self.pipeline, self.training, self.audit, self.rewards, config=self.config)
        self.assertEqual(result["report"]["status"], "rejected")
        self.assertFalse(result["report"]["gates"]["qualityNonRegression"])
        self.assertFalse(result["report"]["eligibleForManualReview"])

    def test_bad_critic_or_missing_human_evidence_blocks_before_sampling(self):
        metadata = self.rewards.metadata
        metadata["metrics"]["quality"]["mse"] = 1.0
        bad = RewardBundle(self.rewards.click_model, self.rewards.review_model, metadata)
        with patch.object(hd_alignment, "_sample") as sampler:
            with self.assertRaisesRegex(ValueError, "quality.mse"):
                align_pipeline(self.pipeline, self.training, self.audit, bad, config=self.config)
            with self.assertRaisesRegex(ValueError, "observed clicks and human"):
                align_pipeline(self.pipeline, self.training, self.audit, self.rewards,
                               config=replace(self.config, fixture_only=False))
            sampler.assert_not_called()

    def test_operator_review_count_boundary(self):
        """Metadata-only boundary fixtures are not real observed campaign evidence."""
        config = replace(self.config, fixture_only=False)
        for count in (99, 100):
            metadata = self.rewards.metadata
            metadata.update(fixtureOnly=False, humanReviewCount=count)
            for part, review_count in (("train", count - 12), ("validation", 12)):
                metadata["sources"]["click"][part] = {"local-ledger": metadata["counts"]["click"][part]["records"]}
                metadata["sources"]["review"][part] = {"operator": review_count}
                metadata["counts"]["review"][part]["records"] = review_count
            bundle = RewardBundle(self.rewards.click_model, self.rewards.review_model, metadata)
            if count < 100:
                with self.assertRaisesRegex(ValueError, "at least 100 operator reviews"):
                    hd_alignment._reward_evidence(bundle, config)
            else:
                self.assertEqual(hd_alignment._reward_evidence(bundle, config)["metadata"]["humanReviewCount"], 100)

    def test_detached_reward_cannot_pass_as_reward_training(self):
        original = self.rewards.predict

        def detached(images, upl):
            return {name: value.detach() for name, value in original(images, upl).items()}

        with patch.object(self.rewards, "predict", side_effect=detached):
            with self.assertRaisesRegex(ValueError, "gradients are detached"):
                align_pipeline(self.pipeline, self.training, self.audit, self.rewards, config=self.config)
        self.assertIs(type(self.pipeline.transformer.transformer_blocks[0].attn1.to_q), nn.Linear)

    def test_artifact_roundtrip_and_non_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            result = align_pipeline(self.pipeline, self.training, self.audit, self.rewards,
                                    config=self.config, output=output)
            saved = load_file(output / "adapter.safetensors")
            report = json.loads((output / "alignment.json").read_text())
            self.assertEqual(report["format"], hd_alignment.FORMAT)
            self.assertFalse(report["likelihoodRatioRL"])
            for name, value in saved.items():
                torch.testing.assert_close(value, result["adapter_state"][name], atol=0, rtol=0)
            case = {**self.audit[0], "embeddings": (self.audit[0]["prompt_embeds"], self.audit[0]["prompt_attention_mask"])}
            images = []
            for state in (saved, result["adapter_state"]):
                with adapter_scope(self.pipeline.transformer, self.config, state=state), torch.no_grad():
                    images.append(hd_alignment._sample(self.pipeline, case, self.config)[1])
            torch.testing.assert_close(*images, atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "new directory"):
                align_pipeline(self.pipeline, self.training, self.audit, self.rewards,
                               config=self.config, output=output)

    def test_limits_group_leakage_and_wrong_scheduler(self):
        for config in (replace(self.config, steps=21), replace(self.config, steps=True),
                       replace(self.config, learning_rate=float("nan")), replace(self.config, rank=9)):
            with self.assertRaises(ValueError):
                align_pipeline(self.pipeline, self.training, self.audit, self.rewards, config=config)
        for field in ("id", "profileId", "seed"):
            audit = [{**self.audit[0], field: self.training[0][field]}]
            with self.assertRaisesRegex(ValueError, "disjoint"):
                align_pipeline(self.pipeline, self.training, audit, self.rewards, config=self.config)
        leaking = [{**self.training[0], "profileId": self.rewards.split_profiles["validation"][0]}]
        with self.assertRaisesRegex(ValueError, "validation profiles"):
            align_pipeline(self.pipeline, leaking, self.audit, self.rewards, config=self.config)
        self.pipeline.scheduler = DDIMScheduler()
        with self.assertRaisesRegex(ValueError, "official SanaSprint"):
            align_pipeline(self.pipeline, self.training, self.audit, self.rewards, config=self.config)

    def test_oom_propagates_without_fallback_and_restores_parent(self):
        flags = [parameter.requires_grad for parameter in self.pipeline.transformer.parameters()]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            with patch.object(hd_alignment, "_sample", side_effect=RuntimeError("out of memory")) as sampler:
                with self.assertRaisesRegex(RuntimeError, "out of memory"):
                    align_pipeline(self.pipeline, self.training, self.audit, self.rewards,
                                   config=self.config, output=output)
                self.assertEqual(sampler.call_count, 1)
            self.assertFalse(output.exists())
        self.assertEqual(flags, [parameter.requires_grad for parameter in self.pipeline.transformer.parameters()])
        self.assertIs(type(self.pipeline.transformer.transformer_blocks[0].attn1.to_q), nn.Linear)


if __name__ == "__main__":
    unittest.main()
