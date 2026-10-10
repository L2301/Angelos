import copy
import unittest

import torch

from angelos_ml.diffusion import DiffusionStudent, cosine_schedule, sample_ddim, seeded_noise
from angelos_ml.rl_policy import PolicyConfig, clipped_objective, constrained_reward, evaluate_candidate, gaussian_log_probability, optimize_policy, rollout, transition


class SyntheticRewardFixture:
    def predict(self, images, upl):
        brightness = images.mean((1, 2, 3))
        return {"click": brightness, "quality": brightness * 0 + 0.9, "brand": brightness * 0 + 0.95,
                "click_std": brightness * 0, "quality_std": brightness * 0, "brand_std": brightness * 0}


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        with torch.random.fork_rng():
            torch.manual_seed(4)
            return DiffusionStudent().eval()

    def test_density_and_ppo_clipping(self):
        density = gaussian_log_probability(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor(1.0))
        self.assertAlmostEqual(density.item(), -1.837877, places=5)
        with self.assertRaises(ValueError):
            gaussian_log_probability(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor(0.0))
        loss = clipped_objective(torch.tensor([2.0]).log(), torch.zeros(1), torch.ones(1), 0.2)
        self.assertAlmostEqual(loss.item(), -1.2, places=5)
        negative = clipped_objective(torch.tensor([0.5]).log(), torch.zeros(1), -torch.ones(1), 0.2)
        self.assertAlmostEqual(negative.item(), 0.8, places=5)

    def test_transition_matches_existing_deployment_at_eta_zero(self):
        model = self.model()
        profile = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
        schedule = cosine_schedule()
        alphas = torch.tensor(schedule["alphaCumprod"])
        state = seeded_noise(42)
        with torch.no_grad():
            for index, timestep in enumerate(schedule["timesteps"]):
                following = schedule["timesteps"][index + 1] if index < 15 else None
                state, _, clean = transition(model, state, profile, timestep, following, alphas, eta=0)
        self.assertTrue(torch.allclose((clean + 1) / 2, sample_ddim(model, profile, 42), atol=0, rtol=0))

    def test_rollouts_replay_and_policy_weights_really_change(self):
        model = self.model()
        schedule = cosine_schedule()
        profiles = [[0.2, 0.3, 0.4, 0.5]]
        first, trajectory = rollout(model, profiles, schedule, 42)
        second, _ = rollout(model, profiles, schedule, 42)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(trajectory), 15)
        self.assertTrue(all(not item["old_log_probability"].requires_grad for item in trajectory))
        before = copy.deepcopy(model.state_dict())
        candidate, report = optimize_policy(model, profiles, SyntheticRewardFixture(), schedule=schedule,
                                           config=PolicyConfig(rounds=2, epochs=1, group_size=2), seed=42)
        self.assertGreater(report["weightDeltaL2"], 0)
        self.assertTrue(any(not torch.equal(value, candidate.state_dict()[key]) for key, value in before.items()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))

    def test_quality_is_not_purchasable_with_predicted_clicks(self):
        config = PolicyConfig(quality_floor=0.8, brand_floor=0.9)
        scores = {"click": torch.tensor([1.0, 0.3]), "quality": torch.tensor([0.0, 0.9]), "brand": torch.tensor([0.0, 0.95]),
                  "click_std": torch.zeros(2), "quality_std": torch.zeros(2), "brand_std": torch.zeros(2)}
        reward = constrained_reward(scores, config)
        self.assertLess(reward[0], reward[1])
        model = self.model()
        result = evaluate_candidate(model, model, [[0.2, 0.3, 0.4, 0.5]], SyntheticRewardFixture(),
                                    schedule=cosine_schedule(), brand=None, config=PolicyConfig(brand_floor=0.99))
        self.assertFalse(result["gates"]["brandFloor"])
        self.assertFalse(result["eligibleForFurtherTesting"])
        self.assertIsNone(result["observedCtrLift"])

    def test_frozen_terminal_decoder_changes_only_the_reward_mapping(self):
        model = self.model()
        terminal = copy.deepcopy(model)
        with torch.no_grad():
            for parameter in terminal.parameters():
                parameter.zero_()
        profiles = [[0.2, 0.3, 0.4, 0.5]]
        original, first = rollout(model, profiles, cosine_schedule(), 42)
        decoded, second = rollout(model, profiles, cosine_schedule(), 42, terminal_model=terminal)
        self.assertFalse(torch.equal(original, decoded))
        for before, after in zip(first, second):
            self.assertTrue(torch.equal(before["states"], after["states"]))
            self.assertTrue(torch.equal(before["actions"], after["actions"]))
            self.assertTrue(torch.equal(before["old_log_probability"], after["old_log_probability"]))

    def test_trust_bound_rolls_back_an_update(self):
        model = self.model()
        before = copy.deepcopy(model.state_dict())
        candidate, report = optimize_policy(model, [[0.2, 0.3, 0.4, 0.5]], SyntheticRewardFixture(),
                                           schedule=cosine_schedule(), seed=42,
                                           config=PolicyConfig(rounds=2, epochs=1, group_size=2,
                                                               learning_rate=0.001, maximum_kl=1e-12))
        self.assertTrue(report["history"][-1]["rolledBack"])
        self.assertEqual(report["weightDeltaL2"], 0)
        self.assertTrue(all(torch.equal(value, candidate.state_dict()[key]) for key, value in before.items()))

    def test_invalid_settings_and_profiles_fail(self):
        for config in (PolicyConfig(rounds=True), PolicyConfig(group_size=1), PolicyConfig(learning_rate=float("nan")), PolicyConfig(maximum_kl=0)):
            with self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            rollout(self.model(), [[float("nan"), 0, 0, 0]], cosine_schedule(), 1)


if __name__ == "__main__":
    unittest.main()
