"""DDPO-style stochastic DDIM policy updates; not a flow-matching policy.

References: https://arxiv.org/abs/2305.13301 and https://arxiv.org/abs/1707.06347.
The native-32 backend is a numerical regression target, not the HD campaign model.
"""

import copy
import math
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional

from .diffusion import LEGACY_ENCODER, conditioning_tensor, composite_rgb, sample_ddim, seeded_noise, validate_schedule, validate_seed


@dataclass(frozen=True)
class PolicyConfig:
    rounds: int = 4
    group_size: int = 4
    epochs: int = 2
    learning_rate: float = 0.000001
    clip_range: float = 0.2
    kl_weight: float = 0.1
    maximum_kl: float = 0.01
    quality_floor: float = 0.6
    brand_floor: float = 0.8
    critic_penalty: float = 5.0
    uncertainty_penalty: float = 1.0
    anchor_weight: float = 0.1

    def validate(self):
        for name, maximum in (("rounds", 100), ("group_size", 16), ("epochs", 8)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"Invalid {name}")
        if self.group_size < 2:
            raise ValueError("At least two stochastic samples per profile are required")
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Nonfinite or boolean policy setting: {name}")
        if not 0 < self.learning_rate <= 0.001 or not 0 < self.clip_range <= 0.3:
            raise ValueError("Unbounded learning rate or PPO clip range")
        if not 0 < self.maximum_kl <= 1 or not 0 < self.kl_weight <= 100:
            raise ValueError("A positive bounded reference KL constraint is required")
        if not 0 <= self.quality_floor <= 1 or not 0 <= self.brand_floor <= 1:
            raise ValueError("Critic floors must be probabilities/normalized ratings")
        if not 1 <= self.critic_penalty <= 100 or not 0 <= self.uncertainty_penalty <= 10 or not 0 <= self.anchor_weight <= 10:
            raise ValueError("Invalid reward or anchoring weights")
        return self


def profiles_tensor(profiles, device, encoder=None):
    values = conditioning_tensor(profiles, encoder, device=device, batched=True)
    if not 1 <= len(values) <= 32:
        raise ValueError("Supply 1–32 UPLs matching the model encoder")
    return values


def transition(model, states, profiles, timestep, next_timestep, alphas, eta=1.0):
    prediction = model(states, profiles, torch.full((len(states),), timestep, device=states.device, dtype=torch.long))
    alpha = alphas[timestep]
    next_alpha = alphas[next_timestep] if next_timestep is not None else alphas.new_tensor(1)
    clean = ((states - (1 - alpha).sqrt() * prediction) / alpha.sqrt()).clamp(-1, 1)
    variance = (eta ** 2 * (1 - next_alpha) / (1 - alpha) * (1 - alpha / next_alpha)).clamp_min(0)
    mean = next_alpha.sqrt() * clean + (1 - next_alpha - variance).clamp_min(0).sqrt() * prediction
    return mean, variance, clean


def gaussian_log_probability(actions, means, variance):
    if not torch.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("Deterministic transitions have no Gaussian action density")
    return (-0.5 * ((actions - means).square() / variance + torch.log(2 * math.pi * variance))).flatten(1).sum(1)


def clipped_objective(new_log_probability, old_log_probability, advantages, clip_range):
    log_ratio = new_log_probability - old_log_probability
    ratio = log_ratio.clamp(-20, 20).exp()
    return -torch.minimum(ratio * advantages, ratio.clamp(1 - clip_range, 1 + clip_range) * advantages).mean()


@torch.no_grad()
def rollout(model, profiles, schedule, seed, terminal_model=None):
    validate_seed(seed)
    validate_schedule(schedule)
    device = next(model.parameters()).device
    profiles = profiles_tensor(profiles, device, model.encoder)
    if terminal_model is not None and terminal_model.encoder != model.encoder:
        raise ValueError("Terminal reference encoder must match the policy")
    states = torch.cat([seeded_noise((seed + index) % 2 ** 32) for index in range(len(profiles))]).to(device)
    generator = torch.Generator().manual_seed(seed)
    alphas = torch.tensor(schedule["alphaCumprod"], dtype=torch.float32, device=device)
    trajectories = []
    for index, timestep in enumerate(schedule["timesteps"]):
        next_timestep = schedule["timesteps"][index + 1] if index + 1 < len(schedule["timesteps"]) else None
        active_model = terminal_model if next_timestep is None and terminal_model is not None else model
        means, variance, clean = transition(active_model, states, profiles, timestep, next_timestep, alphas)
        if next_timestep is None:
            break
        actions = means + variance.sqrt() * torch.randn(means.shape, generator=generator).to(device)
        probabilities = gaussian_log_probability(actions, means, variance)
        if not torch.isfinite(actions).all() or not torch.isfinite(probabilities).all():
            raise ValueError("Nonfinite stochastic policy rollout")
        trajectories.append({"states": states, "actions": actions, "old_log_probability": probabilities, "timestep": timestep, "next_timestep": next_timestep})
        states = actions
    return (clean + 1) / 2, trajectories


def delivered_batch(images, brand):
    return torch.stack([composite_rgb(image[None], brand).permute(2, 0, 1) for image in images])


def constrained_reward(scores, config):
    conservative = {}
    for name in ("click", "quality", "brand"):
        values = scores[name]
        deviation = scores[f"{name}_std"]
        if values.ndim != 1 or deviation.shape != values.shape or not torch.isfinite(values).all() or not torch.isfinite(deviation).all():
            raise ValueError("Invalid reward-model output")
        if ((values < 0) | (values > 1)).any() or (deviation < 0).any():
            raise ValueError("Reward predictions must be bounded; uncertainty must be nonnegative")
        conservative[name] = (values - config.uncertainty_penalty * deviation).clamp(0, 1)
    penalty = functional.relu(config.quality_floor - conservative["quality"]) + functional.relu(config.brand_floor - conservative["brand"])
    return conservative["click"] - config.critic_penalty * penalty


def optimize_policy(parent, profiles, rewards, *, schedule, brand=None, config=None, seed=2026, anchor=None):
    config = (config or PolicyConfig()).validate()
    validate_seed(seed)
    validate_schedule(schedule)
    if getattr(rewards, "encoder", LEGACY_ENCODER) != parent.encoder:
        raise ValueError("Reward encoder must exactly match the policy encoder")
    model = copy.deepcopy(parent).eval()
    reference = copy.deepcopy(parent).eval().requires_grad_(False)
    device = next(model.parameters()).device
    profiles = profiles_tensor(profiles, device, parent.encoder)
    if len(profiles) * config.group_size > 32:
        raise ValueError("Bounded rollouts allow at most 32 total samples")
    repeated = profiles.repeat_interleave(config.group_size, dim=0)
    alphas = torch.tensor(schedule["alphaCumprod"], dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    initial = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    history = []
    for round_index in range(config.rounds):
        images, trajectories = rollout(model, repeated, schedule, (seed + round_index * 1009) % 2 ** 32, terminal_model=reference)
        scores = rewards.predict(delivered_batch(images, brand).cpu(), repeated.cpu())
        reward = constrained_reward(scores, config).detach().to(device)
        groups = reward.reshape(-1, config.group_size)
        other_mean = (groups.sum(1, keepdim=True) - groups) / (config.group_size - 1)
        advantages = (groups - other_mean).flatten()
        round_start = copy.deepcopy(model.state_dict())
        rejected = False
        for epoch in range(config.epochs):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for trajectory in trajectories:
                means, variance, _ = transition(model, trajectory["states"], repeated, trajectory["timestep"], trajectory["next_timestep"], alphas)
                new_probability = gaussian_log_probability(trajectory["actions"], means, variance)
                with torch.no_grad():
                    reference_mean, _, _ = transition(reference, trajectory["states"], repeated, trajectory["timestep"], trajectory["next_timestep"], alphas)
                kl = (means - reference_mean).square().mean() / (2 * variance)
                loss = (clipped_objective(new_probability, trajectory["old_log_probability"], advantages, config.clip_range) + config.kl_weight * kl) / len(trajectories)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite PPO objective")
                loss.backward()
                total_loss += loss.item()
            if anchor is not None and config.anchor_weight:
                anchor_loss = config.anchor_weight * anchor(model, round_index)
                if not torch.isfinite(anchor_loss):
                    raise ValueError("Nonfinite teacher anchor")
                anchor_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                divergence = []
                for trajectory in trajectories:
                    means, variance, _ = transition(model, trajectory["states"], repeated, trajectory["timestep"], trajectory["next_timestep"], alphas)
                    reference_mean, _, _ = transition(reference, trajectory["states"], repeated, trajectory["timestep"], trajectory["next_timestep"], alphas)
                    divergence.append(((means - reference_mean).square().mean() / (2 * variance)).item())
                measured_kl = sum(divergence) / len(divergence)
            if not math.isfinite(measured_kl) or measured_kl > config.maximum_kl:
                model.load_state_dict(round_start)
                rejected = True
            history.append({"round": round_index + 1, "epoch": epoch + 1, "surrogateReward": reward.mean().item(), "ppoLoss": total_loss, "gradientNorm": float(gradient_norm), "referenceKlPerCoordinate": measured_kl, "rolledBack": rejected})
            if rejected:
                break
        if rejected:
            break
    final = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
    if not torch.isfinite(final).all():
        raise ValueError("Policy produced invalid weights")
    return model, {"algorithm": "ddpo-style-ppo-leave-one-out-v1", "config": asdict(config), "history": history, "weightDeltaL2": (final - initial).norm().item(), "trainingSampler": "stochastic-DDIM-eta1-15-density-transitions-frozen-reference-terminal", "evaluationSampler": "candidate-deterministic-DDIM16-eta0", "klUnits": "mean conditional Gaussian KL per image coordinate on rollout states", "warning": "Optimizes learned click predictions, not observed candidate CTR. Training freezes the terminal decoder to define a parameter-independent reward map; deployment uses candidate weights at every step and is audited separately."}


@torch.no_grad()
def evaluate_candidate(parent, candidate, profiles, rewards, *, schedule, brand, config, seed=9000):
    if candidate.encoder != parent.encoder or getattr(rewards, "encoder", LEGACY_ENCODER) != parent.encoder:
        raise ValueError("Parent, candidate and reward encoders must match exactly")
    device = next(parent.parameters()).device
    profiles = profiles_tensor(profiles, device, parent.encoder)
    summaries = {}
    for name, model in (("parent", parent), ("candidate", candidate)):
        images = torch.cat([sample_ddim(model, profile, (seed + index) % 2 ** 32, schedule) for index, profile in enumerate(profiles)])
        scores = rewards.predict(delivered_batch(images, brand).cpu(), profiles.cpu())
        constrained_reward(scores, config)
        summaries[name] = {key: scores[key].detach().cpu().tolist() for key in ("click", "quality", "brand", "click_std", "quality_std", "brand_std")}
    baseline, proposed = summaries["parent"], summaries["candidate"]
    mean = lambda values: sum(values) / len(values)
    quality_min = min(value - config.uncertainty_penalty * uncertainty for value, uncertainty in zip(proposed["quality"], proposed["quality_std"]))
    brand_min = min(value - config.uncertainty_penalty * uncertainty for value, uncertainty in zip(proposed["brand"], proposed["brand_std"]))
    conservative_delta = mean(proposed["click"]) - mean(baseline["click"]) - config.uncertainty_penalty * (mean(proposed["click_std"]) + mean(baseline["click_std"]))
    gates = {
        "qualityFloor": quality_min >= config.quality_floor,
        "brandFloor": brand_min >= config.brand_floor,
        "qualityNonRegression": mean(proposed["quality"]) >= mean(baseline["quality"]),
        "brandNonRegression": mean(proposed["brand"]) >= mean(baseline["brand"]),
        "conservativePredictedClickGain": conservative_delta > 0,
    }
    return {"samples": len(profiles), "scores": summaries, "gates": gates, "eligibleForFurtherTesting": all(gates.values()), "conservativePredictedClickDelta": conservative_delta, "observedCtrLift": None, "promotion": "never-automatic", "limitation": "Ensemble spread is a heuristic, not a calibrated confidence interval. A learned critic gate is not a guarantee of brand correctness."}
