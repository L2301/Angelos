"""Fixed native-32 epsilon U-Net and explicit CPU/device DDIM reference.

The cosine schedule uses the standard .999 beta cap and a near-Gaussian terminal
prior. Its small terminal alpha amplifies untrained epsilon errors: DDIM clips
x0 at every step, not epsilon. Clipping is not evidence of good generated images.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from .teacher import apply_brand, pixel_coordinates


ARCHITECTURE = {"name": "conditional-unet-v1", "resolution": 32, "channels": [10, 16, 32, 32, 16, 3]}
LAYER_SHAPES = ((10, 16, 1), (16, 32, 2), (32, 32, 1), (48, 16, 1), (16, 3, 1))


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
        raise ValueError("Seed must be a uint32 integer")
    return seed


def seeded_noise(seed=2026, shape=(1, 3, 32, 32)):
    """LCG/Box-Muller cos-then-sin, each value rounded once to float32."""
    state = validate_seed(seed)
    count = math.prod(shape)
    values = np.empty(count, dtype=np.float32)
    for offset in range(0, count, 2):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        first = (state + 0.5) / 2 ** 32
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        second = (state + 0.5) / 2 ** 32
        radius = math.sqrt(-2 * math.log(first))
        angle = 2 * math.pi * second
        values[offset] = radius * math.cos(angle)
        if offset + 1 < count:
            values[offset + 1] = radius * math.sin(angle)
    return torch.from_numpy(values.reshape(shape))


def cosine_schedule():
    """64 cosine DDPM steps (offset .008, beta cap .999), 16 DDIM steps."""
    positions = np.arange(65, dtype=np.float64) / 64
    cumulative = np.cos((positions + 0.008) / 1.008 * math.pi / 2) ** 2
    cumulative /= cumulative[0]
    betas = np.minimum(1 - cumulative[1:] / cumulative[:-1], 0.999)
    alphas = np.cumprod(1 - betas).astype(np.float32)
    return {
        "predictionType": "epsilon",
        "alphaCumprod": alphas.tolist(),
        "timesteps": np.rint(np.linspace(63, 0, 16)).astype(int).tolist(),
    }


def validate_schedule(schedule):
    if schedule.get("predictionType") != "epsilon":
        raise ValueError("Only epsilon prediction is supported")
    alphas = np.asarray(schedule.get("alphaCumprod"), dtype=np.float32)
    timesteps = schedule.get("timesteps", [])
    if alphas.shape != (64,) or not np.isfinite(alphas).all() or not ((alphas > 0) & (alphas < 1)).all() or not (np.diff(alphas) < 0).all():
        raise ValueError("Schedule needs 64 strictly decreasing alphas in (0,1)")
    if len(timesteps) != 16 or any(type(step) is not int for step in timesteps) or timesteps[0] != 63 or timesteps[-1] != 0 or any(left <= right for left, right in zip(timesteps, timesteps[1:])):
        raise ValueError("Schedule needs 16 descending integer timesteps from 63 to 0")
    return schedule


class DiffusionStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1) for inputs, outputs, stride in LAYER_SHAPES)
        coordinates = pixel_coordinates(32, 32).T.reshape(1, 2, 32, 32)
        self.register_buffer("coordinates", coordinates, persistent=False)

    def forward(self, noisy, upl, timesteps, return_nodes=False):
        batch = noisy.shape[0]
        if noisy.shape != (batch, 3, 32, 32) or upl.shape != (batch, 4) or timesteps.shape != (batch,):
            raise ValueError("Expected noisy [N,3,32,32], upl [N,4], timesteps [N]")
        conditioning = upl[:, :, None, None].expand(-1, -1, 32, 32)
        time_plane = (timesteps.to(dtype=noisy.dtype) / 63)[:, None, None, None].expand(-1, 1, 32, 32)
        inputs = torch.cat((noisy, conditioning, time_plane, self.coordinates.expand(batch, -1, -1, -1)), dim=1)
        conv0 = functional.relu(self.layers[0](inputs))
        conv1 = functional.relu(self.layers[1](conv0))
        conv2 = functional.relu(self.layers[2](conv1))
        skip = torch.cat((functional.interpolate(conv2, scale_factor=2, mode="nearest"), conv0), dim=1)
        conv3 = functional.relu(self.layers[3](skip))
        epsilon = self.layers[4](conv3)
        if return_nodes:
            return epsilon, {"input": inputs, "conv0": conv0, "conv1": conv1, "conv2": conv2, "skip": skip, "conv3": conv3, "epsilon": epsilon}
        return epsilon

    def export_layers(self):
        result = []
        for index, layer in enumerate(self.layers):
            if not all((torch.isfinite(parameter) & (parameter.abs() <= 1000)).all().item() for parameter in layer.parameters()):
                raise ValueError("Diffusion weights must be finite and bounded by 1000")
            result.append({
                "input": layer.in_channels, "output": layer.out_channels,
                "kernel": 3, "stride": layer.stride[0],
                "activation": "linear" if index == 4 else "relu",
                "weights": layer.weight.detach().cpu().reshape(-1).tolist(),
                "bias": layer.bias.detach().cpu().tolist(),
            })
        return result

    @classmethod
    def from_artifact(cls, artifact, device="cpu"):
        if artifact.get("format") != "angelos.diffusion.v1" or artifact.get("architecture") != ARCHITECTURE:
            raise ValueError("Unsupported diffusion artifact architecture")
        validate_schedule(artifact["schedule"])
        if len(artifact["layers"]) != 5:
            raise ValueError("Expected five convolution layers")
        model = cls()
        with torch.no_grad():
            for index, (layer, exported) in enumerate(zip(model.layers, artifact["layers"])):
                expected = {"input": layer.in_channels, "output": layer.out_channels, "kernel": 3, "stride": layer.stride[0], "activation": "linear" if index == 4 else "relu"}
                if any(exported.get(key) != value for key, value in expected.items()):
                    raise ValueError("Invalid diffusion layer contract")
                weights = torch.tensor(exported["weights"], dtype=torch.float32)
                bias = torch.tensor(exported["bias"], dtype=torch.float32)
                if weights.shape != (layer.weight.numel(),) or bias.shape != layer.bias.shape or not torch.isfinite(weights).all() or not torch.isfinite(bias).all():
                    raise ValueError("Invalid diffusion parameter shape or finite values")
                layer.weight.copy_(weights.reshape_as(layer.weight))
                layer.bias.copy_(bias)
        return model.to(device).eval()


@torch.no_grad()
def sample_ddim(model, upl, seed=2026, schedule=None, return_trace=False):
    schedule = validate_schedule(cosine_schedule() if schedule is None else schedule)
    device = next(model.parameters()).device
    profile = torch.as_tensor(upl, dtype=torch.float32, device=device).reshape(-1, 4)
    if profile.shape != (1, 4) or not torch.isfinite(profile).all() or ((profile < 0) | (profile > 1)).any():
        raise ValueError("Sampling requires one finite UPL in [0,1]^4")
    noisy = seeded_noise(seed).to(device)
    alphas = torch.tensor(schedule["alphaCumprod"], dtype=torch.float32, device=device)
    trace = []
    for index, timestep in enumerate(schedule["timesteps"]):
        time_tensor = torch.tensor([timestep], device=device)
        if return_trace and index == 0:
            epsilon, nodes = model(noisy, profile, time_tensor, return_nodes=True)
            node_checks = {name: {"shape": list(value.shape), "firstValues": value.flatten()[:16].cpu().tolist()} for name, value in nodes.items()}
        else:
            epsilon = model(noisy, profile, time_tensor)
        alpha = alphas[timestep]
        clean = ((noisy - (1 - alpha).sqrt() * epsilon) / alpha.sqrt()).clamp(-1, 1)
        next_alpha = alphas[schedule["timesteps"][index + 1]] if index < 15 else alphas.new_tensor(1)
        noisy = next_alpha.sqrt() * clean + (1 - next_alpha).sqrt() * epsilon
        if return_trace:
            trace.append({"timestep": timestep, "epsilon": epsilon.flatten()[:16].cpu().tolist(), "x0": clean.flatten()[:16].cpu().tolist(), "nextSample": noisy.flatten()[:16].cpu().tolist()})
    result = (clean + 1) / 2
    if not torch.isfinite(result).all():
        raise ValueError("Non-finite DDIM sample")
    if return_trace:
        return result, {"initialNoise": seeded_noise(seed).flatten()[:16].tolist(), "nodes": node_checks, "steps": trace}
    return result


def composite_rgb(rgb, brand, width=32, height=32):
    """Scale the native image, then apply the exact plate at requested pixels."""
    coordinates = pixel_coordinates(width, height, rgb.device)
    scaled = functional.interpolate(rgb, size=(height, width), mode="nearest")
    colors = scaled[0].permute(1, 2, 0).reshape(-1, 3)
    if brand is not None and brand.get("mode") == "composite":
        colors = apply_brand(colors, coordinates, brand)
    return colors.reshape(height, width, 3)


def render_cpu(artifact, upl, width=32, height=32, composite=True, seed=2026):
    """Explicit numerical reference only; not a browser GPU implementation."""
    model = DiffusionStudent.from_artifact(artifact)
    rgb = sample_ddim(model, upl, seed, artifact["schedule"])
    colors = composite_rgb(rgb, artifact.get("brand") if composite else None, width, height)
    return torch.cat((colors, torch.ones((height, width, 1))), dim=-1)
