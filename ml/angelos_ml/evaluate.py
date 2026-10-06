"""Pure-neural held-out evaluation, explicit compositor comparison, and fixtures."""

import torch

from .teacher import apply_brand, brand_masks, brand_probe_coordinates, pixel_coordinates


BRAND_TOLERANCE = 0.06


@torch.no_grad()
def evaluate(model, teacher, profiles, width=128, height=96, device="cpu"):
    coordinates = pixel_coordinates(width, height, device)
    plate = None
    probes = None
    if teacher.brand is not None:
        plate, _ = brand_masks(coordinates, teacher.brand)
        probes = brand_probe_coordinates(teacher.brand, device)
    per_profile_mse = []
    per_profile_mae = []
    per_profile_composite_mse = []
    brand_maximum = 0.0
    brand_squared_error = 0.0
    brand_channel_count = 0
    composite_maximum = 0.0
    for profile in profiles:
        upl = profile.to(device).expand(len(coordinates), -1)
        expected = teacher(upl, coordinates, validate=False)
        predicted = model(torch.cat((coordinates, upl), dim=1))
        difference = predicted - expected
        per_profile_mse.append(difference.square().mean().item())
        per_profile_mae.append(difference.abs().mean().item())
        if teacher.brand is None:
            per_profile_composite_mse.append(per_profile_mse[-1])
        else:
            composite = apply_brand(predicted, coordinates, teacher.brand)
            composite_difference = composite - expected
            per_profile_composite_mse.append(composite_difference.square().mean().item())
            probe_upl = profile.to(device).expand(len(probes), -1)
            probe_expected = teacher(probe_upl, probes, validate=False)
            probe_predicted = model(torch.cat((probes, probe_upl), dim=1))
            errors = torch.cat((difference[plate].reshape(-1), (probe_predicted - probe_expected).reshape(-1)))
            brand_maximum = max(brand_maximum, errors.abs().max().item())
            brand_squared_error += errors.square().sum().item()
            brand_channel_count += errors.numel()
            probe_composite = apply_brand(probe_predicted, probes, teacher.brand)
            composite_errors = torch.cat((composite_difference[plate].reshape(-1), (probe_composite - probe_expected).reshape(-1)))
            composite_maximum = max(composite_maximum, composite_errors.abs().max().item())
    values = torch.tensor(per_profile_mse, dtype=torch.float64)
    return {
        "mse": values.mean().item(),
        "mae": sum(per_profile_mae) / len(per_profile_mae),
        "p95ProfileMse": torch.quantile(values, 0.95).item(),
        "worstProfileMse": max(per_profile_mse),
        "compositeMse": sum(per_profile_composite_mse) / len(per_profile_composite_mse),
        "brandMaxError": brand_maximum if teacher.brand is not None else None,
        "brandMse": brand_squared_error / brand_channel_count if brand_channel_count else None,
        "compositeBrandMaxError": composite_maximum if teacher.brand is not None else None,
        "brandChannelCount": brand_channel_count,
        "brandProbeDescription": "native raster plate pixels plus all 63 plate-cell centers per profile",
        "profileCount": len(profiles),
        "width": width,
        "height": height,
        "perProfileMse": per_profile_mse,
    }


def byte_rgba(colors):
    alpha = torch.ones((len(colors), 1), dtype=colors.dtype, device=colors.device)
    rgba = torch.cat((colors, alpha), dim=1)
    return torch.floor(rgba.clamp(0, 1) * 255 + 0.5).to(torch.int64).reshape(-1).cpu().tolist()


def float_rgba(colors):
    alpha = torch.ones((len(colors), 1), dtype=colors.dtype, device=colors.device)
    return torch.cat((colors, alpha), dim=1).reshape(-1).cpu().tolist()


@torch.no_grad()
def parity_fixture(model, teacher, profiles, artifact_id, graph_hash, brand_mode, device="cpu"):
    raster_cases = []
    for profile in profiles[:2]:
        width, height = 16, 12
        coordinates = pixel_coordinates(width, height, device)
        upl = profile.to(device).expand(len(coordinates), -1)
        expected = teacher(upl, coordinates, validate=False)
        raw = model(torch.cat((coordinates, upl), dim=1))
        composite = apply_brand(raw, coordinates, teacher.brand) if brand_mode == "composite" else raw
        raster_cases.append({
            "upl": profile.tolist(),
            "width": width,
            "height": height,
            "teacherRgba": float_rgba(expected),
            "rawStudentRgba": float_rgba(raw),
            "studentRgba": float_rgba(composite),
            "teacherBytes": byte_rgba(expected),
            "rawStudentBytes": byte_rgba(raw),
            "studentBytes": byte_rgba(composite),
        })
    coordinates = torch.tensor([[0, 0], [0.5, 0.5], [1, 1], [0.13, 0.79]], dtype=torch.float32, device=device)
    if teacher.brand is not None:
        coordinates = torch.cat((coordinates, brand_probe_coordinates(teacher.brand, device)))
    upl = profiles[0].to(device).expand(len(coordinates), -1)
    raw = model(torch.cat((coordinates, upl), dim=1))
    expected = teacher(upl, coordinates, validate=False)
    composite = apply_brand(raw, coordinates, teacher.brand) if brand_mode == "composite" else raw
    return {
        "format": "angelos.parity.v1",
        "artifactId": artifact_id,
        "graphHash": graph_hash,
        "floatAbsoluteTolerance": 0.001,
        "byteAbsoluteTolerance": 1,
        "byteRounding": "floor(clamp(channel,0,1)*255+0.5)",
        "colorSpace": "display-encoded-rgb",
        "rasterCases": raster_cases,
        "pointCase": {
            "upl": profiles[0].tolist(),
            "xy": coordinates.cpu().tolist(),
            "teacherRgb": expected.cpu().tolist(),
            "rawStudentRgb": raw.cpu().tolist(),
            "studentRgb": composite.cpu().tolist(),
        },
    }
