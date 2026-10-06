"""Fixed, disjoint synthetic adult profile splits and coordinate sampling."""

import hashlib
import json

import torch


def profile_splits(seed, train_count=128, heldout_count=32):
    if train_count < 2 or heldout_count < 2:
        raise ValueError("Both profile splits require at least two profiles")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    profiles = []
    observed = set()
    while len(profiles) < train_count + heldout_count:
        age_band = torch.randint(0, 6, (), generator=generator).item() / 5
        region = torch.randint(0, 4, (), generator=generator).item() / 3
        declared = torch.rand(2, generator=generator).tolist()
        candidate = torch.tensor([age_band, region, *declared], dtype=torch.float32)
        key = tuple(candidate.tolist())
        if key not in observed:
            observed.add(key)
            profiles.append(candidate)
    stacked = torch.stack(profiles)
    return stacked[:train_count], stacked[train_count:]


def profiles_hash(profiles):
    encoded = json.dumps(profiles.cpu().tolist(), separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sample_batch(profiles, batch_size, generator, brand=None, plate_fraction=0.25):
    if not 0 <= plate_fraction < 1:
        raise ValueError("Plate sampling fraction must be in [0,1)")
    indices = torch.randint(len(profiles), (batch_size,), generator=generator)
    upl = profiles[indices]
    coordinates = torch.rand((batch_size, 2), generator=generator)
    if brand is not None:
        plate_count = int(batch_size * plate_fraction)
        coordinates[:plate_count, 0] = brand["x"] + coordinates[:plate_count, 0] * (7 * brand["cell"])
        coordinates[:plate_count, 1] = brand["y"] + coordinates[:plate_count, 1] * (9 * brand["cell"])
    return upl, coordinates
