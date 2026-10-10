"""Untrained, model-independent UPL token adapter; not FLUX/SANA conditioning."""

import torch
from torch import nn

from .upl import DIMENSIONS, _identifier, encoder_metadata


class UPLConditioningAdapter(nn.Module):
    """Project aligned semantic, recent-intent and context feature batches to tokens."""

    def __init__(self, semantic_dim=512, recent_dim=8, context_dim=8, *, token_dim=256,
                 num_tokens=4, hidden_dim=256, feature_schema, seed=2026):
        super().__init__()
        dimensions = {"semantic_dim": semantic_dim, "recent_dim": recent_dim, "context_dim": context_dim,
                      "token_dim": token_dim, "num_tokens": num_tokens, "hidden_dim": hidden_dim}
        if any(type(value) is not int or not 1 <= value <= 4096 for value in dimensions.values()):
            raise ValueError("Adapter dimensions must be bounded positive integers")
        if semantic_dim not in DIMENSIONS or num_tokens > 16 or recent_dim > 1024 or context_dim > 1024:
            raise ValueError("Unsupported semantic width or adapter feature/token budget")
        total = semantic_dim + recent_dim + context_dim
        if (total + 1) * hidden_dim + (hidden_dim + 1) * token_dim * num_tokens > 8000000:
            raise ValueError("Adapter exceeds eight million parameters")
        _identifier(feature_schema, "versioned recent/context feature_schema")
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError("Adapter seed must be uint32")
        self.dimensions = dimensions
        self.feature_schema = feature_schema
        self.seed = seed
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.projection = nn.Sequential(nn.Linear(total, hidden_dim, dtype=torch.float32), nn.GELU(),
                                            nn.Linear(hidden_dim, token_dim * num_tokens, dtype=torch.float32))

    def specification(self):
        return {"format": "angelos.upl-conditioning-adapter.v1", "encoder": encoder_metadata(self.dimensions["semantic_dim"]),
                "dimensions": dict(self.dimensions), "featureSchema": self.feature_schema, "seed": self.seed,
                "initialization": "random-untrained", "pretrained": False, "validatedGenerator": None}

    def forward(self, semantic, recent, context):
        reference = self.projection[0].weight
        batch = None
        for name, value in (("semantic", semantic), ("recent", recent), ("context", context)):
            if (not isinstance(value, torch.Tensor) or not value.is_floating_point() or value.ndim != 2
                    or not 1 <= value.shape[0] <= 256 or value.shape[1] != self.dimensions[name + "_dim"]
                    or value.device != reference.device or value.dtype != reference.dtype or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid {name} feature tensor; dimensions, dtype and device must match explicitly")
            if batch is not None and value.shape[0] != batch:
                raise ValueError("Feature batches must align")
            batch = value.shape[0]
        norms = torch.linalg.vector_norm(semantic.float(), dim=1)
        if (semantic.abs() > 1).any() or not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=0):
            raise ValueError("Semantic inputs must be L2-normalized; no silent rescaling")
        tokens = self.projection(torch.cat((semantic, recent, context), dim=1))
        if not torch.isfinite(tokens).all():
            raise ValueError("Adapter produced nonfinite tokens")
        return tokens.reshape(batch, self.dimensions["num_tokens"], self.dimensions["token_dim"])
