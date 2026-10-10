"""Untrained token-adapter shape and autograd checks, not personalization evidence."""

import unittest

import torch

from angelos_ml.upl import DIMENSIONS, REVISION
from angelos_ml.upl_adapter import UPLConditioningAdapter


class UPLAdapterTests(unittest.TestCase):
    def adapter(self, **arguments):
        return UPLConditioningAdapter(feature_schema="fixture.recent-context.v1", **arguments)

    def inputs(self, dimension=512):
        generator = torch.Generator().manual_seed(5)
        semantic = torch.nn.functional.normalize(torch.randn(2, dimension, generator=generator), dim=1)
        recent = torch.randn(2, 8, generator=generator)
        context = torch.randn(2, 8, generator=generator)
        return semantic, recent, context

    def test_all_semantic_widths_produce_conditioning_tokens(self):
        for dimension in DIMENSIONS:
            adapter = self.adapter(semantic_dim=dimension)
            self.assertEqual(adapter(*self.inputs(dimension)).shape, (2, 4, 256))
            metadata = adapter.specification()
            self.assertEqual(metadata["encoder"]["revision"], REVISION)
            self.assertFalse(metadata["pretrained"])
            self.assertEqual(metadata["initialization"], "random-untrained")
            self.assertIsNone(metadata["validatedGenerator"])

    def test_genuine_input_and_parameter_gradients_and_weight_update(self):
        adapter = self.adapter()
        values = [value.requires_grad_() for value in self.inputs()]
        before = {name: value.clone() for name, value in adapter.state_dict().items()}
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
        adapter(*values).square().mean().backward()
        for value in values:
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for parameter in adapter.parameters():
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[name], value) for name, value in adapter.state_dict().items()))

    def test_recent_and_context_are_separate_influences(self):
        adapter = self.adapter()
        semantic, recent, context = self.inputs()
        base = adapter(semantic, recent, context)
        self.assertFalse(torch.equal(base, adapter(semantic, recent + 1, context)))
        self.assertFalse(torch.equal(base, adapter(semantic, recent, context + 1)))

    def test_seed_reproducibility_and_state_dict_roundtrip(self):
        global_state = torch.random.get_rng_state()
        first, second = self.adapter(seed=72), self.adapter(seed=72)
        self.assertTrue(torch.equal(global_state, torch.random.get_rng_state()))
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], atol=0, rtol=0)
        third = self.adapter(seed=73)
        third.load_state_dict(first.state_dict())
        torch.testing.assert_close(first(*self.inputs()), third(*self.inputs()), atol=0, rtol=0)

    def test_invalid_dimensions_and_feature_schema(self):
        for arguments in ({"semantic_dim": 4}, {"num_tokens": 17}, {"recent_dim": 0}, {"context_dim": True},
                          {"token_dim": 4097}, {"seed": True}, {"hidden_dim": 4096, "token_dim": 4096}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.adapter(**arguments)
        with self.assertRaises(ValueError):
            UPLConditioningAdapter(feature_schema="")

    def test_invalid_feature_values_shapes_dtype_and_normalization(self):
        adapter = self.adapter()
        semantic, recent, context = self.inputs()
        invalid = ((semantic * 2, recent, context), (semantic[:, :128], recent, context),
                   (semantic, recent[:1], context), (semantic, recent.long(), context),
                   (semantic, recent, context.double()), (semantic, recent, torch.full_like(context, float("nan"))),
                   (semantic[:0], recent[:0], context[:0]), (semantic.tolist(), recent, context))
        for values in invalid:
            with self.assertRaises(ValueError):
                adapter(*values)


if __name__ == "__main__":
    unittest.main()
