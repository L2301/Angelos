"""Offline contracts and injected encoder fixtures; no pretrained semantic claims."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from angelos_ml import upl


ROOT = Path(__file__).resolve().parents[2]


def ingredients_fixture():
    facts = {"ageBand": "25-34", "region": None, "openness": 1e-7, "energy": -0.0,
             "interests": ["hiking", "日本語 🌲"], "goals": None, "preferredStyles": [], "preferredLanguage": "English"}
    return {"format": upl.INGREDIENTS_FORMAT, "synthetic": True, "adult": True, "factsSchema": upl.FACTS_SCHEMA,
            "facts": facts, "provenance": [{"field": name, "source": "synthetic:declared", "occurredAt": 100, "receivedAt": 110}
                                          for name in facts], "eventsSchema": upl.EVENTS_SCHEMA,
            "events": [{"id": "event-1", "kind": "save", "topic": "outdoor travel", "occurredAt": 190,
                        "receivedAt": 195, "expiresAt": 300, "source": "synthetic:fixture", "campaignId": "campaign-a"}],
            "factRevision": 2, "eventRevision": 1, "consentVersion": 1, "campaignId": "campaign-a", "asOf": 200}


class FixtureTokenizer:
    def __init__(self, count=3):
        self.count = count
        self.calls = []

    def __call__(self, text, **arguments):
        self.calls.append((text, arguments))
        return {"input_ids": torch.ones((1, self.count), dtype=torch.long),
                "attention_mask": torch.ones((1, self.count), dtype=torch.long)}


class FixtureEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3", hidden_size=1024)
        self.values = nn.Parameter(torch.linspace(-1, 1, 1024))
        self.calls = []

    def forward(self, input_ids, attention_mask, **arguments):
        self.calls.append((torch.is_grad_enabled(), arguments))
        return SimpleNamespace(last_hidden_state=self.values.expand(*input_ids.shape, 1024))


class UPLTests(unittest.TestCase):
    def setUp(self):
        self.ingredients = ingredients_fixture()
        self.request = {"ingredients": self.ingredients, "dimension": 512}
        self.model, self.tokenizer = FixtureEmbeddingModel(), FixtureTokenizer()
        self.encoder = upl.LocalSemanticEncoder(self.model, self.tokenizer, "cpu")

    def encode(self, request=None):
        with patch.object(upl, "load_local_encoder", return_value=self.encoder):
            return upl.encode_request(request or self.request)

    def test_exact_output_all_widths_signed_normalized_and_frozen(self):
        before = copy.deepcopy(self.ingredients)
        for dimension in upl.DIMENSIONS:
            result = self.encode({**self.request, "dimension": dimension})
            self.assertEqual(set(result), {"format", "synthetic", "adult", "factsSchema", "facts", "encoder", "upl", "context", "recentIntent"})
            self.assertEqual(result["encoder"], upl.encoder_metadata(dimension))
            vector = torch.tensor(result["upl"])
            torch.testing.assert_close(vector, torch.nn.functional.normalize(self.model.values[:dimension], dim=0))
            self.assertTrue((vector < 0).any())
            self.assertEqual(set(result["context"]), {"campaignId", "factRevision", "eventRevision", "consentVersion", "asOf", "ingredientsHash"})
            self.assertEqual(result["recentIntent"], [{"kind": "save", "topic": "outdoor travel", "occurredAt": 190, "expiresAt": 300}])
            self.assertNotIn("provenance", result)
            self.assertEqual(result["facts"], self.ingredients["facts"])
        self.assertEqual(self.ingredients, before)
        self.assertFalse(self.model.training)
        self.assertFalse(self.model.values.requires_grad)
        self.assertTrue(all(not grad for grad, unused in self.model.calls))

    def test_only_stable_facts_enter_semantics(self):
        first = self.encode()
        changed = copy.deepcopy(self.ingredients)
        changed.update(campaignId="other-opaque-id", eventRevision=7, factRevision=9, consentVersion=2, asOf=201)
        changed["events"][0].update(campaignId="other-opaque-id", topic="never encode this event", id="different-event")
        changed["provenance"][0]["source"] = "private-provenance-marker"
        second = self.encode({"ingredients": changed, "dimension": 512})
        self.assertEqual(first["upl"], second["upl"])
        self.assertEqual(self.tokenizer.calls[0][0], self.tokenizer.calls[1][0])
        self.assertNotEqual(first["context"]["ingredientsHash"], second["context"]["ingredientsHash"])
        self.assertNotEqual(first["recentIntent"], second["recentIntent"])
        text = self.tokenizer.calls[0][0]
        for forbidden in ("campaign-a", "event-1", "synthetic:declared", "consentVersion", "outdoor travel"):
            self.assertNotIn(forbidden, text)
        self.assertIn("hiking", text)
        altered_facts = {**self.ingredients["facts"], "interests": ["painting"]}
        self.assertNotEqual(text, upl.semantic_text(altered_facts))
        self.assertEqual(upl.semantic_text({"ageBand": "65+", "energy": 1}), upl.semantic_text({"ageBand": "65+", "energy": 1.0}))

    def test_javascript_supplied_byte_hash_interoperability(self):
        script = """
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
  return JSON.stringify(value);
}
const ingredients = JSON.parse(readFileSync(0, 'utf8'));
const ingredientsJson = canonical(ingredients);
console.log(JSON.stringify({ ingredientsJson, hash: createHash('sha256').update(ingredientsJson, 'utf8').digest('hex') }));
"""
        response = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(self.ingredients),
                                  text=True, capture_output=True, check=True)
        javascript = json.loads(response.stdout)
        result = self.encode({**self.request, "ingredientsJson": javascript["ingredientsJson"]})
        self.assertEqual(result["context"]["ingredientsHash"], javascript["hash"])
        self.assertNotEqual(javascript["ingredientsJson"], upl.canonical_json(self.ingredients))
        spaced = "\n " + javascript["ingredientsJson"] + "\n"
        second = self.encode({**self.request, "ingredientsJson": spaced})
        self.assertEqual(second["context"]["ingredientsHash"], hashlib.sha256(spaced.encode()).hexdigest())
        self.assertNotEqual(result["context"]["ingredientsHash"], second["context"]["ingredientsHash"])
        validation = """
import { readFileSync } from 'node:fs';
import { validateSemanticBundle } from './src/core/upl.js';
const request = JSON.parse(readFileSync(0, 'utf8'));
validateSemanticBundle(request.bundle, request.ingredients);
console.log('valid');
"""
        checked = subprocess.run(["node", "--input-type=module", "-e", validation], cwd=ROOT,
                                 input=json.dumps({"bundle": result, "ingredients": self.ingredients}),
                                 text=True, capture_output=True, check=True)
        self.assertEqual(checked.stdout.strip(), "valid")

    def test_python_fallback_hash_and_request_copy(self):
        prepared = upl.prepare_request(self.request)
        expected = hashlib.sha256(upl.canonical_json(self.ingredients).encode("utf-8")).hexdigest()
        self.assertEqual(self.encode()["context"]["ingredientsHash"], expected)
        prepared["ingredients"]["facts"]["interests"].append("new")
        self.assertNotIn("new", self.ingredients["facts"]["interests"])

    def test_serialized_ingredients_must_validate_and_match(self):
        source = upl.canonical_json(self.ingredients)
        bad_strings = [source.replace('"asOf":200', '"asOf":201'), source.replace('"asOf":200', '"asOf":200,"asOf":200'),
                       source.replace('"energy":-0.0', '"energy":NaN'), source.replace('"energy":-0.0', '"energy":false'), "[]", "null"]
        for serialized in bad_strings:
            with self.subTest(serialized=serialized[:60]), self.assertRaises(ValueError):
                upl.prepare_request({**self.request, "ingredientsJson": serialized})

    def test_strict_json_input_size_constants_duplicates_and_unknown_keys(self):
        for raw in (b'{"ingredients":{},"dimension":512,"dimension":128}', b'NaN', b'Infinity', b'[]', b'\xff',
                    b' ' * (upl.MAX_STDIN_BYTES + 1)):
            with self.assertRaises((ValueError, UnicodeError)):
                upl.read_request(io.BytesIO(raw))
        for dimension in (4, 127, 2048, True, "512", float("nan"), 10 ** 400):
            with self.assertRaises(ValueError):
                upl.prepare_request({**self.request, "dimension": dimension})
        with self.assertRaises(ValueError):
            upl.prepare_request({**self.request, "modelId": "arbitrary/repository"})
        self.assertEqual(upl.read_request(io.BytesIO(json.dumps(self.request).encode()))["dimension"], 512)

    def test_unknown_and_nonadult_contracts_fail_before_loading(self):
        changes = ({"synthetic": False}, {"synthetic": 1}, {"adult": False}, {"format": "angelos.upl-input.v1"},
                   {"factsSchema": "angelos.facts.v1"}, {"eventsSchema": "unknown"}, {"opaqueUserId": "secret"},
                   {"campaignId": "../escape"}, {"consentVersion": 0}, {"factRevision": 1.5}, {"asOf": True})
        with patch.object(upl, "load_local_encoder") as loader:
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    upl.encode_request({"ingredients": {**self.ingredients, **change}, "dimension": 512})
            loader.assert_not_called()

    def test_nullable_facts_and_text_list_bounds(self):
        self.assertEqual(upl.validate_facts({"ageBand": "18-24", "region": None}), {"ageBand": "18-24", "region": None})
        self.assertEqual(len(upl.validate_facts({"ageBand": "65+", "interests": [str(index) for index in range(16)]})["interests"]), 16)
        invalid = ({}, {"ageBand": None}, {"ageBand": "under-18"}, {"ageBand": "65+", "region": "Mars"},
                   {"ageBand": "65+", "openness": True}, {"ageBand": "65+", "openness": float("inf")},
                   {"ageBand": "65+", "energy": -0.1}, {"ageBand": "65+", "energy": 10 ** 400},
                   {"ageBand": "65+", "interests": ["same", "same"]}, {"ageBand": "65+", "goals": ["x"] * 17},
                   {"ageBand": "65+", "goals": ["x" * 81]}, {"ageBand": "65+", "goals": ["🌲" * 41]},
                   {"ageBand": "65+", "preferredStyles": [""]}, {"ageBand": "65+", "preferredLanguage": "\ud800"},
                   {"ageBand": "65+", "goals": [" padded"]}, {"ageBand": "65+", "goals": ["padded "]},
                   {"ageBand": "65+", "goals": ["\ufeffpadded"]}, {"ageBand": "65+", "goals": ["DEL\x7f"]},
                   {"ageBand": "65+", "interests": ["line\nbreak"]}, {"ageBand": "65+", "diagnosis": "unknown"})
        for facts in invalid:
            with self.subTest(facts=facts), self.assertRaises(ValueError):
                upl.validate_facts(facts)

    def test_provenance_is_latest_unique_and_time_correct(self):
        for update in ({"field": "missing"}, {"occurredAt": 201}, {"receivedAt": 99}, {"receivedAt": 201}, {"source": ""}):
            changed = copy.deepcopy(self.ingredients)
            changed["provenance"][0].update(update)
            with self.assertRaises(ValueError):
                upl.validate_ingredients(changed)
        for records in ([], self.ingredients["provenance"] + [self.ingredients["provenance"][0]],
                        [self.ingredients["provenance"][0]] * len(self.ingredients["facts"])):
            with self.assertRaises(ValueError):
                upl.validate_ingredients({**self.ingredients, "provenance": records})

    def test_events_must_be_live_campaign_scoped_and_bounded(self):
        for update in ({"kind": "purchase"}, {"topic": "x" * 81}, {"expiresAt": 200}, {"occurredAt": 201},
                       {"receivedAt": 201}, {"receivedAt": 189}, {"campaignId": "campaign-b"}, {"extra": True}):
            changed = copy.deepcopy(self.ingredients)
            changed["events"][0].update(update)
            with self.subTest(update=update), self.assertRaises(ValueError):
                upl.validate_ingredients(changed)
        for length in (2, 65):
            with self.assertRaises(ValueError):
                upl.validate_ingredients({**self.ingredients, "events": self.ingredients["events"] * length})
        self.assertEqual(self.encode({"ingredients": {**self.ingredients, "events": [], "eventRevision": 0}, "dimension": 128})["recentIntent"], [])

    def test_pooling_left_right_and_mixed_padding(self):
        hidden = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]])
        torch.testing.assert_close(upl.last_token_pool(hidden, mask), hidden[torch.arange(3), torch.tensor([3, 1, 2])])
        for invalid in (torch.zeros(3, 4), torch.full((3, 4), 2), torch.ones(3, 3)):
            with self.assertRaises(ValueError):
                upl.last_token_pool(hidden, invalid)

    def test_tiny_real_qwen_transformer_matches_explicit_last_token_reference(self):
        from transformers import Qwen3Config, Qwen3Model

        configuration = Qwen3Config(vocab_size=16, hidden_size=1024, intermediate_size=32,
                                   num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                                   head_dim=16, max_position_embeddings=32, pad_token_id=0)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(91)
            model = Qwen3Model(configuration)
        encoder = upl.LocalSemanticEncoder(model, FixtureTokenizer(), "cpu")
        actual = torch.tensor(encoder.encode("random offline architecture fixture", 128))
        with torch.inference_mode():
            tokens = torch.ones((1, 3), dtype=torch.long)
            hidden = model(input_ids=tokens, attention_mask=tokens, use_cache=False).last_hidden_state
            expected = torch.nn.functional.normalize(hidden[0, -1, :128].float(), dim=0)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()))

    def test_token_budget_zero_norm_and_nonfinite_fail_without_substitution(self):
        tokenizer = FixtureTokenizer(count=upl.MAX_INPUT_TOKENS + 1)
        encoder = upl.LocalSemanticEncoder(self.model, tokenizer, "cpu")
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            encoder.encode("bounded fixture", 128)
        self.assertEqual(self.model.calls, [])
        self.assertFalse(tokenizer.calls[0][1]["truncation"])
        for value in (0, float("nan"), float("inf")):
            with torch.no_grad():
                self.model.values.fill_(value)
            with self.assertRaises(ValueError):
                self.encoder.encode("fixture", 128)

    def test_missing_cache_never_calls_hub_download(self):
        with tempfile.TemporaryDirectory() as temporary, patch("huggingface_hub.snapshot_download", side_effect=AssertionError("network forbidden")) as download:
            with self.assertRaisesRegex(RuntimeError, "explicitly authorized provisioning"):
                upl.load_local_encoder(cache_dir=temporary)
            download.assert_not_called()

    def test_existing_snapshot_loads_only_pinned_safe_local_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots" / upl.REVISION
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text(json.dumps({"model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28}))
            (snapshot / "tokenizer_config.json").write_text("{}")
            (snapshot / "tokenizer.json").write_text("{}")
            (snapshot / "model.safetensors").write_bytes(b"explicit mocked loader fixture")
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=self.tokenizer) as tokenizer_loader, \
                    patch("transformers.AutoModel.from_pretrained", return_value=self.model) as model_loader:
                loaded = upl.load_local_encoder(cache_dir=temporary)
                self.assertIs(loaded.model, self.model)
                for loader in (tokenizer_loader, model_loader):
                    self.assertEqual(loader.call_args.args[0], str(snapshot))
                    self.assertTrue(loader.call_args.kwargs["local_files_only"])
                    self.assertFalse(loader.call_args.kwargs["trust_remote_code"])
                    self.assertEqual(loader.call_args.kwargs["revision"], upl.REVISION)
                self.assertTrue(model_loader.call_args.kwargs["use_safetensors"])
                self.assertEqual(model_loader.call_args.kwargs["dtype"], torch.float32)
            (snapshot / "tokenizer_config.json").write_text('{"auto_map": {}}')
            with self.assertRaisesRegex(ValueError, "custom code"):
                upl.load_local_encoder(cache_dir=temporary)

    def test_cli_and_node_wrapper_missing_cache_fail_actionably(self):
        with tempfile.TemporaryDirectory() as temporary:
            commands = ([sys.executable, "-m", "angelos_ml.upl"], ["node", "scripts/upl.js"])
            for command in commands:
                completed = subprocess.run([*command, "--stdin", "--device", "cpu", "--cache-dir", temporary],
                                           input=json.dumps(self.request), text=True, capture_output=True, cwd=ROOT,
                                           env={**os.environ, "PYTHONPATH": str(ROOT / "ml")}, timeout=30)
                self.assertEqual(completed.returncode, 2, completed.stderr)
                self.assertEqual(completed.stdout, "")
                self.assertIn(upl.REVISION, completed.stderr)
                self.assertIn("never downloads", completed.stderr)

    def test_cli_stdout_is_exactly_one_output_payload(self):
        source = io.TextIOWrapper(io.BytesIO(json.dumps(self.request).encode()))
        output = io.StringIO()
        with patch.object(sys, "stdin", source), patch.object(sys, "stdout", output), \
                patch.object(upl, "load_local_encoder", return_value=self.encoder):
            upl.main(["--stdin", "--device", "cpu"])
        result = json.loads(output.getvalue())
        self.assertEqual(result["format"], upl.OUTPUT_FORMAT)
        self.assertEqual(len(output.getvalue().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
