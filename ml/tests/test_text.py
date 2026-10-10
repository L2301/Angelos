"""Offline injected text fixtures test contracts, not pretrained-model quality."""

import copy
import fnmatch
import io
import json
import os
import subprocess
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from angelos_ml.dataset import build_dataset, write_dataset
from angelos_ml.graph import graph_hash, validate_graph
from angelos_ml.models import MODELS, download_model
from angelos_ml.providers import PipelineRegistry, node_seed
from angelos_ml.teacher import RasterTeacher
from angelos_ml.text import main, read_request
from angelos_ml.text_runtime import CausalTextPipeline, load_text_pipeline
from angelos_ml.train_diffusion import load_dataset, validate_model_provenance
from test_generative import FakePipeline, example, named


ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = {"system": "Return concise copy.", "max_tokens": 8, "temperature": 0.7}
REQUEST = {"model": "qwen3-0.6b", "prompt": "Aurora above a lake", **PARAMETERS, "seed": 17}


class FixtureTokenizer:
    model_max_length = 77

    def __init__(self, count=3, decoded=None):
        self.count = count
        self.decoded = decoded
        self.decoded_tokens = None
        self.template_arguments = None

    def apply_chat_template(self, messages, **arguments):
        self.template_arguments = (messages, arguments)
        return "fixture chat template"

    def __call__(self, text, **arguments):
        if arguments.get("return_tensors") == "pt":
            return {"input_ids": torch.ones((1, self.count), dtype=torch.long), "attention_mask": torch.ones((1, self.count), dtype=torch.long)}
        return {"input_ids": [1] * self.count}

    def decode(self, tokens, **arguments):
        self.decoded_tokens = list(tokens)
        return self.decoded if self.decoded is not None else " ".join(str(token) for token in tokens if token != 3)


class FixtureCausalModel:
    def __init__(self, stop=False):
        self.config = SimpleNamespace(max_position_embeddings=8192)
        self.generation_config = SimpleNamespace(eos_token_id=[3])
        self.stop = stop
        self.calls = []

    def eval(self):
        return self

    def __call__(self, **arguments):
        self.calls.append(arguments)
        if self.stop:
            logits = [0.0, 0.0, 0.0, 10.0] if arguments["past_key_values"] is not None else [0.0, 10.0, 0.0, 0.0]
        else:
            logits = [1.0, 1.0, 1.0, -100.0]
        return SimpleNamespace(logits=torch.tensor([[logits]]), past_key_values=("fixture cache",))


def fixture_registry():
    raster = FakePipeline()
    raster.tokenizer = FixtureTokenizer()
    return PipelineRegistry(pipelines={
        "qwen3-0.6b": CausalTextPipeline(FixtureCausalModel(), FixtureTokenizer(decoded="Fixture aurora scene"), "cpu"),
        "tiny-sd": raster,
    })


class TextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.graph = example("aurora-text")

    def test_cross_language_hash_types_and_validation(self):
        validate_graph(self.graph)
        program = "import {graphHash,validateGraph} from './src/core/graph.js'; const graph=JSON.parse(process.argv[1]); try {validateGraph(graph); console.log(await graphHash(graph));} catch {process.exitCode=2;}"
        valid = copy.deepcopy(self.graph)
        named(valid, "generate_text")["params"].update(system="Literal {braces} and 🌌", max_tokens=8.0)
        invalid = []
        for field, value in (("model", "tiny-sd"), ("system", " "), ("system", "\ufeff"), ("system", "🌌" * 1025), ("system", "\ud800"), ("max_tokens", True), ("max_tokens", 0), ("max_tokens", 513), ("max_tokens", 1.2), ("temperature", True), ("temperature", 2.1)):
            graph = copy.deepcopy(self.graph)
            named(graph, "generate_text")["params"][field] = value
            invalid.append(graph)
        graph = copy.deepcopy(self.graph)
        named(graph, "generate_text")["inputs"]["prompt"] = "openness"
        invalid.append(graph)
        graph = copy.deepcopy(self.graph)
        graph["output"] = "copy"
        invalid.append(graph)
        for graph in [self.graph, valid, *invalid]:
            result = subprocess.run(["node", "--input-type=module", "-e", program, json.dumps(graph)], cwd=ROOT, text=True, capture_output=True)
            if graph in invalid:
                with self.assertRaises(ValueError):
                    validate_graph(graph)
                self.assertEqual(result.returncode, 2)
            else:
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(graph_hash(graph), result.stdout.strip())

    def test_seeded_sampling_concurrency_and_no_global_rng_leak(self):
        pipeline = CausalTextPipeline(FixtureCausalModel(), FixtureTokenizer(), "cpu")
        registry = PipelineRegistry(pipelines={"qwen3-0.6b": pipeline})
        state = torch.random.get_rng_state().clone()
        seeds = [17, 18, 17, 19] * 4
        def generate(seed):
            return registry.generate_text("qwen3-0.6b", "scene", PARAMETERS, node_seed(seed, "copy"))
        serial = [generate(seed) for seed in seeds]
        with ThreadPoolExecutor(max_workers=4) as executor:
            concurrent = list(executor.map(generate, seeds))
        self.assertEqual(serial, concurrent)
        self.assertEqual(serial[0], serial[2])
        self.assertNotEqual(serial[0], serial[1])
        torch.testing.assert_close(torch.random.get_rng_state(), state)
        self.assertEqual(registry.metadata("qwen3-0.6b")["backend"], "injected-test-pipeline")

    def test_nonthinking_only_new_tokens_eos_greedy_and_budget(self):
        model = FixtureCausalModel(stop=True)
        tokenizer = FixtureTokenizer()
        registry = PipelineRegistry(pipelines={"qwen3-0.6b": CausalTextPipeline(model, tokenizer, "cpu")})
        result = registry.generate_text("qwen3-0.6b", "user text", {**PARAMETERS, "temperature": 0}, 12)
        self.assertEqual(result, {"text": "1", "usage": {"input_tokens": 3, "output_tokens": 2}, "finish_reason": "stop"})
        self.assertEqual(tokenizer.decoded_tokens, [1, 3])
        self.assertFalse(tokenizer.template_arguments[1]["enable_thinking"])
        self.assertEqual(tokenizer.template_arguments[0][1]["content"], "user text")
        self.assertEqual(model.calls[1]["input_ids"].shape, (1, 1))
        self.assertEqual(model.calls[1]["attention_mask"].shape, (1, 4))
        limited = registry.generate_text("qwen3-0.6b", "user text", {**PARAMETERS, "max_tokens": 1, "temperature": 0}, 12)
        self.assertEqual(limited["finish_reason"], "length")

    def test_context_and_malformed_model_outputs_fail(self):
        for token_count, context in ((4097, 8192), (4096, 4096)):
            model = FixtureCausalModel()
            model.config.max_position_embeddings = context
            registry = PipelineRegistry(pipelines={"qwen3-0.6b": CausalTextPipeline(model, FixtureTokenizer(token_count), "cpu")})
            with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
                registry.generate_text("qwen3-0.6b", "scene", PARAMETERS, 12)

            self.assertEqual(model.calls, [])
        for decoded in ("", "   ", "🌌" * 4097, "\ud800"):
            registry = PipelineRegistry(pipelines={"qwen3-0.6b": CausalTextPipeline(FixtureCausalModel(), FixtureTokenizer(decoded=decoded), "cpu")})
            with self.assertRaises(ValueError):
                registry.generate_text("qwen3-0.6b", "scene", PARAMETERS, 12)
        for logits in (torch.tensor([[[float("nan")]]]), torch.zeros(1, 2), torch.zeros(1, 1, 0)):
            model = FixtureCausalModel()
            failing = Mock(return_value=SimpleNamespace(logits=logits))
            failing.config = model.config
            failing.generation_config = model.generation_config
            failing.eval.return_value = failing
            registry = PipelineRegistry(pipelines={"qwen3-0.6b": CausalTextPipeline(failing, FixtureTokenizer(), "cpu")})
            with self.assertRaises(ValueError):
                registry.generate_text("qwen3-0.6b", "scene", PARAMETERS, 12)
        good = {"text": "fixture", "usage": {"input_tokens": 3, "output_tokens": 8}, "finish_reason": "length"}
        for malformed in (None, "fake", {**good, "extra": 1}, {**good, "text": []}, {**good, "usage": {"input_tokens": True, "output_tokens": 8}}, {**good, "finish_reason": "unknown"}, {**good, "usage": {"input_tokens": 3, "output_tokens": 1}}):
            registry = PipelineRegistry(pipelines={"qwen3-0.6b": Mock(return_value=malformed)})
            with self.assertRaises(ValueError):
                registry.generate_text("qwen3-0.6b", "scene", PARAMETERS, 12)

    def test_untrained_qwen_architecture_fixture_supports_incremental_cache(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = Qwen3ForCausalLM(Qwen3Config(
                vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                max_position_embeddings=128, eos_token_id=31,
            ))
        registry = PipelineRegistry(pipelines={
            "qwen3-0.6b": CausalTextPipeline(model, FixtureTokenizer(decoded="Untrained architecture fixture"), "cpu"),
        })
        state = torch.random.get_rng_state().clone()
        first = registry.generate_text("qwen3-0.6b", "scene", PARAMETERS, 17)
        second = registry.generate_text("qwen3-0.6b", "scene", PARAMETERS, 17)
        self.assertEqual(first, second)
        self.assertEqual(first["text"], "Untrained architecture fixture")
        torch.testing.assert_close(torch.random.get_rng_state(), state)

    def test_loader_and_download_pin_local_safe_root_weights(self):
        for model in ("qwen3-0.6b", "qwen3-8b"):
            checkpoint = Mock()
            checkpoint.to.return_value = checkpoint
            checkpoint.eval.return_value = checkpoint
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=FixtureTokenizer()) as tokenizer_loader, patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=checkpoint) as loader, patch("huggingface_hub.snapshot_download", side_effect=AssertionError("runtime download")):
                pipeline = load_text_pipeline(model, cache_dir="/offline/cache")
                self.assertIs(pipeline.model, checkpoint)
                for adapter in (tokenizer_loader, loader):
                    self.assertEqual(adapter.call_args.args, (MODELS[model].repo,))
                    self.assertEqual(adapter.call_args.kwargs["revision"], MODELS[model].revision)
                    self.assertEqual(adapter.call_args.kwargs["cache_dir"], "/offline/cache")
                    self.assertTrue(adapter.call_args.kwargs["local_files_only"])
                    self.assertFalse(adapter.call_args.kwargs["trust_remote_code"])
                self.assertTrue(loader.call_args.kwargs["use_safetensors"])
            with patch("huggingface_hub.snapshot_download", return_value="/snapshot") as download:
                download_model(model)
                patterns = download.call_args.kwargs["allow_patterns"]
                for filename in ("model.safetensors", "model-00001-of-00005.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "merges.txt"):
                    self.assertTrue(any(fnmatch.fnmatch(filename, pattern) for pattern in patterns), filename)
                self.assertFalse(any(fnmatch.fnmatch("malware.py", pattern) for pattern in patterns))
            with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("missing cache")):
                with self.assertRaisesRegex(RuntimeError, "never downloads.*no procedural fallback"):
                    load_text_pipeline(model)

    def test_dataset_complete_text_provenance_and_binding_tamper_rejection(self):
        registry = fixture_registry()
        arrays, manifest = build_dataset(self.graph, train_count=2, heldout_count=2, seeds_per_profile=1, registry=registry)
        self.assertEqual(manifest["source"], "injected-test-pipeline")
        self.assertEqual({metadata["id"] for metadata in manifest["models"]}, {"qwen3-0.6b", "tiny-sd"})
        text = manifest["samples"][0]["generations"][0]
        self.assertEqual(text["text"], "Fixture aurora scene")
        self.assertEqual(text["seed"], node_seed(11, "copy"))
        self.assertEqual(registry.get("tiny-sd").calls[0]["prompt"], text["text"])
        faults = ("none", "text", "usage", "params", "seed", "model", "generations", "binding", "model-list", "source", "count", "profile", "prompt")
        with tempfile.TemporaryDirectory() as directory:
            for fault in faults:
                changed = copy.deepcopy(manifest)
                generation = changed["samples"][0]["generations"][0]
                if fault in ("text", "usage", "params", "seed", "model"):
                    del generation[fault]
                elif fault == "generations":
                    del changed["samples"][0]["generations"]
                elif fault == "binding":
                    changed["samples"][0]["generations"][1]["prompt"] = "different input"
                elif fault == "model-list":
                    changed["models"] = changed["models"][1:]
                elif fault == "source":
                    changed["source"] = "transformers-diffusers-local-open-weights"
                elif fault == "count":
                    changed["samples"] = 4
                elif fault == "profile":
                    del changed["samples"][0]["upl"]
                elif fault == "prompt":
                    generation["prompt"] = "wrong exposure-time prompt"
                output, unused_sidecar = write_dataset(Path(directory) / f"{fault}.npz", arrays, changed)
                if fault == "none":
                    load_dataset(output, self.graph, graph_hash(self.graph), allow_procedural=True)
                    with self.assertRaisesRegex(ValueError, "opt-in"):
                        load_dataset(output, self.graph, graph_hash(self.graph))
                else:
                    with self.assertRaises(ValueError, msg=fault):
                        load_dataset(output, self.graph, graph_hash(self.graph), allow_procedural=True)
        claimed = copy.deepcopy(manifest)
        claimed["source"] = "transformers-diffusers-local-open-weights"
        for metadata in claimed["models"]:
            metadata["backend"] = "transformers" if metadata["kind"] == "text" else "diffusers"
        validate_model_provenance(claimed, self.graph, False)

    def test_chained_text_and_raster_truncation_are_explicit(self):
        graph = copy.deepcopy(self.graph)
        second = copy.deepcopy(named(graph, "generate_text"))
        second["id"] = "rewrite"
        second["inputs"]["prompt"] = "copy"
        graph["nodes"].append(second)
        named(graph, "generate_image")["inputs"]["prompt"] = "rewrite"
        registry = fixture_registry()
        teacher = RasterTeacher(graph, registry)
        teacher.render([[0, 0, 0, 0]], seed=11)
        self.assertEqual(teacher.last_provenance[0]["generations"][1]["prompt"], "Fixture aurora scene")
        registry.get("tiny-sd").tokenizer.count = 78
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            teacher.render([[0, 0, 0, 0]], seed=11)
        del registry.get("tiny-sd").tokenizer
        with self.assertRaisesRegex(ValueError, "no tokenizer"):
            teacher.render([[0, 0, 0, 0]], seed=11)

    def test_stdin_bounded_exact_json_and_cli_output_channels(self):
        self.assertEqual(read_request(io.BytesIO(json.dumps(REQUEST).encode())), REQUEST)
        malformed = [b"{", b"{}", b"[]", b"x" * 65537, b"\xff", json.dumps(REQUEST).replace('"seed": 17', '"seed": 17, "seed": 18').encode()]
        for field, value in (("extra", 1), ("prompt", " "), ("prompt", "x" * 8193), ("model", "tiny-sd"), ("seed", True), ("seed", -1), ("seed", 2 ** 53), ("seed", 1.2), ("temperature", float("inf")), ("max_tokens", False)):
            malformed.append(json.dumps({**REQUEST, field: value}).encode())
        for payload in malformed:
            with self.assertRaises((ValueError, UnicodeError)):
                read_request(io.BytesIO(payload))
        registry = fixture_registry()
        with patch("sys.stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(REQUEST).encode()))), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch("sys.stderr", new_callable=io.StringIO), patch("angelos_ml.providers.PipelineRegistry", return_value=registry):
            main(["--stdin", "--device", "cpu", "--dtype", "float32"])
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["model"]["id"], REQUEST["model"])
        self.assertEqual(result["model"]["backend"], "injected-test-pipeline")
        self.assertEqual(result["node_seed"], str(node_seed(17, "text")))
        self.assertEqual(result["seed"], 17)
        with patch("sys.stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(REQUEST).encode()))), patch("sys.stdout", new_callable=io.StringIO) as stdout, patch("sys.stderr", new_callable=io.StringIO) as stderr, patch("angelos_ml.providers.load_pipeline", side_effect=RuntimeError("missing local weights")):
            with self.assertRaises(SystemExit) as failure:
                main(["--stdin"])
            self.assertEqual(failure.exception.code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("missing local weights", stderr.getvalue())
        process = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "angelos_ml.text", "--stdin"], input="{}", text=True, capture_output=True, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "ml")}, timeout=30)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stdout, "")
        self.assertIn("requires exactly", process.stderr)

    def test_raster_limits_include_flux_chat_template_and_textual_inversion(self):
        registry = fixture_registry()
        tokenizer = FixtureTokenizer()
        tokenizer.model_max_length = 10 ** 30
        tokenizer.apply_chat_template = Mock(return_value="rendered chat")
        tokenize = Mock(return_value={"input_ids": [1] * 513})
        class ChatTokenizer:
            model_max_length = 10 ** 30
            apply_chat_template = tokenizer.apply_chat_template

            def __call__(self, text, **arguments):
                return tokenize(text, **arguments)

        class FluxFixture:
            tokenizer = ChatTokenizer()

            def __call__(self, *, max_sequence_length=512):
                raise AssertionError("Preflight must not run inference")

        with self.assertRaisesRegex(ValueError, "512.*truncation"):
            registry.validate_raster_prompt("flux2-klein-4b", FluxFixture(), "short raw input")
        self.assertEqual(tokenize.call_args.args, ("rendered chat",))
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs["enable_thinking"])
        raster = registry.get("tiny-sd")
        raster.maybe_convert_prompt = Mock(return_value="expanded inversion")
        registry.validate_raster_prompt("tiny-sd", raster, "original")
        raster.maybe_convert_prompt.assert_called_once_with("original", raster.tokenizer)


if __name__ == "__main__":
    unittest.main()
