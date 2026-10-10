"""Local Qwen causal inference with private sampling RNG and bounded context."""

from importlib.metadata import version

import torch

from .models import default_cache_dir, model_spec
from .text_contract import MAX_INPUT_TOKENS, MAX_TEXT_UNITS, bounded_text, validate_text_result


class CausalTextPipeline:
    def __init__(self, model, tokenizer, device):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def __call__(self, *, prompt, system, max_tokens, temperature, generator):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False, truncation=False)
        input_ids = encoded["input_ids"].to(self.device)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not 1 <= input_ids.shape[1] <= MAX_INPUT_TOKENS:
            raise ValueError("Rendered text input exceeds 4096 tokens or is malformed; truncation is forbidden")
        input_count = input_ids.shape[1]
        context = self.model.config.max_position_embeddings
        if not isinstance(context, int) or input_count + max_tokens > context:
            raise ValueError("Input plus requested output exceeds the checkpoint context; truncation is forbidden")
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        if not isinstance(eos, (list, tuple)) or not eos or any(type(token) is not int or token < 0 for token in eos):
            raise ValueError("Checkpoint has no valid end-of-sequence token IDs")
        generated = []
        cache = None
        finish_reason = "length"
        for unused_step in range(max_tokens):
            prediction = self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache, use_cache=True, return_dict=True, logits_to_keep=1)
            if not isinstance(prediction.logits, torch.Tensor) or prediction.logits.ndim != 3 or prediction.logits.shape[0] != 1 or prediction.logits.shape[1] < 1 or prediction.logits.shape[2] < 1:
                raise ValueError("Causal model returned malformed logits")
            logits = prediction.logits[0, -1].to(device="cpu", dtype=torch.float64)
            if not torch.isfinite(logits).all():
                raise ValueError("Causal model returned nonfinite logits")
            if temperature == 0:
                token = int(logits.argmax().item())
            else:
                probabilities = torch.softmax((logits - logits.max()) / temperature, dim=-1)
                token = int(torch.multinomial(probabilities, 1, generator=generator).item())
            generated.append(token)
            if token in eos:
                finish_reason = "stop"
                break
            cache = prediction.past_key_values
            if cache is None:
                raise ValueError("Causal model did not return the required KV cache")
            input_ids = torch.tensor([[token]], device=self.device, dtype=torch.long)
            attention_mask = torch.cat((attention_mask, torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)), dim=1)
        text = self.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        bounded_text(text, MAX_TEXT_UNITS, "Generated text")
        return validate_text_result({
            "text": text.strip(), "usage": {"input_tokens": input_count, "output_tokens": len(generated)},
            "finish_reason": finish_reason,
        }, max_tokens)


def load_text_pipeline(model, device="cpu", dtype=None, cache_dir=None):
    specification = model_spec(model)
    if specification.kind != "text":
        raise ValueError("Causal loader requires a text model")
    try:
        from packaging.version import Version
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for package, minimum in (("torch", "2.6.0"), ("transformers", "5.18.0")):
            if Version(version(package)) < Version(minimum):
                raise RuntimeError(f"{package}>={minimum} is required")
        options = {
            "revision": specification.revision, "local_files_only": True,
            "trust_remote_code": False, "cache_dir": cache_dir or default_cache_dir(),
        }
        tokenizer = AutoTokenizer.from_pretrained(specification.repo, **options)
        causal_model = AutoModelForCausalLM.from_pretrained(
            specification.repo, **options, use_safetensors=True, dtype=dtype or torch.float32,
        ).to(device)
        return CausalTextPipeline(causal_model, tokenizer, device)
    except Exception as error:
        raise RuntimeError(
            f"Cannot load pinned local text model {model}: {error}. "
            f"Run python -m angelos_ml.models download {model} separately with the same cache directory. "
            "Runtime never downloads models and has no procedural fallback."
        ) from error
