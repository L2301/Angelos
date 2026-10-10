# Local text generation

## Fixed subprocess interface

Run from the repository root with `PYTHONPATH=ml`:

```sh
.venv/bin/python -m angelos_ml.text --stdin --device cpu --dtype float32
```

Send exactly one UTF-8 JSON object on stdin, close stdin, and require successful exit:

```json
{"model":"qwen3-0.6b","prompt":"Write a short aurora scene.","system":"Return only concise scene copy.","max_tokens":128,"temperature":0.7,"seed":17}
```

All six fields are required; extra/duplicate keys, nonfinite numbers and booleans in numeric fields fail. Model is `qwen3-0.6b` or `qwen3-8b`. Prompt is nonblank, at most 8192 UTF-16 code units; system is a nonblank literal, at most 2048 UTF-16 code units. `max_tokens` is an integer 1–512; `temperature` is finite 0–2 (0 means greedy); `seed` is an integer 0–9007199254740991. The entire stdin payload is at most 65536 bytes. No repository, revision, device, cache path, or executable comes from JSON.

`--device` accepts `cpu`, `mps`, or `cuda`; `--dtype` accepts `float32`, `float16`, or `bfloat16`. Defaults are CPU/float32. Optional operator-controlled `--cache-dir` selects the local Hugging Face cache. Runtime never downloads. Failures print diagnostics to stderr, exit 2, and emit no stdout result. Successful stdout is one JSON object with `text`, `model` (pinned metadata), `seed`, `parameters` (`system`, `max_tokens`, `temperature`), `usage` (`input_tokens`, `output_tokens`), and `finish_reason` (`stop` or `length`). Text is nonblank and at most 8192 UTF-16 code units. Treat generated text as untrusted data, never HTML or executable input. A server should independently bound process output, concurrency and wall time and reject malformed responses/nonzero exits.

Input is never silently truncated: the rendered chat must fit 4096 tokens, and input plus requested output must fit the checkpoint context. Only newly generated tokens are decoded. Output reaching the token budget is returned with `finish_reason: "length"`; invalid, empty or oversized decoded output fails. Thinking is disabled through Qwen's chat template. Sampling uses a private per-call CPU generator, never process-global seeding.

`seed` echoes the user seed. Additional `node_seed` is derived with the existing `angelos.teacher-seed.v1` algorithm and fixed standalone node ID `text`. Graph executions derive their seed from the actual node ID. Derived seeds are signed-64-bit integers and may exceed JavaScript's safe-integer range, so the standalone CLI/API serializes `node_seed` as a decimal string without precision loss. Replay uses the original user `seed`. `usage.output_tokens` includes the EOS token when present. Sampling is greedy at zero, otherwise categorical softmax over the full vocabulary at the requested temperature; no hidden top-k/top-p policy. Reproducibility is within the same model/device/dtype/software stack, not a promise of cross-device bit parity.

Generated text passed to image/video generation is checked against every pipeline tokenizer and its sequence-length default before inference; overflow or an unverifiable context fails rather than allowing Diffusers to truncate. The legacy deterministic `prompt` path retains its previous behavior. A short `max_tokens` and explicit concise-copy instruction are recommended for CLIP-backed image models.

## Graph contract

`generate_text(prompt: source, model: "qwen3-0.6b", system: "Return concise scene copy.", max_tokens: 128, temperature: 0.7)` emits type `text`. Its prompt port, and image/video generation prompt ports, explicitly accept `prompt` or `text`. The existing deterministic `prompt` node remains unchanged; graph output must still be an image. No text student or browser text inference is claimed. Text is a teacher intermediate with complete input/output, parameters, seed and model provenance.

Mixed real-model datasets identify `source: "transformers-diffusers-local-open-weights"` and each model's `transformers`/`diffusers` backend. Any injected adapter forces `injected-test-pipeline` provenance and explicit fixture opt-in during training. Manifest validation includes every text model binding, output usage/finish reason, and text-to-downstream-prompt equality. Provenance consistency is not independent evidence that supplied outputs came from those weights.

## Pinned upstream sources

Both checkpoints use Apache-2.0 and Transformers `AutoModelForCausalLM` plus `AutoTokenizer`, local files only, remote code disabled, safetensors required:

- `qwen3-0.6b`: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`; [upstream API](https://huggingface.co/api/models/Qwen/Qwen3-0.6B), [immutable model card](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md).
- `qwen3-8b`: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`; [upstream API](https://huggingface.co/api/models/Qwen/Qwen3-8B), [immutable model card](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/README.md).

Explicit installation is separate: `PYTHONPATH=ml .venv/bin/python -m angelos_ml.models download qwen3-0.6b`. Download patterns include root-level safetensors and sharded safetensors plus tokenizer/config files. Do not interpret offline injected fixtures as actual model inference. Weight downloads and real-model smoke tests are coordinated separately.

Dependencies are the existing `ml/requirements-generative.txt` environment; the text loader requires torch >= 2.6 and Transformers >= 5.18. The alias `node scripts/text.js --stdin --device cpu --dtype float32` forwards stdin/stdout and explicitly sets Hugging Face/Transformers offline mode. Model memory and latency depend on the selected checkpoint, device and dtype; no quantization or automatic device fallback is provided.

## Offline validation

```sh
node --test test/graph.test.js
PYTHONPATH=ml .venv/bin/python -m unittest discover -s ml/tests -p test_text.py -v
PYTHONPATH=ml .venv/bin/python -m unittest discover -s ml/tests
```

Tests include injected tokenizers/models, a tiny randomly initialized Qwen architecture solely as an API/cache fixture, malformed outputs, exact CLI I/O, JS/Python graph parity, legacy hash snapshots, per-node RNG isolation/concurrency, rendered prompt limits, and dataset provenance tampering. These do not establish pretrained-copy quality or GPU execution. Actual pinned-weight checks are recorded below, separate from fixtures.

## Local HTTP probe

The lab's manual **Generate text** button posts to `/api/text-preview`. The endpoint
accepts the same six fields as the CLI, but intentionally limits both prompt and
system to 2048 UTF-16 code units. It starts one fixed Python subprocess with JSON
on stdin, no shell, offline mode, a 120-second timeout and bounded output. Only one
request runs at a time; concurrent calls return 429. Cancel/disconnect terminates
the worker. Missing Python/cache and model errors are visible; no replacement
text is invented. The result is not stored as a profile, impression or training
record. Output is rendered as text, never HTML.

The server defaults to CPU/float32. Operator environment variables
`ANGELOS_TEXT_DEVICE` and `ANGELOS_TEXT_DTYPE` change the worker configuration;
requests cannot select executable paths, repositories or downloads. This is a
loopback development endpoint, not an authenticated public inference service.

## Actual pinned-weight execution — 2026-10-08

Qwen3-0.6B weights were explicitly downloaded at the revision above. The standalone
CLI and actual Chrome → local HTTP worker both produced:

> A quiet alpine morning, a moss-green backpack, and natural light.

Inputs: system `Write concise, grounded creative direction. Return only the requested copy. Do not invent product claims.`;
prompt `Write one short visual prompt for a fictional Aurora hiking campaign: a quiet alpine morning, a moss-green backpack, natural light. No text or logos in the image.`;
seed 2026, temperature 0.7, max_tokens 96. Result: 70 input tokens, 16 output tokens,
finish reason `stop`; derived standalone node seed `5164180612389070700`.
The HTTP run took 54.1 seconds including process/model loading under concurrent
test load. This is execution evidence, not an isolated latency benchmark.

`examples/aurora-text.ang` also executed end to end using actual Qwen3-0.6B and
Tiny-SD weights, CPU/float32, synthetic UPL `[0.4, 0, 0.72, 0.38]`, seed 2026.
The `copy` node produced:

> A radiant aurora dances above the mountains, shimmering with colors that change with the sky’s hues.

That exact text reached Tiny-SD's prompt. Qwen used 45 input / 22 output tokens,
finish reason `stop`; the image had shape `[1,3,256,256]`, finite pixels, 12
diffusion steps and guidance 7.5. Both pinned model identities, node seeds,
parameters and text output were present in provenance. Total observed execution
was 196.5 seconds including loads under concurrent test load. This single image
does not establish ad quality, conditioning quality or distillation success.

**Known device failure:** the same mixed graph on Apple MPS/float32 failed in the
Qwen stage with `Causal model returned nonfinite logits`. The runtime rejected
it rather than returning invalid text or silently falling back. Use CPU for this
demonstrated text graph. CUDA, MPS text fixes, Qwen3-8B weights, quantization and
text-conditioned student training were not validated in this run.
