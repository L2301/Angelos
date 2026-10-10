# Bounded native-HD Aurora baseline

This is a **pretrained local baseline**, not an Aurora/campaign-trained model,
production quality claim, UPL effectiveness result, or WebGPU implementation.
Aurora is fictional. Quiet/bold are synthetic adult demo labels, not real user
profiles; this standalone source is not connected to tracking, training or UI.
No paid inference, cloud jobs, provisioning, or remote generation is supported.

## Model and API

- Official `diffusers==0.40.0` `SanaSprintPipeline`.
- Repository: `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`.
- Immutable revision: `aa76e7f4f4928f378716b6716a2130fba3caf5b1`.
- **0.6B describes the transformer, not the entire pipeline**: it also loads
  Gemma 2 approximately 2.2B text-encoder parameters and DC-AE. The coordinating
  checkpoint audit reported 7.664839188 GB of repository files and non-gated
  access; that is not a runtime RAM/VRAM requirement or a download performed here.
- Model-card license: Apache-2.0, with additional Gemma terms/use restrictions.

The [official API documentation](https://huggingface.co/docs/diffusers/v0.40.0/en/api/pipelines/sana_sprint)
describes resolution binning followed by resizing. This runner explicitly passes
`use_resolution_binning=False`. It requests **1920×1088**, observes latent tensors
`[1,32,34,60]` at every step, requires native RGB output of exactly that size, and
center-crops `(0,4,1920,1084)` to **1920×1080**, removing four rows at each edge.
It never resizes, upscales, or silently switches to smaller generation. Learned
DC-AE decoding/tiling is part of the native pipeline, not a post-render upscaler.
The checkpoint is 1024px-based: native HD dimensions do **not** establish visual
quality at this size.

Generation requests `output_type="np"` so NaNs cannot silently become black PIL
images. Before PNG conversion, floating output must be `[1,1088,1920,3]`, finite,
within `[0,1]`, and have mean per-channel spatial variance greater than `1e-6`.
Constant colors and nearly constant outputs fail. This is only a degeneracy check,
**not evidence of good quality**. PNG uses the usual `round(pixels*255)` uint8
quantization, without corrective clipping or normalization. Float statistics and
the quantization rule are recorded in the manifest.

The installed 0.40.0 implementation and pinned `model_index.json` were inspected.
Its real `SCMScheduler` requires `intermediate_timesteps=None` for four steps;
two steps use `1.3`. Its DC-AE supports 512×512 tiling (448-pixel strides) and
the pipeline casts decoded latents to the VAE dtype. The loader uses bfloat16
transformer/text encoder and float32 VAE, without dtype/device fallback.
The [pinned model card and component metadata](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/tree/aa76e7f4f4928f378716b6716a2130fba3caf5b1)
identify Gemma2Model, SanaTransformer2DModel, AutoencoderDC and SCMScheduler.

## Explicit commands

Run from the repository root, using the existing `.venv` and generative
requirements. The wrapper does not require a package.json change.

```sh
node scripts/hd.js list
# Only if the coordinating process has NOT already downloaded this pin:
node scripts/hd.js download sana-sprint-0.6b
# One image, two steps from the example, explicit local accelerator:
node scripts/hd.js generate examples/aurora-hd.json --device cuda
# Explicitly authorize at most two images; choose a NEW directory:
node scripts/hd.js generate examples/aurora-hd.json --device cuda --variant both --max-images 2 --output-dir artifacts/hd/aurora-pair-001
```

`PYTHONPATH=ml .venv/bin/python -m angelos_ml.hd` exposes the same subcommands.
`--cache-dir` is available for download/generate; the default reuses
`HF_HUB_CACHE`, or `HF_HOME/hub`, or `artifacts/huggingface/hub`.
**Do not start a duplicate download while the parent is fetching the pin.**
Download allows only named metadata/tokenizer assets and safetensors patterns,
never Python/source code or pickle weights. Generate resolves only the immutable
local snapshot and uses `local_files_only=True`, `use_safetensors=True`, and
`trust_remote_code=False`; it cannot download missing weights. Missing or partial
cache, unavailable devices, OOM, operator errors, and wrong dimensions fail rather
than retrying, falling back, or claiming success.

## Source contract and budget

`examples/aurora-hd.json` is a versioned standalone source, not graph IR. It has
fixed fictional/synthetic-adult declarations, allowlisted model, exact sizes in
`[width,height]` order, `steps` **2 or 4**, one or two unique quiet/bold variants
with prompt and integer seed, and `output_dir`. Unknown fields and duplicate JSON
keys fail. Prompts are at most 4096 UTF-8 bytes and 300 tokenizer tokens; silent
truncation is forbidden. The official pipeline lowercases/strips the prompt; the
manifest includes original and effective text. Hidden complex-prompt instructions
are explicitly disabled.

Default selection is `--variant quiet --max-images 1`. `--variant bold` selects
one other profile; `--variant both` requires `--max-images 2` for the example.
The hard limit is **two images, four steps each, batch size one**. There are no
retries, sweeps, training, or unattended runs. This bounds work, not wall time or
peak memory. An existing output directory, even empty, is rejected; use a fresh
`--output-dir` for every attempt. Partial failures remain marked `failed`.

`--device cuda`, `--device mps`, or `--device cpu` is mandatory. CPU is opt-in and
may be very slow; it is never described as GPU success. The coordinating process
completed one separate native MPS probe at this pin: seed 2026, two steps,
bfloat16 transformer/text encoder, float32 tiled VAE, 1920×1088 followed by the
four-row-per-edge crop, approximately 135.1 seconds including loading. Its local
report is `/tmp/angelos-aurora-hd-probe.json`; this is not a run of this CLI or its
new float-validation path, and is not a quality assessment. MPS still has
**hardware-dependent operator/memory risk**: bfloat16 and every Gemma/Sana/DC-AE
operator must work on the installed macOS/PyTorch combination.
`PYTORCH_ENABLE_MPS_FALLBACK=1` is rejected.
512-pixel VAE tiling reduces decoding working memory but does not guarantee the
whole pipeline fits. CPU RNG generates seeded noise for cross-device-compatible
generators; all learned model components stay on the explicitly selected device.
Identical seeds do not guarantee bitwise equivalence across hardware/versions.

## Artifacts and tests

Each image keeps the untouched native decoded RGB PNG plus the crop-only PNG.
`manifest.json` records status, full source and its canonical SHA-256, immutable
model revision SHA, model-index SHA-256, component classes/dtypes/devices,
dependency versions, complete call settings, seed, prompt, observed latent/native/
output shapes, exact crop, artifact SHA-256 hashes and elapsed time. Revision SHA
is checkpoint provenance, not an independently computed digest of all weight
bytes. The manifest makes no model-quality assertion; only `status=complete`
with `test_only=false` describes a completed real local baseline run.

```sh
PYTHONPATH=ml .venv/bin/python -m unittest discover -s ml/tests -p test_hd.py -v
```

Tests are offline: contract/crop/budget checks, mocked pinned loading, and the
actual installed SanaSprintPipeline/SCMScheduler control flow with fixture model
components. They do not load pretrained weights or establish image quality or GPU
success. Any injected pipeline requires explicit `test_only=True`; its filenames
and manifest are unmistakably TEST_ONLY, even when its outer pipeline class is
the real Diffusers class. There is no fixture CLI flag.

For other local code, `angelos_ml.hd_models.load_pipeline(model, device,
cache_dir=None)` returns `(pipeline, runtime_metadata)`. It does not generate,
train, or load adapters. It sets the three learned components to evaluation mode;
an explicitly separate training owner must manage its own trainable parameters,
gradients and artifact provenance. The inference loader's immutable base-model
scope must not be expanded to claim that adapters are the pinned checkpoint.
