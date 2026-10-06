# Open-weight graph teachers

This document covers the **currently implemented** three-model allowlist only.
For the broader, capability-by-capability research and proposed replacements, see
[the full creative-model audit](MODEL-COVERAGE.md). Research entries are not executable adapters.

Graph IR, raster teacher execution, dataset collection, student training, and deployment are separate stages. Generative graph nodes call actual local Diffusers pipelines; they never substitute gradients, noise textures, synthetic test pixels, or an untrained network for a pretrained model.

## Pinned allowlist

Only these model IDs are accepted. Repository names and revisions cannot be supplied by a graph.

| Graph model ID | Repository | Immutable revision | Adapter | Weight format |
| --- | --- | --- | --- | --- |
| `tiny-sd` | `segmind/tiny-sd` | `cad0bd7495fa6c4bcca01b19a723dc91627fe84f` | `StableDiffusionPipeline` | `.bin`, restricted weights-only loading |
| `flux2-klein-4b` | `black-forest-labs/FLUX.2-klein-4B` | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | `Flux2KleinPipeline` | safetensors |
| `ltx-video` | `Lightricks/LTX-Video` | `8984fa25007f376c1a299016d0957a37a2f797bb` | `LTXPipeline` | safetensors |

Review the model cards and licenses before downloading or distributing derivatives: [tiny-sd, CreativeML OpenRAIL-M](https://huggingface.co/segmind/tiny-sd/tree/cad0bd7495fa6c4bcca01b19a723dc91627fe84f), [FLUX.2 klein 4B, Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294), and [LTX-Video, custom open-weights license](https://huggingface.co/Lightricks/LTX-Video/tree/8984fa25007f376c1a299016d0957a37a2f797bb). Open weights do not imply identical licensing terms.

Adapters require Diffusers >=0.40.0, Transformers >=5.18.0, and PyTorch >=2.6.0. The development environment uses torch 2.14.1, Diffusers 0.40.0, Transformers 5.18.0, Accelerate 1.15.0, and Pillow 12.3.0. Keep these dependencies patched.

Tiny-sd's pinned revision has no safetensors weights, so its adapter explicitly uses `use_safetensors=False`. Modern [Diffusers loading code](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/models/model_loading_utils.py) calls `torch.load(weights_only=True)`; Transformers also defaults to restricted weights-only loading. No unrestricted pickle fallback, custom pipeline, remote Python code, arbitrary repository, or single-file checkpoint path is enabled. Restricted loading reduces arbitrary-code risk; it does not make hostile weight files generally harmless. Use only the pinned trusted cache, not untrusted replacements. The upstream tiny-sd snapshot has no safety checker; none is silently downloaded or claimed. Generated content still requires review.

## Download separately

From the repository root:

```sh
PYTHONPATH=ml .venv/bin/python -m angelos_ml.models list
PYTHONPATH=ml .venv/bin/python -m angelos_ml.models download tiny-sd
```

`model-download` remains an alias for `download`. Downloads use the allowlist's exact revision and fetch model/config/tokenizer/license files, not executable repository code. The default cache is `artifacts/huggingface/hub`; `HF_HOME` and `HF_HUB_CACHE` are respected. Both model download and dataset CLIs accept `--cache-dir` to override the Hub cache directory. This is a cache directory, not a replacement checkpoint path.

Runtime always passes `local_files_only=True` and the immutable revision to `from_pretrained`. An incomplete cache, loader error, unsupported operation, or resource failure is fatal and actionable. Graph execution never runs the download command. FLUX and LTX weights are large; do not download them merely to run offline tests. Existing local credentials may be needed if an upstream repository requires access.

## Typed source operations

```text
scene = prompt(condition: openness, low: "Quiet aurora", high: "Vivid aurora", threshold: 0.5)
picture = generate_image(prompt: scene, model: "tiny-sd", steps: 12, guidance: 7.5, width: 256, height: 256)
```

- `prompt`: scalar → prompt text. Values below `threshold` select `low`; equality and above select `high`. Both alternatives must be nonempty and at most 2048 UTF-16 code units. The only interpolation token is `{condition}`, replaced with six decimal places. For numeric-only conditioning, use the same template in both alternatives. No expressions, arbitrary formatting, evaluation, or external text retrieval.
- `generate_image`: prompt → image. Models: `tiny-sd`, `flux2-klein-4b`. All of `model`, `steps`, `guidance`, `width`, and `height` are required.
- `generate_video`: prompt → video. Model: `ltx-video`. Requires the same parameters plus `num_frames`.
- `frame`: video → image, with required zero-based `index`. This lets a video teacher feed existing image transforms and image output.

Steps must be integral in `[1,100]`, guidance finite in `[0,20]`, and dimensions multiples of 32 in `[256,512]`. LTX frame counts must be `8n+1`, at most 121. Frame indices must be inside the declared video. The bounded dimensions are this application's resource policy, not the full capability of upstream models.

The [FLUX.2 Klein API](https://huggingface.co/docs/diffusers/api/pipelines/flux2) uses `num_inference_steps`, `guidance_scale`, and an explicit `generator`; guidance is ignored by distilled Klein variants. The [LTX API](https://huggingface.co/docs/diffusers/api/pipelines/ltx_video) uses the same routing plus `num_frames`. Providers request `output_type="pt"` and validate actual tensor dimensions, finiteness, and `[0,1]` pixels. No adapter silently resizes a malformed generator result.

`examples/aurora-generative.ang`, `examples/aurora-flux.ang`, and `examples/aurora-ltx.ang` all retain campaign ID `aurora`. Tiny-sd uses 12 steps at 256px; FLUX uses 4 steps at 512px; LTX uses 20 steps and 9 frames at 512px, then selects frame 4. Each executes exposure, vignette, and the terminal brand after generation.

## Dataset CLI

```sh
node src/cli.js compile examples/aurora-generative.ang artifacts/aurora-generative.graph.json
PYTHONPATH=ml .venv/bin/python -m angelos_ml.dataset \
  --graph artifacts/aurora-generative.graph.json \
  --output artifacts/aurora-generative.npz \
  --device mps --dtype float32 \
  --train-count 16 --heldout-count 4 --seeds-per-profile 2 \
  --seed 7 --teacher-seed 11
```

The CLI defaults to 16 training profiles, 4 held-out profiles, 2 seeds per profile (40 images), and CPU float32. Generation is explicit and can be slow. For a smoke test use `--train-count 1 --heldout-count 1 --seeds-per-profile 1`. Device choices are `cpu`, `mps`, and `cuda`; dtype choices are `float32`, `float16`, and `bfloat16`. Hardware and model support are the caller's responsibility; there is no silent device/precision fallback. Tiny-sd enables attention slicing. FLUX/LTX may require substantially more memory, especially at float32; quantization and automatic offload are not implemented.

Optional `--graph-hash HASH --graph-canonical FILE` validates compiler identity before execution. `--graph-canonical` requires `--graph-hash`. `--threads` controls CPU threads (default 2, range 1–8). `--width` and `--height` select the native raster size for deterministic-only graphs (default 256); generative nodes always own their declared dimensions. The complete graph runs at native generation resolution, including its terminal brand, and only then area-downsamples to 32×32. No overlay is added after downsampling. Generation output, deterministic torch image transforms, and downsampling remain on the selected device, with CPU transfer only for file serialization. This is ordered eager tensor execution, not a fused or compiled whole graph. Manifests record `deterministicTensorDevice` globally and per sample.

Optional `--save-previews NEW_DIRECTORY` preserves each native-resolution full-graph result as `sample-000000.png`, etc. The directory must not already exist. The manifest records each PNG path. These are rounded 8-bit RGB previews, not training inputs or raw pre-transform generator outputs. A failed collection can leave partial preview files; no completed dataset is claimed in that case.

UPL profiles are synthetic adults only. Unique profiles are partitioned before any generation. Repeated stochastic samples from the same UPL always remain in one split; a second random seed never moves a profile into holdout. Profile generation and sampling use local explicit generators, not the process-global RNG. Dataset generation never trains a student.

The output is a non-pickled `.npz` archive:

| Array | Shape | Type / meaning |
| --- | --- | --- |
| `images` | `[N,3,32,32]` | float32, RGB `[0,1]`, full-graph target |
| `upl` | `[N,4]` | float32, normalized structured-demo-v1 profile |
| `split` | `[N]` | int64, `0` train / `1` holdout |
| `teacher_seed` | `[N]` | int64, sample execution seed |

The sidecar is `NAME.manifest.json` (replaces `.npz`, does not append to it). It binds graphHash, encoder, complete graph, model IDs/repos/immutable revisions, backend/device/dtype/weight-loader metadata, package versions, source, `synthetic: true`, profile split policy, and every sample's UPL, split, teacher seed, native shape, resolved prompt, node seed, and generation parameters. Existing dataset or manifest files are not overwritten. Generation failure writes neither file. Output is assembled in memory; very large requested counts can exhaust memory. Disk failure during the final two-file write can leave a partial artifact; validate both files before consuming.

## Python API

```python
from angelos_ml.providers import PipelineRegistry
from angelos_ml.teacher import RasterTeacher

registry = PipelineRegistry(device="mps")
teacher = RasterTeacher(graph, registry=registry)
images = teacher.render([[0.2, 0.3, 0.7, 0.4]], seed=11)
```

`render` accepts `[N,4]` or one `[4]` profile and returns float32 `[N,3,H,W]` on `registry.device` at native output resolution. `seed` is mandatory: either one nonnegative JavaScript-safe integer (successive batch rows use successive seeds) or an explicit seed list matching the batch. Each stochastic node derives its independent CPU generator seed from SHA256 of `angelos.teacher-seed.v1:{sampleSeed}:{nodeId}`, taking the first eight bytes big-endian modulo 2^63. Diffusers transfers generator-created noise to the execution device; raster image operations stay on that device. Node ordering and unrelated RNG consumption cannot change the seed mapping. Reproducibility across different hardware, precision, dependency versions, or nondeterministic kernels is not guaranteed.

`teacher.last_provenance` describes the latest batch. `build_dataset(...)` returns `(arrays, manifest)`; its Python API retains small defaults of 2 training profiles, 1 held-out profile, and 1 seed per profile for programmatic smoke tests. CLI defaults are deliberately larger. `write_dataset(...)` persists that pair. Registry injection is available for offline unit tests only: `PipelineRegistry(pipelines={"tiny-sd": fake_pipeline})`. Such manifests explicitly say `injected-test-pipeline`, never actual pretrained inference.

## Validation and limitations

- Offline tests exercise exact adapter arguments/local-only loading, prompts, type/shape/range failures, seed repeatability, frame selection, all deterministic raster operations, native-resolution branding, split isolation, cross-language graph hashes, NPZ schema, and provenance. Injected fake pipelines validate routing, not model quality or live generation.
- A separate offline MPS test passed outside the sandbox, verifying device-resident postprocessing and dataset serialization with injected pixels. The sandbox skips this test when MPS is unavailable. This is GPU tensor execution evidence, not evidence that an injected pipeline generated real model images.
- A separate live tiny-sd run was reported successful on MPS using the pinned cached snapshot, float32, 256×256, 12 steps, guidance 7.5, and an explicit CPU generator. This does not validate FLUX or LTX, dataset quality, student fidelity, production use, or CUDA support.
- FLUX/LTX adapters follow the official APIs, but full-weight inference has not been validated here. No full FLUX/LTX weights were downloaded for these tests.
- Browser reference execution rejects generative nodes with instructions to use the Python raster teacher. Browser compiler callers can use `assertDeterministicGraph` for the same guard. Source-to-IR compilation still accepts these nodes; it is not model compilation.
- Coordinate `TensorTeacher` and strict ATen teacher export reject generative graphs. The complete pretrained pipelines have not been compiled to ATen. There is no procedural substitute, interpreter compilation claim, or implicit training.
- The dataset is a small synthetic demonstration, not a production creative-quality, personalization, fairness, conversion, or safety evaluation. A tiny 32px student cannot be assumed to preserve full-resolution text or logo fidelity.
