# Open-weight teachers and diffusion student

This report supersedes the original MLP milestone in `EXECUTION.md`. The measured
path uses synthetic adult UPLs, local open weights and no paid APIs or ad traffic.

## Implemented separation

1. Source/typed DAG compiler validates and hashes the mixed program.
2. Registered, pinned Diffusers pipelines execute generative nodes. Tensor operations
   run on the selected teacher device. Prompt construction and pipeline orchestration
   remain host operations; this is not a fused CUDA executable.
3. Dataset collection renders the entire native-resolution graph, including branding,
   then downsamples to the student's native raster. Profiles are partitioned before
   sampling; repeated seeds for a profile never cross training/held-out splits.
4. A freshly initialized conditional U-Net learns DDPM noise prediction on those
   complete outputs. There is no pretrained student, human critic or CTR objective yet.
5. The exported artifact performs 16-step DDIM denoising in browser WebGPU. Its
   explicit CPU reference is for tests, not a hidden serving fallback.

## Model and device evidence

- Real Tiny-SD weights: `segmind/tiny-sd` at
  `cad0bd7495fa6c4bcca01b19a723dc91627fe84f`, approximately 1.1 GB download.
- Mac with 24 GiB unified memory; PyTorch 2.14.1, Diffusers 0.40.0,
  Transformers 5.18.0. Apple MPS is available outside the terminal sandbox.
- An initial live 256×256, 12-step image generated successfully in 12.55 seconds
  including pipeline loading; this is an observation, not a stable latency guarantee.
- Deterministic 256×256 full-graph MPS execution was independently compared with
  CPU: maximum channel error `1.7881393432617188e-7`, including brand composition.
- FLUX.2 Klein 4B and LTX-Video adapters, signatures, type/shape/seed contracts and
  failure behavior are implemented and tested offline. Their full weights were
  **not downloaded or executed** on this machine. CUDA was not tested.

The initial corpus ran generation on MPS and deterministic postprocessing on CPU.
It is retained locally under `aurora-cpu-postprocess.dataset.*`, not selected for the
final student. The final corpus is regenerated with GPU-resident image operations
and native PNG previews under `artifacts/teacher-native/`.

The final corpus contains 40 genuine Tiny-SD full-graph samples: 16 training UPLs
and 4 held-out UPLs, with two teacher seeds each. All sample provenance records
`deterministicTensorDevice: mps:0`. The graph hash is
`f0a99add82b7caab74c1ea45c5af801ed4cecfbae208c2ec7972e06b644987bd`.

## Actual browser validation

Chrome reported Apple `metal-3`, not a fallback adapter. The exported learned
student—not an untrained test double—passed the following numerical checks:

| Check | Maximum absolute error |
| --- | ---: |
| WebGPU versus PyTorch fixture | `0.0000139475` |
| Explicit JS CPU versus PyTorch | `0.0000141859` |
| WebGPU versus JS CPU | `0.0000125170` |
| Exact composited brand | `0` |
| Same-seed GPU replay | `0` |

Byte output differed by at most one level. Different seeds produced distinct
images; native-to-display nearest-neighbor scaling matched exactly. Tests also
exercise single-flight rendering, disposal, malformed artifacts, unavailable GPU,
device errors, exposure qualification, seed identity, preset graph loading and
consent revocation.

Warm benchmark, 20 completed renders after 3 warmups: 32×32 native generation,
192×192 displayed raster, 16 DDIM steps; p50 **14.4 ms**, p95 **17.5 ms**.
Renderer initialization measured 9.4 ms. Artifact JSON is 478,590 bytes; FP32
parameters occupy 90,828 bytes. Timing includes GPU completion/readback, not
network download, CPU noise/mask preparation, canvas presentation or full cold
start. No latency guarantee or physical-GPU peak-memory measurement is implied.

Validation: **104 JavaScript tests**, **70 Python tests**, and **8 actual Chrome
tests passed**. One obsolete MLP browser-parity test is deliberately skipped when
the diffusion artifact is selected; the separate diffusion parity test ran.

The final audit additionally tightened exact model/revision provenance, aligned
diffusion brand-boundary arithmetic with PyTorch float32, and corrected queued
render authorization, graph-write ordering and renderer replacement on graph changes.

## Quality boundaries

### Measured student run

- Artifact: `aurora-diffusion-5ee99096c31e4c6ea572b98b4e572aca`.
- 22,707 trainable parameters; 2,000 Adam steps on MPS, batch 16, seed 2026.
- Trainer elapsed time including evaluation: 32.17 seconds, excluding imports.
- Held-out noise MSE: `1.0108144 → 0.1128036`.
- **Conditioning audit failed:** mismatched-UPL noise MSE `0.1116711` was slightly
  better than correct-UPL MSE. Useful personalization has not been demonstrated;
  numerical pixel changes alone are insufficient. This is visible in the UI.
- Raw brand maximum error `0.8961216` exceeds `0.06`: exact compositing selected.
- Measured composited brand error: `0`. Seed-diversity MSE: `0.0458484`.
- Weights are learned from the real full-graph corpus, with no procedural fixture
  substitution, no human preference reward, no CTR learning and no production promotion.

The student is a 32×32, five-convolution U-Net. Enlarging its raster for display
does not add detail. Forty teacher images are a small execution experiment, not
an adequate campaign dataset or evidence of useful ad quality. Noise-prediction
MSE measures denoising training, not image quality or conversion improvement.

Seed replay fixes initial Gaussian noise but does not promise bitwise equality
across devices. A seed-diversity/UPL-sensitivity measurement only shows numerical
dependence, not meaningful or desirable personalization. Exact brand fallback
does not repair deficiencies elsewhere in the generated image.

Improving student quality requires a larger corpus, appropriate pretrained
initialization or latent-space model, more capacity and independent perceptual/
brand evaluations. Learned preference critics, nightly click optimization,
sampling-step distillation, quantization and production serving remain unfinished.
