# Angelos

**GPU Accelerated Ads — a proof of concept by Louis Anderson.**

**Author a creative graph. Execute a teacher. Distill one diffusion model per campaign.**

Angelos is a local research laboratory for UPL-conditioned creative generation.
Its typed graph combines deterministic tensor image operations with stochastic,
locally executed open-weight text/image/video models. The browser runs a separate,
campaign-specific conditional diffusion student in WebGPU.

This is not a finished ad-generation platform. The browser student remains a
**32×32 regression model**. A separate SANA-Sprint baseline now produces genuine
**1920×1080** imagery locally; that baseline is not yet an Angelos campaign-trained
student. [Recorded HD evidence](docs/HD-EXECUTION.md) distinguishes those claims.

The [full creative-model capability audit](docs/MODEL-COVERAGE.md) covers the
broader intended graph: image editing/control, video, audio, avatars, language,
documents, vector/3D assets and deterministic operations. Its sourced candidates
are research recommendations, not a claim that these adapters are implemented.

## Quick start

Explore the idea with **Node 24+ only**. No Python, model download or GPU is needed
for the explanatory homepage, recorded evidence, or graph/source exploration.

```sh
npm ci
npm start
```

Open `http://127.0.0.1:4310` for the concept site, or `/lab` for the local laboratory.
Missing student weights are an explicit state, not substituted model output.
The server binds only to loopback; its unauthenticated API must not be deployed publicly.

For public sharing, use the [static-only GitHub Pages deployment](docs/DEPLOYMENT.md).
The hosted lab link explains local setup; it does not expose the lab API or profiles.

### Use ComfyUI to create the ad

The local lab can also convert a [supported ComfyUI API graph into a typed program](docs/COMFYUI.md#workflow-to-program), then compile it back to API JSON. This preserves execution inputs rather than substituting another model. Use **Convert a ComfyUI graph to a program** in the lab, or `npm run comfy:program -- import INPUT.json --output NEW_OUTPUT.comfy.ang`. Unsupported nodes fail explicitly. This separate Comfy backend dialect is not the student-training language; translation does not train weights or execute the workflow.

Keep your existing ComfyUI canvas, model loaders, samplers and compositing nodes.
Install the local [Angelos custom nodes](integrations/comfyui_angelos/README.md),
then export the **finished** creative with campaign/audience metadata and workflow
provenance. [The walkthrough](docs/COMFYUI.md) explains the handoff and its limits.
Angelos does not silently execute imported workflows, install their dependencies,
or claim that exporting images has already distilled the graph into weights.

The built-in typed graph remains a smaller inspectable alternative, not a
replacement for ComfyUI. The lab walks from making the ad to audience variants,
review, campaign training and candidate evaluation, in that order.

### High-resolution baseline and learning experiments

See [HD generation](docs/HD.md), [reward-alignment experiments](docs/HD-ALIGNMENT.md),
and [the click + critic candidate loop](docs/RL.md). All are explicit operator
actions. No paid jobs, nightly scheduler or automatic promotion is enabled.

Research reviews: [diffusion](docs/RESEARCH-DIFFUSION.md),
[distillation](docs/RESEARCH-DISTILLATION.md), [RL](docs/RESEARCH-RL.md),
[advertising identity and richer UPL design](docs/RESEARCH-UPL.md).
The four-coordinate encoder remains a **legacy regression fixture**. The new
[UPL v2 path](docs/UPL.md) separates richer declared facts, expiring campaign-scoped
events, a pinned local Qwen3 semantic encoder, and model-specific conditioning.
The lab supports 128/256/**512**/1024-dimensional embeddings, immutable handoffs to
ComfyUI, and explicit missing-checkpoint errors. It never pads four coordinates
or silently feeds a new embedding into old student weights. The pretrained
encoder is a semantic baseline, not a click-trained personality model; the HD
student does not yet have a trained UPL adapter.
See [actual local encoder and browser execution](docs/UPL-EXECUTION.md), including
the synthetic 128/256/512/1024-dimensional comparison and its limitations.

### Real local text generation

`prompt()` formats/selects text deterministically. **`generate_text()` runs a
pretrained Qwen model** and emits a separate `text` type; image/video prompt ports
can consume either. The lab includes a manual text probe and a text → image preset.
Neither loading that preset nor visiting the site downloads or runs teacher models.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements-generative.txt
npm run models -- download qwen3-0.6b
```

Then use **Generate text** in `/lab`. CPU/float32 is the explicit default;
`ANGELOS_TEXT_DEVICE` and `ANGELOS_TEXT_DTYPE` configure the server's text worker.
See [text execution and limits](docs/TEXT.md). Qwen3-8B is also registered but has
not been smoke-tested with real weights here. Generated copy requires review.

### Reproduce the image student

Requires the Python environment above and hardware WebGPU in Chrome for playback.
Teacher generation supports CPU, Apple MPS and CUDA; executed paths are identified
in the [execution report](docs/DIFFUSION-EXECUTION.md).

```sh
npm run models -- list
npm run models -- download tiny-sd
npm run dataset -- examples/aurora-generative.ang
npm run train -- examples/aurora-generative.ang 2000
```

On Apple hardware,
prefix dataset/training commands with `ANGELOS_DEVICE=mps`; on NVIDIA use `cuda`.
CPU is the explicit default, not a hidden fallback. Model downloads happen only
through the explicit download command, never by opening a graph or the UI.

Tiny-SD downloads roughly 1.1 GB. FLUX/LTX require substantially more memory and
disk. Model licenses differ; consult [the pinned registry](docs/MODELS.md) before
choosing a model. No weights or generated datasets are committed to GitHub.

### Publish the explanation, not the API

```sh
npm run build:site
npm run build:site -- artifacts/site-subpath /Angelos/
```

Upload the contents of `dist/` (root hosting), or the alternative output with its
matching base path. Output directories must be new. The allowlisted static export
includes the explanation, recorded proof and research documents, not profiles,
model weights, inference endpoints or the live lab. Its `/lab/` is a local setup
guide. Nothing is deployed automatically. See [the readiness audit](docs/SHOWCASE-AUDIT.md).

## What changed

- **Real generative teachers:** registered Diffusers adapters for Tiny-SD,
  FLUX.2 Klein 4B and LTX-Video, with immutable upstream revisions, explicit
  sampling settings and seeds. Adapter implementation is not proof of execution
  on every model/device; see the measured report.
- **Mixed typed graphs:** UPL features feed text prompts and generative nodes;
  image outputs flow through tensor exposure, vignette and brand operations.
  Video is a distinct type, with explicit frame extraction for image students.
- **Diffusion instead of the MLP:** five convolutional U-Net layers, UPL/time/
  spatial conditioning, cosine DDPM noise training and 16-step DDIM sampling.
  Gaussian seed controls variation; the full denoising loop executes in WebGPU.
- **Full-graph learning:** training uses complete rendered teacher outputs, not
  individually distilled nodes. Held-out UPLs are split before teacher generation.
  Datasets and students retain graph, encoder, seed and model provenance.
- **Brand fallback:** training includes the brand. Raw fidelity is measured;
  an exact GPU compositor remains explicit when the learned mark fails.
- **Attribution:** exposure and review identities include the creative seed,
  frozen UPL snapshot and model. Previews are not impressions.

## Build workflow

Examples: `examples/aurora-generative.ang` (small image teacher),
`examples/aurora-text.ang` (Qwen3 → Tiny-SD), `examples/aurora-flux.ang`,
`examples/aurora-ltx.ang`, and `examples/aurora.ang`
(the original deterministic regression fixture).

```sh
node src/cli.js check examples/aurora-generative.ang
node src/cli.js compile examples/aurora-generative.ang
npm run dataset -- examples/aurora-generative.ang
npm run train -- examples/aurora-generative.ang 2000 artifacts/aurora.dataset.npz
```

Pass `--save-previews artifacts/teacher-native` to the dataset command to retain
native-resolution complete-graph PNGs. Existing datasets are not overwritten;
use `--output artifacts/another.dataset.npz` and pass that path to the trainer.

Source and graph are backend-independent representations, not learned weights.
`compile:teacher` exports deterministic-only graphs to ATen without training;
generative pipelines currently execute through registered adapters rather than
claiming fused whole-pipeline CUDA compilation. CUDA optimization is unfinished.

After editing the graph in the UI, regenerate the dataset and student using
`data/campaign.ang`. Old graph hashes are rejected. Rebuilds create new artifacts;
`artifacts/preview.json` selects a local preview, not a production deployment.
Improving held-out **noise-prediction MSE** is a training sanity check, not proof
of perceptual quality, brand compliance or click-rate improvement.

The generative teacher runs offline; the UI never shows a procedural image as a
live FLUX output. Raw/delivered students share a seed. **New variation** changes it
and invalidates pending exposure. Save ingredients, wait for the GPU result, then
expose the creative before recording a click or review.

## Validation

```sh
npm test
npm run test:ml
npm run test:showcase
npm run test:browser
```

The showcase suite needs installed Chrome but no model weights or trained artifact.
The full browser suite needs a trained diffusion preview artifact and installed Chrome.
It launches an isolated browser/test server and compares actual WebGPU inference
against PyTorch fixtures. Offline provider tests use explicit test doubles;
live model runs are reported separately. `npm run train:fixture` retains the old
MLP solely for regression work and changes the local preview pointer.

See [showcase validation](docs/SHOWCASE-EXECUTION.md), [diffusion execution results](docs/DIFFUSION-EXECUTION.md), [diffusion format](docs/DIFFUSION.md),
[graph/data contracts](docs/CONTRACTS.md), and [model registry](docs/MODELS.md).

## Boundaries

Still unproven: campaign-trained high-resolution student quality, learned UPL
encoders, a critic validated on substantial human ratings, measured click-rate
lift and HD browser inference. Reward fitting and bounded candidate optimization
are implemented experiments, not evidence that these quality gates have passed.
Quantization, production identity, real auctions and marketplace integration
remain unfinished.
No recurring task, paid inference, external tracker or ad spending is enabled.

This is a trusted local operator demo using synthetic adult profiles. Consent,
telemetry and training grants remain separate. Revocation/deletion removes eligible
local records but does not promise unlearning from trained weights. Do not expose
this unauthenticated loopback service publicly or use real personal data.

No Rhizome/FLORA implementation, catalog or former-employer assets were copied.
The [original backward plan](docs/PLAN.md) and [research backlog](docs/BACKLOG.md)
remain design history; current executable behavior is described here.
