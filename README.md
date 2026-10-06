# Angelos

**Author a creative graph. Execute a teacher. Distill one diffusion model per campaign.**

Angelos is a local research laboratory for UPL-conditioned creative generation.
Its typed graph combines deterministic tensor image operations with stochastic,
locally executed open-weight image/video models. The browser runs a separate,
campaign-specific conditional diffusion student in WebGPU.

This is not a finished ad-generation platform. The current student is a small,
freshly trained **32×32 image-space U-Net**, not a pretrained photorealistic model.
The enlarged preview is not genuine high-resolution output. Model capacity,
training data, perceptual quality and conditioning need further research.

The [full creative-model capability audit](docs/MODEL-COVERAGE.md) covers the
broader intended graph: image editing/control, video, audio, avatars, language,
documents, vector/3D assets and deterministic operations. Its sourced candidates
are research recommendations, not a claim that these adapters are implemented.

## Quick start

Requires Node 24+, Python 3.11 and hardware WebGPU in Chrome. Teacher generation
supports CPU, Apple MPS and CUDA; tested devices are listed in the execution report.

```sh
npm ci
python3.11 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements-generative.txt
npm run models -- list
npm run models -- download tiny-sd
npm run dataset -- examples/aurora-generative.ang
npm run train -- examples/aurora-generative.ang 2000
npm start
```

Open `http://127.0.0.1:4310`. The server binds only to loopback. On Apple hardware,
prefix dataset/training commands with `ANGELOS_DEVICE=mps`; on NVIDIA use `cuda`.
CPU is the explicit default, not a hidden fallback. Model downloads happen only
through the explicit download command, never by opening a graph or the UI.

Tiny-SD downloads roughly 1.1 GB. FLUX/LTX require substantially more memory and
disk. Model licenses differ; consult [the pinned registry](docs/MODELS.md) before
choosing a model. No weights or generated datasets are committed to GitHub.

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
`examples/aurora-flux.ang`, `examples/aurora-ltx.ang`, and `examples/aurora.ang`
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
npm run test:browser
```

The browser suite needs a trained diffusion preview artifact and installed Chrome.
It launches an isolated browser/test server and compares actual WebGPU inference
against PyTorch fixtures. Offline provider tests use explicit test doubles;
live model runs are reported separately. `npm run train:fixture` retains the old
MLP solely for regression work and changes the local preview pointer.

See [execution results](docs/DIFFUSION-EXECUTION.md), [diffusion format](docs/DIFFUSION.md),
[graph/data contracts](docs/CONTRACTS.md), and [model registry](docs/MODELS.md).

## Boundaries

Still unfinished: useful high-resolution student quality, pretrained student
initialization, learned UPL encoders, preference-critic training, nightly CTR-driven
updates, quantization, production identity, real auctions and marketplace integration.
No recurring task, paid inference, external tracker or ad spending is enabled.

This is a trusted local operator demo using synthetic adult profiles. Consent,
telemetry and training grants remain separate. Revocation/deletion removes eligible
local records but does not promise unlearning from trained weights. Do not expose
this unauthenticated loopback service publicly or use real personal data.

No Rhizome/FLORA implementation, catalog or former-employer assets were copied.
The [original backward plan](docs/PLAN.md) and [research backlog](docs/BACKLOG.md)
remain design history; current executable behavior is described here.
