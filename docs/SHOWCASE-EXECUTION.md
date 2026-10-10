# Showcase execution — 2026-10-08

**GPU Accelerated Ads — a proof of concept by Louis Anderson.**

This report separates the communication upgrade from model-quality evidence.
The project remains a local research prototype, not a deployed ad platform.

## ComfyUI, learning and HD follow-up

The later 2026-10-08 update makes `/lab` ComfyUI-first: author an ad, connect
audience inputs, inspect a result, grade, then understand separate student and
candidate training. Advanced internals remain available rather than leading the
first visit. Desktop/mobile layouts and failure states were inspected in Chrome.

- The standalone plugin has **12 nodes**, including versioned UPL parsing,
  explicit fact/coordinate extraction, FLOAT/INT range mapping, BOOLEAN
  comparison, safe prompt formatting/selection and complete-image export.
- The lab's **Copy UPL JSON** exports only saved, matching synthetic-adult
  snapshots with personalization permission. Unsaved edits and revoked permission
  invalidate the handoff; clipboard failure provides selectable text. Actual
  browser-exported JSON was accepted by the Python plugin parser.
- ComfyUI node/pack tests use supplied tensors and an injected output-directory
  module, not a running ComfyUI installation. The local importer verifies bindings,
  workflow hashes and native PNG bytes; it neither executes graphs nor invents
  reviews. See [setup and limits](COMFYUI.md).
- **124 Node tests passed.** **192 Python tests ran: 191 passed, one MPS-gated
  test skipped inside the sandbox.** That exact MPS test then passed separately
  with local GPU access and no model download. **20 Chrome tests passed**, with one
  inapplicable legacy MLP test skipped. Actual diffusion WebGPU parity remains
  covered by its own passing tests on a non-fallback Apple `metal-3` adapter.
- Genuine local pretrained **1920×1080** evidence is documented separately in
  [HD-EXECUTION.md](HD-EXECUTION.md). It is not the native-32 browser student or a
  campaign-trained HD model. The lab loads it only on explicit request.
- The corrected [native-32 RL smoke experiment](RL.md) changed two candidate
  weight sets, then rejected both. Its reward labels were synthetic. Separate
  [HD adapter-gradient tests](HD-ALIGNMENT.md) validate the actual SCM sampler
  on tiny Diffusers modules, not trained foundation-weight HD quality.
- The portable static export builds with the ComfyUI installation guide, research
  documents and recorded HD image; it contains no live API, profile database or
  model weights.

No paid jobs, real ad traffic, live ComfyUI inference, public deployment, commit
or push were performed. A useful campaign-trained HD student, human-validated
critic, learned UPL encoder and measured click-rate lift remain unproven.

The sections below preserve the earlier showcase-only execution record.

## Delivered

- `/`: responsive explanatory site, original architectural SVG, five-step system
  walkthrough, an explicitly hand-authored illustrative campaign, recorded
  teacher/student comparison, research links and author credit.
- `/lab`: the existing workbench plus separate `prompt` and `generate_text` nodes,
  manual local Qwen probe, source download/restore and graph-aware rebuild commands.
- A source-only startup path and persistent missing/corrupt-artifact states.
- A static-only export with a local setup guide instead of public inference APIs.
- A prioritized [readiness audit](SHOWCASE-AUDIT.md).

## Verified locally

Environment: macOS / Apple Silicon, Node 26.5.0, Python 3.11, installed Chrome;
existing generative dependencies and explicitly downloaded pinned Qwen3-0.6B.

- JavaScript regressions: 117 passing, including the source-only checkout test.
  That test copies source without `.venv`, `node_modules`,
  `data` or `artifacts`, serves both routes and verifies actionable text-worker
  failure without breaking graph access. It does not reinstall dependencies.
- Python regressions: 80 passing, including genuine MPS tensor tests. Text-model
  unit fixtures are explicitly marked and are not pretrained-weight evidence.
- Full Chrome suite: 11 passing, one inapplicable legacy MLP test skipped for the
  diffusion artifact. Tests cover WebGPU parity, brand composition, exposure/click
  attribution, graph changes, consent revocation and the new showcase.
- Showcase checks: responsive widths 320–1920px without horizontal overflow,
  mobile menu keyboard behavior, no-JavaScript explanations, evidence-image
  loading, no homepage API requests, explicit mocked text UI, distinct typed
  nodes, model-aware rebuild instructions and source restore.
- Static export: served under `/Angelos/` in Chrome at mobile width; all eight
  unique internal homepage links, recorded images, persona interaction and local
  setup CTA worked, with no JavaScript errors or API requests. This used a local
  static HTTP server, not a public host. The final showcase-only rerun passed all
  three tests, including keyboard focus on the skip link.
- Actual Qwen3-0.6B CLI and browser HTTP inference, plus actual Qwen → Tiny-SD
  graph execution on CPU; see [inputs, outputs and device failure](TEXT.md).
- Chrome WebGPU used an Apple `metal-3` non-fallback adapter. Maximum GPU/PyTorch
  error was 0.00001395; exact-brand error was 0. The 192px display benchmark had
  p50 13.6ms / p95 19.4ms over 20 warmed samples, including completion/readback,
  excluding display/network cold start. Native learned resolution remains 32px;
  this is not a production latency guarantee or new quality result.

## Recorded visual evidence

`web/evidence/manifest.json` binds the first held-out teacher sample (index 32),
graph identity, model artifact hash, UPL and seeds to portable SVG wrappers of
actual PNG outputs. Teacher/student noise is independent, not reconstruction from
the same seed. The student was captured in hardware WebGPU on 2026-10-08. The
original corpus and trained weights predate this UI work and were not retrained.

The student still fails the useful-conditioning audit. The neural mark also fails;
delivery explicitly composites the exact mark. The attractive vector campaign on
the homepage is labeled an illustration, not this model's output.

## Not claimed

No full replacement training run after graph edits, new preference critic,
CTR-based optimization, useful personalization, high-resolution student,
Qwen3-8B/FLUX/LTX weight execution, CUDA execution, production privacy, public
deployment, commit or push was performed for this showcase update. Static export
is a publishable artifact, not evidence that the private repository or website
has been made public.
