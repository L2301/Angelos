# First execution report

Historical MLP milestone: this records the original procedural fixture, not the
current diffusion student. See [the diffusion execution report](DIFFUSION-EXECUTION.md)
for current open-weight teacher and WebGPU diffusion validation.

Date: 2026-10-06. Local Apple Silicon Mac, Node 26.5.0, Python 3.11.15,
PyTorch 2.14.1, NumPy 2.4.6, installed Google Chrome.

## Work performed

- Reviewed the supplied Rhizome language as a conceptual reference before this build;
  no implementation, catalog, or employer asset was copied into Angelos.
- Created the private GitHub repository `L2301/Angelos` and a local repository at
  `/Users/main/Angelos`. Code remains local/uncommitted; no push was performed.
- Wrote a backward dependency plan and shared contracts before implementing the app.
  Three parallel agents reviewed ML, browser/GPU execution, and personal-data architecture,
  then implemented disjoint slices. The parent built the compiler, UI, server and integration.
- Tested real learned weights and actual hardware WebGPU, not a mocked image-generation API.

## Separate compilation result

Command: `npm run compile:teacher -- examples/aurora.ang`.

The complete typed graph exported to a fixed-batch PyTorch ATen program with 256 coordinate
samples. Reloaded exported-program versus eager teacher maximum error was **0.0**, including
brand probes. Capture/export plus validation took approximately **1.51 seconds** in this run.
The `.pt2` program and compile report are in ignored `artifacts/`.

This establishes CPU export parity for this supported graph. It is **not** evidence of
CUDA kernel generation, CUDA hardware performance, arbitrary graph support or photorealism.

## First distillation experiment

Command: `npm run train -- examples/aurora.ang 1600`.

| Item | Measured value |
| --- | --- |
| Teacher | Complete procedural Aurora graph, including the brand plate |
| Student | 6→64→64→3, ReLU/ReLU/sigmoid |
| Parameter count | 4,803 |
| Raw FP32 parameter bytes | 19,212 (18.76 KiB) |
| JSON package bytes | 100,225 |
| Training | 1,600 Adam updates, seed 2026, CPU, two threads |
| Profiles | 128 training, 32 disjoint held-out synthetic vectors |
| Trainer elapsed | 3.27 seconds, excludes interpreter/import startup |
| Untrained held-out MSE | 0.0829955309 |
| Trained raw held-out MSE | 0.00537907565 |
| Relative MSE reduction | 93.52% against untrained baseline |
| Delivered/composited held-out MSE | 0.00249110895 |
| Raw student mean absolute error | 0.0381422306 |
| Raw brand maximum channel error | 0.794326067 |
| Declared brand tolerance | 0.06 |
| Composited brand maximum error | 0.0 on recorded evaluation |

**The learned-only mark failed.** The student genuinely attempted it: the brand operation
was included in training targets, with extra plate samples. It did not meet the threshold,
so the package selected an explicit exact GPU plate/glyph compositor. The UI preserves
the raw student preview instead of presenting corrected output as neural success.

This is a narrow infrastructure experiment. The reconstruction improvement is not CTR
uplift, a preference-model result, or proof that the student is superior to the cheap exact
procedural renderer. Its held-out MAE does not meet the proposed 0.02 research target.
The preview build gate requires improvement over an untrained baseline, not product readiness.

## Real browser evidence

Chrome reported vendor **apple**, architecture **metal-3**, and a **non-fallback adapter**.
The browser executed the student and brand composition in WGSL. Output was read back for
the canvas; this first implementation does not claim a zero-copy display path.
CPU setup classifies the small row/column lookup tables for brand-plate cells; the GPU
performs glyph selection and pixel replacement. The reported render timing excludes this
mask-table preparation, and the runtime currently allocates buffers per render.

Two held-out 16×12 raster cases, raw and composited, compared with PyTorch:

| Comparison | Maximum absolute channel error |
| --- | --- |
| Actual WebGPU student vs PyTorch | 2.384185791015625e-7 |
| JavaScript f32 CPU student vs PyTorch | 2.682209014892578e-7 |
| JavaScript teacher vs PyTorch teacher | 2.386715797775274e-7 |
| Composited brand pixels | 0.0 |

A separate 192×192 run used three warmups and 20 measured renders; final-suite results:

- Warm p50: **3.00 ms**.
- Warm p95: **6.80 ms**.
- Renderer creation in this already-running browser: **2.70 ms**.

An earlier repetition measured 5.30 ms p50 / 5.90 ms p95. These small-sample variations
are why the suite checks correctness, not a brittle performance threshold.

These include inference completion and GPU readback, but exclude display/presentation,
network cold start, empty driver caches and model transfer. They are one desktop run, not
a mobile, ad-network, tail-latency or production SLA claim. Do not generalize these numbers
to a larger generative model. No total physical GPU-memory measurement was made.

## Validation

- JavaScript suite: **61 tests passed**, including graph/type/hash round trips, malformed
  inputs, consent, immutable snapshots, idempotency, attribution windows, deletion,
  HTTP Host/Origin/static-file restrictions, neural math, and mocked device lifecycle.
- Python suite: **32 tests passed**, including teacher math/gradients, disjoint profile
  splits, real short training, artifact/provenance bounds and exported-program parity.
- Real Chrome suite: **6 tests passed**, including numerical parity, responsive UI,
  saved-UPL exposure/click/review, hidden-exposure rejection, measured inference, and
  revoked-personalization / failed-profile-switch regression tests.

Initial HTTP tests hit sandbox socket restrictions and were rerun with approved loopback
access. A test-client Host-header override issue and no-brand metric contract mismatch were
found and corrected. This report describes the passing runs, not the initial failures as
successful verification.

Independent browser review also identified two display/consent issues: inference could
continue after personalization revocation, and a failed profile switch could retain the
previous person's canvas. Rendering now requires a current authorized snapshot, and
profile changes clear old imagery immediately. Both fixes have real-browser regressions.

Screenshots and failure traces, when any, live under ignored `test-results/`.

## Remaining limits

- Procedural still-image teacher only; no pretrained generator or real ad campaign assets.
- No live semantic text embedding, cross-advertiser identity service or data collection.
- Review storage is real; human-critic training is not implemented.
- Mature attribution-correct examples are exportable; nightly CTR weight updates and
  automatic model promotion are not implemented or scheduled.
- The single-candidate placement sandbox logs probability one; it is not randomized
  exploration or a causal experiment. A preview is not counted as an exposure.
- This is a trusted local synthetic-data service. Browser evidence is not verified attention,
  and loopback/Origin checks are not production authentication or advertiser isolation.
- No external GPU, CUDA validation, paid resource, public deployment, commit or push.

See `BACKLOG.md` for ordered next experiments and their acceptance gates.
