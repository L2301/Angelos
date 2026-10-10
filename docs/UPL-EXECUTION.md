# UPL v2 execution evidence

## Real local encoder, not an injected fixture

On 2026-10-08 the pinned public `Qwen/Qwen3-Embedding-0.6B` checkpoint was
explicitly downloaded into the ignored local model cache. This was a free,
operator-initiated model download, not paid inference or an automatic side
effect of loading the site, importing a graph, or running the encoder.

The actual execution path was:

```text
Synthetic adult facts + one explicit synthetic save event
  → versioned Store ingredients
  → Node UPL worker (offline environment)
  → real pretrained Qwen embedding inference on CPU
  → validated signed 512D public bundle
  → immutable Store snapshot
```

| Recorded property | Result |
| --- | --- |
| Checkpoint revision | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| Execution device | CPU, float32; no GPU or fallback claim |
| Dimensions | 512 |
| Negative coordinates | 267 |
| L2 norm | 0.9999999803845473 |
| Separate recent-intent events | 1 |
| Full cold worker duration | 21.110 seconds, including process/model startup |
| Input ingredients SHA-256 | `7527b01502a5d4c83d540eeb40757616232d8031589d4ea0b84208515eee3bca` |
| Stored snapshot SHA-256 | `36be955f98ee510b33d97e8a98463c8bc962596f873baa22918f15e7d9a98df5` |

The detailed local receipt is `artifacts/upl-smoke.json`; model files and full
local receipts are ignored by Git. These hashes identify this particular run's
timestamped synthetic inputs, not a stable person identity or a transferable
authorization. The screenshot/ComfyUI/reward unit fixtures are separate evidence.

A second, real **Chrome → local API → cached CPU encoder → copied v2 JSON**
round trip also succeeded after editing interests/goals and recording a save
event in the new lab interface. It used no mocked API or embedding response.
The browser reported no page errors; its encode interaction took 13.408 seconds
and produced a 512D vector with norm 1.0000000560361886. Clipboard permission
was deliberately denied to exercise the selectable-JSON fallback. The saved
receipt is `artifacts/upl-browser-real.json`, with a screenshot at
`artifacts/upl-lab-real.png`.

That browser-produced payload was accepted by both the actual ComfyUI UPL
parser and the teacher-pack importer's audience validator. Interests and recent
intent were extracted as typed strings, and vector/encoder/intent values stayed
unchanged. This checks the installable plugin's contract, not a live ComfyUI
workflow or GPU render.

The opt-in Playwright test also passed with **actual clipboard permission**,
real local model inference and no mocked routes. Its separate round trip took
21.931 seconds and produced 512 coordinates (269 negative; norm
0.999999963261989). Reproduce it after provisioning the cache:

```sh
ANGELOS_TEST_REAL_UPL=1 ANGELOS_UPL_DEVICE=cpu npm run test:browser -- \
  test/browser/upl.spec.js --grep 'real cached CPU encoder'
```

**This is not ad-placement latency.** Cold loading the encoder per request is
deliberately straightforward and does not meet an ad-serving latency budget.
The architecture keeps user-state encoding outside creative generation; cached
representations, persistent workers, batching and freshness evaluation still
need engineering before any real deployment.

## Reproducible dimension smoke study

The bounded fixture in `examples/upl-aurora-evaluation.json` contains twelve
fictional adults with manually declared interest groups. The four legacy facts
are held identical, while interests, goals and visual preferences vary.

```sh
PYTHONPATH=ml .venv/bin/python -m angelos_ml.upl_evaluate \
  --device cpu --output artifacts/upl-dimension-study-new.json
```

The command loads the pinned cached encoder once, encodes each profile at 1024D,
and evaluates normalized Matryoshka prefixes at 128/256/512/1024D. It reports
norms, payload sizes, timings, nearest neighbors and explicit ties. Existing
output files are never overwritten, and no download occurs. Model input excludes
the synthetic person identifier and the expected group label.

The real CPU run completed with twelve encodes and one model load. All four
widths produced finite usable vectors, and each profile's nearest neighbor
belonged to its manually declared group (12/12 at every width, with no ties).
This fixture therefore **does not distinguish which semantic width is best**.
The intentionally identical legacy vectors tied across all eleven other people.

| Width | FP32 vector bytes/person | Compact vector JSON bytes, all 12 profiles |
| --- | --- | --- |
| 128 | 512 | 31,855 |
| 256 | 1,024 | 64,182 |
| 512 | 2,048 | 129,293 |
| 1024 | 4,096 | 260,262 |

These exclude facts, context and provenance. The receipt records 13.883 seconds
for model loading and 16.081 seconds for twelve sequential encodes. These are
single-run timings, not a controlled throughput benchmark. The complete receipt
is `artifacts/upl-dimensions-aurora.json`; its source corpus SHA-256 is
`4e28ff80cdb2e01223ab8c0a0373c7ca936427ea28879e42d05f082508859d5c`.

The study is an inspectable **semantic smoke test**, not held-out advertising
evaluation. The four-coordinate baseline cannot distinguish these profiles by
construction; that demonstrates omitted ingredients, not superior CTR. Even
perfect separation of this small hand-authored corpus cannot select an optimal
dimension or establish a learned preference model. The 512D default remains a
starting hypothesis, pending the person/time/campaign holdouts and ablations in
[the research review](RESEARCH-UPL.md).

## What this does not establish

- No actual customer, cookie matching service, publisher integration or tracker.
- No learned temporal-interest model: recent events remain a separate bundle branch.
- No trained HD UPL adapter or evidence of improved image quality or click-through rate.
- No compatibility between a 512D embedding and existing legacy 4D browser weights.
- No paid jobs, automated nightly training, or automatic model promotion.

## Regression validation

- Node: 145 tests passed, including legacy graph hash parity, semantic graph
  contracts, consent, immutable snapshots, worker boundaries and HTTP repairs.
- Python: 264 tests collected; 263 passed in the sandbox and the single MPS
  availability-dependent test passed separately with hardware access.
- Chrome: 33 passed in the regular suite, with the obsolete MLP and opt-in real
  encoder cases skipped. The real encoder case then passed separately with
  `ANGELOS_TEST_REAL_UPL=1`; the suite also exercised actual legacy WebGPU DDIM.
- Static export: 37 allowlisted files built under `/Angelos/`; no local profiles,
  model weights or laboratory API were packaged.

The current portable export is `artifacts/site-upl-v2-final/`. It has not been deployed
to a remote host. The local preview was restarted and the lab, new ingredients
endpoint and execution document all returned HTTP 200.

See [UPL contracts and operation](UPL.md) and [the ComfyUI handoff](COMFYUI.md).
