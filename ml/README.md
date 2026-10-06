# Angelos ML

The default trainer is now `angelos_ml.train_diffusion`, using a conditional
convolutional DDPM denoiser and DDIM sampling. Build a complete open-weight teacher
dataset first, then run `npm run train`. See the root README,
[diffusion contract](../docs/DIFFUSION.md), and [model registry](../docs/MODELS.md).
The open-weight pipeline dependencies are in `requirements-generative.txt`.

The remainder describes the preserved **legacy MLP regression fixture** and
deterministic-only ATen compiler, not the default campaign model.

## Legacy fixture

This package distills a deterministic tensor teacher into freshly initialized
6 → 64 → 64 → 3 ReLU/ReLU/sigmoid weights. It does not download pretrained models,
access external services, learn a preference critic, claim CTR uplift, or promote
artifacts. Dependencies are pinned to the versions observed in the parent-managed
environment. Dependency installation belongs to the parent project.

## Run from the repository root

Set `PYTHONPATH=ml` and use `.venv/bin/python`. The compiler must supply the
authoritative graph hash:

```sh
PYTHONPATH=ml .venv/bin/python -m angelos_ml.train \
  --graph examples/aurora.graph.json \
  --graph-hash "$GRAPH_HASH" \
  --output ml/artifacts/aurora.student.json \
  --steps 1200 --device cpu
```

The CLI requires `--graph-hash` and compares it with Python's canonical graph hash
before initializing training. Objects are recursively key-sorted, nodes sorted by
ASCII ID, integers serialized without `.0`, negative zero normalized, and ordinary
JavaScript exponent formatting reproduced. Python and JavaScript shortest-decimal
formatting can differ at rare floating-point tie cases. For those, pass
`--graph-canonical PATH` containing the compiler's exact canonical JSON; Python
checks its graph semantics and SHA-256 instead of guessing the serialization.
Canonical sidecars may have surrounding whitespace, which is stripped before hashing.
No subprocess or executable graph payload is used by the trainer.

Default training uses 128 fixed synthetic adult profiles and 32 independently
held-out profiles. Age band is an index divided by five; region is an index divided
by three; declared openness and energy are normalized synthetic coordinates.
The independently versioned encoder is not trained. Sampled training coordinates
mix 75% uniform image coverage and 25% uniform plate coverage when a brand exists.
MSE on that mixture is deliberately brand-weighted; reported raster MSE is not.
Targets always execute the entire graph, including its brand node. No glyph or
image is pasted into the student during training or raw evaluation.

Training defaults to two CPU threads and a 180-second deadline checked between
steps. Baseline evaluation counts against that deadline; final evaluation/export
add work after it. `--max-seconds` cannot exceed 600. A deadline-stopped model
records actual steps and stop reason. A run with zero completed updates cannot
export. `--train-profiles`, `--heldout-profiles`, `--batch-size`, `--eval-width`,
and `--eval-height` support smaller tests. Profiles never move between splits;
there is no held-out checkpoint selection or gradient computation.

## Output contract

For output `NAME.student.json`, the CLI writes:

- `NAME.student.json`: frozen `angelos.student.v1` artifact with finite row-major
  `[output,input]` flattened weights, raw metrics, encoder binding, and provenance.
- `NAME.student.metrics.json`: untrained and trained raster MSE, per-profile
  errors, raw/composite comparisons, profile split values/hashes, training trace,
  source identity, artifact checksum, and explicit limitations.
- `NAME.student.parity.json`: two 16×12 held-out raster cases with float32 RGBA
  and byte references, plus RGB point probes including all brand-cell centers.
  Raw student, display student, and teacher outputs are distinct. Tolerances are
  `0.001` absolute per float channel and one byte after `floor(clamp(v)*255+0.5)`.

Coordinates are unmodified `[0,1]` pixel-center coordinates, top-left origin.
Colors follow the frozen contract's display-encoded RGB, not linear-light RGB.
Evaluation uses a 128×96 full raster per held-out profile and additionally probes
all 63 plate-cell centers, so a tiny logo cannot disappear between raster samples.
`brandMaxError` measures maximum raw-neural channel error across those observations.
If it exceeds `0.06`, the manifest says `composite`, replacing the entire opaque
plate at display time. A separately measured composite error never substitutes
for raw `heldoutMse`. Without a brand, brand metadata and detailed measurements are
null; the runtime-required numeric `metrics.brandMaxError` is zero, with
`metrics.hasBrand=false` explicitly marking that it is not a fidelity measurement.
The evaluation is finite-domain evidence, not a universal fidelity guarantee.

Build IDs are unique. Existing artifacts or sidecars are never overwritten. An
artifact with worse held-out MSE can still be exported as a documented experiment,
but `improvedOverUntrained` is false and `training.promoted` remains false.

## Compile the teacher without training

```sh
PYTHONPATH=ml .venv/bin/python -m angelos_ml.compile \
  --graph examples/aurora.graph.json \
  --graph-hash "$GRAPH_HASH" \
  --graph-canonical "$GRAPH_CANONICAL_PATH" \
  --output ml/artifacts/aurora.teacher.pt2 --samples 256
```

`--graph-canonical` is optional and has the same verified-source semantics as the
training CLI. This separate stage validates the IR and graph hash, wraps
`TensorTeacher.color_at(..., validate=False)` in an `nn.Module`, and calls
`torch.export.export(..., strict=True)`. The teacher's Python node traversal is
captured as an ATen tensor graph with lifted tensor constants, including the glyph
row bits. No student is initialized or trained. An audit rejects non-ATen calls,
non-tensor constants, and trainable parameters; export errors fail the command.
There is no silent interpreter fallback.

`--samples` is the fixed input batch size, not the number of training examples.
Supported values are 1–4096. The artifact accepts CPU float32 tensors `upl` with
shape `[samples,4]` and `coordinates` with shape `[samples,2]`, returning
`[samples,3]` display-encoded RGB. Callers must validate finite normalized values;
the exported forward deliberately does not execute Python input validation.
Other batch sizes require a new export. Padding/chunking is a caller concern.

Before writing output, the compiler serializes and reloads the program, re-audits
its graph, verifies embedded source metadata, and compares the loaded module to
the eager teacher on fresh synthetic profile batches, corners, center, and all
63 brand-plate cell centers. The absolute CPU parity gate is `1e-6`, including
brand pixels. Failure writes no artifact. It emits:

- `NAME.teacher.pt2`: `torch.export.save` archive, including lifted constants and
  embedded `angelos.metadata.json` provenance/input contracts.
- `NAME.teacher.compile.metrics.json`: artifact SHA-256, ATen operator inventory,
  constant shapes/dtypes, export and loaded-parity timing, exact measured maximum
  and mean CPU errors, and brand-region error when applicable.

Existing outputs are never overwritten. `elapsedSeconds` includes validation,
capture, in-memory serialization/reload, audit, and parity; final filesystem writes
and Python process startup are excluded. The compilation parity brand error is
not the neural student's brand reconstruction error: compilation aims to preserve
the procedural teacher, whereas distillation approximates it.

Load only locally produced or otherwise trusted `.pt2` archives using a compatible
PyTorch toolchain. This is not an untrusted-model loader. The original source
module uses coordinate-device-aware tensor construction, leaving a future
CUDA-target export possible; neither CUDA execution nor moving this CPU-captured
archive to CUDA has been validated. ATen graph lowering is real compilation to an
intermediate representation, not native machine-code compilation or GPU evidence.

## Tests

```sh
PYTHONPATH=ml .venv/bin/python -m unittest discover -s ml/tests -v
```

Tests cover strict graph rejection, graph hashing against the actual Node compiler
when available, all operation formulas, source changes, glyph cells and half-open
bounds, differentiability, disjoint profile sampling, genuine short CPU training,
brand-fallback measurement, NumPy/JSON forward parity, and non-overwriting exports.
Compilation tests additionally reload fixed-batch `.pt2` files, disable the Python
teacher to prove no runtime interpreter dependence, verify source edits/constants,
and exercise strict failure, compiler sidecars, and batch-shape constraints.
