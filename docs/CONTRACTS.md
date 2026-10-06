# Graph and data contracts

These are deliberately bounded engineering contracts, not the full multimodal product.

The coordinate MLP contract below is retained only for regression fixtures. The
default campaign student follows [the diffusion contract](DIFFUSION.md).
[Model contracts](MODELS.md) extend the graph with typed text, stochastic open-weight
generation, explicit execution seeds, and local-only model loading. A generative
graph never silently falls back to the procedural interpreter or claims whole-pipeline
ATen export. Creative IDs include artifact, exposure-time snapshot and uint32 seed.
Raw and composited previews share that seed; changing it invalidates pending exposure.

## Creative graph v1

```json
{
  "version": "angelos.graph.v1",
  "id": "aurora",
  "encoder": {"id": "structured-demo-v1", "dimension": 4},
  "nodes": [
    {"id": "person", "op": "upl", "inputs": {}, "params": {}},
    {"id": "mood", "op": "feature", "inputs": {"input": "person"}, "params": {"index": 3}},
    {"id": "field", "op": "gradient", "inputs": {"condition": "mood"}, "params": {"low": [0.03, 0.08, 0.09], "high": [0.22, 0.72, 0.55], "angle": 0.4}}
  ],
  "output": "field"
}
```

Port types: `upl` (four finite normalized coordinates), `scalar`, `image` (RGB f32,
display-encoded RGB in [0,1], no alpha). Image functions are evaluated at pixel centers:
`x=(column+0.5)/width`, `y=(row+0.5)/height`, origin top-left. This deliberately uses
display-encoded values in the fixture; physically linear compositing is future work.

All nodes must be reachable from output. Unknown fields, ports, parameters and operations
are errors. IDs match `[A-Za-z_][A-Za-z0-9_]*`, count <=64. References may be forward
declared; execution is topological, with lexicographic tie breaking. Cycles are errors.
Exactly one `upl` source for this slice. No remote URLs, executable code or dynamic loads.
The hash is SHA-256 of recursively key-sorted JSON with nodes sorted by ID, excluding
UI positions; no whitespace. Hashing the validated graph binds the student to the source.

| Operation | Required inputs | Required params | Output and semantics |
| --- | --- | --- | --- |
| upl | none | none | UPL input vector |
| feature | input: upl | index: integer 0..3 | scalar |
| constant | none | value: finite 0..1 | scalar |
| gradient | condition: scalar | low/high: RGB, angle: -1..1 | mix low/high by clamp(0.65*x + angle*(y-0.5) + 0.35*condition,0,1) |
| disk | image: image, condition: scalar | color: RGB, center: xy, radius: .02..1, softness: .001..1 | centerX=center[0]+.12*(condition-.5); alpha=clamp((radius-distance)/softness+.5,0,1); mix image/color |
| exposure | image: image, condition: scalar | amount: -1..1 | clamp(image * (1 + amount*(2*condition-1)),0,1) |
| vignette | image: image | strength: 0..1 | image*(1-strength*clamp(2*((x-.5)^2+(y-.5)^2),0,1)) |
| brand | image: image | x,y:0..1, cell:.005...05, color/background: RGB | fixed 5x7 A glyph plus one-cell opaque plate padding |

At most one brand node, and it must be the output. Plate width=7*cell, height=9*cell;
it must fit inside [0,1]^2. Glyph row bits are [14,17,17,31,17,17,17], most-significant
of five bits at the left. Plate index is floor((coordinate-origin)/cell); glyph occupies
columns 1..5 and rows 1..7. Outside plate preserve image. Half-open bounds.

Source form is named calls with explicit typed ports, e.g.:

```text
campaign aurora
person = upl()
mood = feature(input: person, index: 3)
field = gradient(condition: mood, low: [0.03, 0.08, 0.09], high: [0.22, 0.72, 0.55], angle: 0.4)
output field
```

Graph JSON and source round-trip semantically. This first syntax omits pipe sugar,
imports, loops, arbitrary models and other modalities. Rejection is preferable to a
pretend implementation. The browser graph view edits the same validated representation.

## Student v1

```text
format: "angelos.student.v1"
id: unique build ID
campaignId: graph.id
graphHash: validated graph SHA-256
encoder: {id: "structured-demo-v1", dimension: 4}
inputOrder: ["x", "y", "upl.0", "upl.1", "upl.2", "upl.3"]
layers: [{input: 6, output: 64, activation: "relu", weights: [...], bias: [...]},
         {input: 64, output: 64, activation: "relu", weights: [...], bias: [...]},
         {input: 64, output: 3, activation: "sigmoid", weights: [...], bias: [...]}]
brand: null OR {mode: "neural"|"composite", x, y, cell, color, background}
metrics: {baselineMse, heldoutMse, brandMaxError, brandTolerance, ...}
training: {seed, steps, device, ...}
```

Row-major weights, PyTorch Linear convention `[output,input]`. Hidden widths may be
1..64; exactly three layers. All parameters finite, bounded by absolute 1000. No arbitrary
shaders, pickle, or executable model payloads. Render input UPL must be length four with
finite values in [0,1]. Width/height integers 1..512. Coordinates are pixel centers.

Train against the full graph INCLUDING brand. Compare pure-neural brand plate output
with the teacher over held-out profiles, separately from overall MSE. If the max channel
error exceeds 0.06, artifact brand mode is `composite`. Raw student output remains
available for evaluation. A composite artifact replaces the entire brand plate, not
only glyph pixels, to hide malformed neural lettering within that plate. No claim of
bitwise cross-device floating-point identity; byte output and float tolerances are tested
separately. Required-copy and real-logo tests are later work.

Runtime API in `src/browser/student.js`:

- `validateStudent(artifact)` returns artifact or throws.
- `renderCpu(artifact, upl, width, height, {composite=true}={})` returns Float32Array RGBA.
- `createGpuRenderer(artifact)` resolves a renderer or throws (never CPU fallback).
- `renderer.render(upl,width,height,{composite=true}={})` resolves `{pixels, milliseconds}`;
  pixels is Float32Array RGBA, milliseconds includes GPU completion/readback.
- `renderer.dispose()` destroys owned buffers/device.

The initial backend reads output back for canvas display and measurement. This is an
explicit performance limitation, not a zero-copy claim. A future compositor can consume
GPU textures directly. Scheduling belongs to the UI: accept only the latest input
revision, bound in-flight renders, and never count cancelled/stale output as exposure.

## Reference implementation boundaries

- `src/core/graph.js`: parse/print/check/order/hash graph, scalar pixel reference.
- `src/core/store.js`: SQLite UPL/event/consent/snapshot/impression/review contracts.
- `src/browser/student.js`: learned-model CPU/GPU execution and brand fallback.
- `ml/angelos_ml`: PyTorch teacher, distillation, artifact export and numerical fixtures.
- `src/server.js`: loopback static/API boundary, validated graph persistence, telemetry.
- `web/`: laboratory UI; native browser ESM, no framework/build required.

Synthetic UPL source values use age bands `18-24`, `25-34`, `35-44`, `45-54`, `55-64`,
`65+`; region codes `NA`, `EU`, `APAC`, `OTHER`; and declared `openness`/`energy` in
[0,1]. Encoder maps age index/5, region index/3, openness, energy. Region ordinality is
an intentionally crude fixture, not a similarity claim. All four fields required for
encoding; missing values remain missing in facts. No inferred personality or minors.
