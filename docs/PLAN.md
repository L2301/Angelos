# Angelos: working backward from a served creative

Status: initial design, before application implementation. 2026-10-06.

## 1. Desired outcome and constraints

A user owns a changing collection of facts and a separately derived personal latent (UPL).
A campaign author builds a typed multimodal creative graph. That graph defines a teacher,
not the final serving workload. Human judgments shape a campaign critic. Whole-graph
distillation produces one campaign-specific creative model. The browser runs the student
with WebGPU. Nightly learning uses actual exposed impressions and clicks, under brand
and quality constraints, rather than treating every generated candidate as an impression.

Compilation and learning are different operations. Compilation preserves the specified
computation; distillation deliberately approximates it. Neither an optimized graph nor a
one-step diffusion model is necessarily small enough to download and run in a browser.

The first implementation must prove real numerical computation, actual learned weights,
and browser inference before expanding model catalogs or building an elaborate marketplace.
No paid GPU resources or pretrained-weight downloads are implicitly authorized.

## 2. Backward dependency chain

### A. A displayed creative must be accountable

To count a click, identify the visible impression, campaign, deployed artifact, immutable
UPL snapshot/encoder version, creative identity, placement, exposure time, selection
probability, and subsequent click time. Speculation, a bid, a won auction, and a displayed
ad are four different events. A losing bid never becomes a negative example.

Every model update produces a candidate, not an automatic replacement. Holdouts,
minimum data thresholds, brand checks, latency/memory measurements, and a rollback target
stand between a candidate and promotion. CTR is the optimization objective; brand and
consent are constraints, not coefficients that CTR can overpower.

### B. A browser deployment must be bounded

The model package needs an architecture and tensor contract, weights, graph hash, encoder
identity, asset hashes, artifact version, evaluation report, and exact-operations policy.
Measure download, shader compilation, model initialization, GPU completion, compositing,
and display separately. Report cold and warm latency, not only command-submission time.

Device capability detection precedes allocation. A CPU diagnostic renderer is useful but
must never be reported as successful WebGPU execution. CUDA is a separate backend; CUDA
code cannot simply be shipped to a browser. Unsupported browser operators fail export.

Neural-only brand fidelity is attempted first. A validation gate decides whether fixed
GPU composition of logos/required text is needed. Passing on sampled cases is evidence,
not a guarantee of exact output over every possible UPL. For strict brand invariants the
fixed path remains the dependable option. Preserve both raw and delivered outputs so
compositing cannot conceal weak student fidelity.

### C. A useful student requires a bounded learning problem

The student learns the campaign teacher's conditional output distribution, not a general
purpose replacement for every model in the teacher. Inputs include a versioned UPL,
placement specification and, when needed, explicit noise. Campaign constants may be
specialized. New products, graph edits, or an incompatible encoder require a new build.

Start with a tiny coordinate-conditioned RGB network distilled from an executable tensor
image graph. This is an engineering fixture, not a photorealistic advertising model.
Its value is that we can test training, teacher/student error, browser export, brand
fallback, profile changes, and version rejection inexpensively on this machine.

The next research milestone substitutes an appropriately licensed open-weight image
teacher and pretrained student. A successful procedural fixture does not establish that
100–1,000 preference labels can train a general image generator or that arbitrary graphs
will compress well. Those labels are intended for preference adaptation; teacher samples
and pretrained representations do the heavy lifting.

### D. A teacher must have executable semantics

Creative graph nodes operate on typed values: UPL, selected factual attributes, scalar,
text, tensors/images, and eventually video/audio. The source and visual editor address
the same graph; positions and UI selection are not computation. Wire names, dependencies,
parameter constraints, shapes, dtypes, color spaces, and stochastic inputs are explicit.

Compilation pipeline:

1. Parse source or validate graph JSON; do not evaluate user code.
2. Resolve explicit references and validate parameter schemas.
3. Type-check every input/output and reject cycles and unreachable mistakes.
4. Canonicalize the computational graph; separate layout metadata.
5. Classify supported backend operations and assets.
6. Lower to tensor execution with a testable reference path.
7. Optionally capture/export/optimize supported regions; measure rather than promise fusion.

The first teacher uses tensor math for gradients, masks, compositing, and exposure, with
no image round trips through a CPU library in the rendering path. Kornia is a candidate
for differentiable geometry/filter/color operations; CV-CUDA is a candidate for measured
server-side image bottlenecks. GPU execution does not imply differentiability. Discrete
text generation and codecs may remain boundaries; distillation can use teacher samples
without backpropagating through every teacher operation.

### E. A UPL must remain replaceable and revocable

Separate four things: identity/consent, ingredient events, a replayable fact snapshot, and
an encoder-derived latent. An embedding is not a source of truth or anonymization.
Consumers must not interpret coordinate 17 as a named personality trait unless the
encoder contract explicitly says so. Named attribute selection reads validated facts;
latent conditioning consumes the vector and encoder identity.

The initial encoder is an explicit structured baseline for synthetic demographics and
declared dimensions, not a claimed semantic embedding. Later encoders may embed canonical
JSON or learn similarities. Record missingness, confidence, provenance, validity time,
and encoder parameters rather than silently replacing unknown facts with known zeros.

Ingestion is idempotent. Late events cannot overwrite newer facts accidentally. Data
collection, personalization, advertiser sharing, and training need separate scope checks.
Revocation invalidates serving and training eligibility. Deletion includes derived
latents and dependent local logs. Deleting a row cannot undo influence on already trained
weights; production needs a retraining/unlearning and artifact-retirement policy.

## 3. Build order and acceptance gates

| Stage | Deliverable | Gate before moving on |
| --- | --- | --- |
| S0 | This plan, threat model, contracts, explicit unknowns | Independent reviews and coherent cross-layer contracts |
| S1 | Typed IR + source parser/printer + sample campaign | Round-trip semantics; invalid references/types/cycles/parameters rejected |
| S2 | Synthetic UPL store + versioned encoder + outcome ledger | Consent, replay, deletion, deduplication, snapshot attribution tests |
| S3 | Tensor teacher + trainable compact student | Held-out loss improves; source edits change teacher; model package reproducible |
| S4 | Browser WebGPU inference + campaign laboratory | GPU/CPU parity, no hidden CPU fallback, graph/artifact mismatch rejected |
| S5 | Review capture + critic/nightly candidate interfaces | Real labels distinguished from fixtures; no automatic promotion |
| S6 | Open-weight teacher and browser-sized generative student | Rights/compatibility audit; held-out aesthetic/brand/latency benchmarks |
| S7 | Preference adaptation and nightly CTR updates | Exposure correctness; exploration support; confidence and rollback gates |
| S8 | Durable multi-user marketplace + real integration | Authentication, tenant isolation, abuse review, operational load and consent audit |

S1–S4 form the first engineering slice. Implemented status and actual test results belong
in README and an execution report; this table is a roadmap, not a list of completed work.
Still images are the initial fixture. The IR must be extensible, but video/audio support
is not earned by adding an unused enum value.

## 4. Alternatives rejected for the first slice

- Entire production marketplace first: hides model feasibility under application scaffolding.
- A mocked image API standing in for distillation: cannot validate the central claim.
- Arbitrary Python/JavaScript graph nodes: prevents bounded compilation and creates code-execution risk.
- Forking the former employer's implementation: unnecessary and not authorized by conceptual reuse.
- Hand-maintained CUDA as the canonical representation: excludes browser and portable testing.
- Assuming few-step diffusion means small weights: compute-step count and model size differ.
- Rewarding only clicks without brand constraints: permits clickbait and critic exploitation.
- Training on non-clicked speculative candidates: introduces false negatives.
- Treating nearby UPL vectors as proof of useful clustering: requires task-specific evaluation.

## 5. Threats and operational limits

- Profiles, raw vectors, and creative personalization can expose private facts. The initial
  server binds only to loopback and uses synthetic records; it is not a production auth system.
- User-controlled facts/prompts can contain injection text. Facts are data, never executable
  source or privileged instructions. Graph parsers and asset registries use allowlists.
- Malicious graphs and weights can exhaust GPU memory. Bound node counts, tensor sizes,
  network shapes, weight lengths, and finite values before allocation.
- Shared GPU devices may be lost. Cancel stale work and fail visibly rather than counting
  an impression or displaying an older person's creative.
- Browser model weights are inspectable. Never ship secrets, raw training profiles, or
  private advertiser credentials in an artifact.
- Click events can be forged. The demo illustrates attribution, not fraud resistance.
- Exposure must pin the previous UPL before a click updates its future state, preventing
  label leakage into the context used to predict that same click.
- A cold browser cannot begin local inference before receiving/preloading model bytes.
  Speculation reduces some serving delay but cannot eliminate distribution costs.

## 6. Evidence and references

These sources inform choices; they do not establish end-to-end feasibility for Angelos.

- PyTorch export IR: https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/ir_spec.html
- PyTorch export/control flow: https://docs.pytorch.org/docs/stable/export.html
- PyTorch compilation: https://docs.pytorch.org/docs/stable/generated/torch.compile
- Kornia GPU operations: https://www.kornia.org/projects/
- CV-CUDA operators: https://cvcuda.github.io/CV-CUDA/modules/python/operators.html
- WebGPU limits/lifecycle: https://www.w3.org/TR/webgpu/
- ONNX Runtime browser backend limits: https://onnxruntime.ai/docs/tutorials/web/
- Distribution Matching Distillation: https://arxiv.org/abs/2311.18828
- ImageReward preference data: https://arxiv.org/abs/2304.05977
- Multi-dimensional VisionReward: https://arxiv.org/abs/2412.21059
- Off-policy evaluation and support: https://arxiv.org/abs/1612.01205

## 7. Unresolved experiments

How much campaign diversity can the student retain? Does a compact generative backbone
meet real cold-start constraints? Which UPL encoder preserves useful similarity without
leaking sensitive information? How many distinct judgments are enough for a stable
campaign critic? Which graph regions must remain fixed operations? Does offline CTR
improvement survive a randomized live test? These require measurements, not architecture
claims. No launch date or production performance promise is attached to this plan.
