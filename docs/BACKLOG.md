# Evidence-driven next stages

Update: open-weight teacher adapters, complete-graph datasets and a WebGPU diffusion
student are now implemented. This list remains a research agenda, not an assertion
that those components have achieved useful high-resolution ad quality. See the
current README and diffusion execution report for what has actually been measured.

The first implementation validates infrastructure, not the full product hypothesis.
These tasks are ordered by the risk they resolve. No background agents or recurring
training services are implied by this file.

## 1. Establish a real campaign dataset and licensed teacher

- Select one image campaign, exact approved product assets, copy, target resolution,
  and an explicitly bounded personalization domain.
- Choose an open-weight teacher after checking training and redistribution rights.
- Add a model registry with checkpoint digests, tensor/text IO, memory estimates,
  determinism limits and backend capability declarations. No dynamic untrusted code.
- Make image assets content-addressed and define decoding, color, alpha and resize rules.
- Introduce typed factual-attribute selectors and text conditioning without pretending
  every UPL coordinate is an interpretable trait.
- Generate and retain provenance for full teacher graph outputs across profiles/noise.

Exit: a reproducible teacher corpus with meaningful campaign imagery and honest CPU/
CUDA execution measurements. External model calls, if allowed later, are dataset sources
and host effects, not magically lowerable GPU tensor operations.

## 2. Test whether a campaign student is useful

- Compare a compact pretrained decoder/student with procedural rendering, retrieval,
  and approved cached creative baselines. Compare total delivery cost, not model-only time.
- Keep randomness explicit. Pixel regression of ambiguous outputs can average away detail.
- Split whole profiles and creative families; add out-of-distribution and boundary tests.
- Attempt learned brand/product/text first with independent region-specific metrics.
- Preserve raw results. If fidelity is inadequate, retain exact operations visibly in
  the deployment package; do not improve a metric by hiding failures behind composition.
- Quantize only after float32/float16 quality and browser runtime parity are established.
- Use ONNX Runtime Web before inventing a broad custom GPU operator framework.

Exit: validated student quality, conditioning sensitivity, download size, peak memory,
and cold/warm p50/p95/p99 latency on named hardware. A one-step diffusion model is not
automatically a small model. Check unsupported WebGPU operators at export time.

## 3. Learn a human critic with measured generalization

- Collect blinded, randomized pairwise comparisons with ties/abstentions; retain quality,
  brand correctness, clarity and relevance labels separately.
- Distinguish brand-owner judgment from audience preference. Preserve reviewer identity
  appropriately and separate operator/synthetic fixtures from real audience labels.
- Begin with a pretrained visual representation plus a small campaign preference head;
  100–1,000 labels are not enough evidence to promise training a general critic from scratch.
- Evaluate by held-out creative family and reviewer, compare against simple baselines,
  and report uncertainty/agreement, not only training accuracy.
- Rank brand-valid candidates first; only then test differentiable student refinement
  with an anchored teacher loss, fresh blind evaluations and bounded optimization.

Exit: held-out preference prediction above chance and baseline, without critical-brand
regression. Human preference is not the same thing as click probability.

## 4. Nightly feedback without attribution mistakes

- Add a finite candidate exploration policy with known, versioned probabilities.
- Preserve eligible action sets, chosen candidate and actual propensity. Predicted CTR
  is not a propensity. Do not call a deterministic single-candidate preview an experiment.
- Keep render, speculative bid, won bid, qualified exposure and click distinct.
- Freeze the exposure-time UPL before adding that click to future user state.
- Build datasets from mature labels under current permissions and deletion epochs.
- Evaluate policy candidates with temporal/user splits, support checks, clipped IPS/DR
  diagnostics and effective sample size; distinguish offline evidence from causal lift.
- Start by updating candidate selection. Generator-weight optimization needs its own
  learned objective and trust-region/brand constraints; it is not equivalent to a bandit.
- A nightly candidate may be rejected. Promotion requires minimum evidence, independent
  brand/quality gates, explicit approval, an immutable registry and rollback.

Exit: recover a known policy advantage in a labeled simulator without treating synthetic
uplift as business improvement. Only then seek enough consented live traffic for a real test.

## 5. Make ownership and durability real

- Introduce authenticated users and advertiser identities. Derive tenant scope from
  credentials, never a body parameter. Cross-advertiser profile portability is not carte
  blanche for pooling activity or exposing raw personal metadata to every advertiser.
- Add scoped grants, retention policies, projection/encoder manifests, per-field provenance,
  missingness/confidence, expiry and user correction UI.
- Separate fact collection, personalization, telemetry, training and recipient sharing.
- Extend the local event ledger with durable jobs, cutoff-based replay, late-event handling,
  source trust, and bounded fraud/poisoning defenses.
- Track model lineage to affected people and support artifact retirement/retraining after
  deletion; neither row deletion nor an embedding makes the privacy problem disappear.
- Review actual legal obligations with appropriate expertise before real-user deployment;
  a signup consent checkbox is not an architectural substitute for that review.

Exit: revocation/deletion races, tenant isolation, capability checks, retraining and recovery
tests pass under concurrent use. No public deployment before those boundaries exist.

## 6. Expand formats and authoring only behind contracts

The [full capability audit](MODEL-COVERAGE.md) now defines this scope. It supersedes
any interpretation that three generation adapters cover the intended graph.
Build typed asset/mask/reference/time contracts before broadening the model list;
the audit's integration order applies across the research stages above.

- Tensor/image/text ports with composable reusable subgraphs and separately stored layout.
- Model-specific parameter schemas and unsupported-backend diagnostics.
- Video/audio time axes, frame rate, duration, codecs and temporal consistency tests.
- Visual wiring, undo/redo, import/export, source locations, resource estimates and assets.
- Graph optimization reports, explicitly retained exact operations and portability tests.

Do not label a feature implemented merely because its node name appears in a registry.
