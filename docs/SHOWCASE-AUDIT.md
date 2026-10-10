# Angelos showcase audit

Audit date: 2026-10-08. Findings describe the starting repository; source line numbers identify that baseline and may move as fixes land.

Later implementation update: native ComfyUI authoring/export and UPL adapters are
documented in [COMFYUI.md](COMFYUI.md); bounded click + critic candidate training
in [RL.md](RL.md); actual local 1080p foundation-model execution in
[HD-EXECUTION.md](HD-EXECUTION.md). The baseline table below is historical, not a
claim that those modules remain absent. Human-rated critic generalization,
campaign-trained HD quality and measured CTR lift remain unproven.

The resulting implementation separates the explanatory `/` from `/lab`, adds a genuine `generate_text` node, packages recorded teacher/student evidence, makes rebuild commands graph-aware, and exports a static-only showcase. The concept is credited as **GPU Accelerated Ads — a proof of concept by Louis Anderson**. These changes improve communication and reproducibility; they do not resolve the failed personalization audit or establish production readiness.

See [showcase execution](SHOWCASE-EXECUTION.md) for actual test outcomes, real text-model execution, and explicit remaining validation limits.

## Verdict

**There is a credible engineering experiment here, but not yet a demonstrated personalized creative product.** The strongest contribution is the combination of a typed campaign program, an independently trained campaign artifact, browser execution, and exposure-time identity/consent contracts. A longer model catalog or more polished canvas does not establish that this combination is useful.

The immediately compelling story is: **“A creative graph is the authoring artifact; campaign weights are the delivery artifact. Angelos tests whether that separation can make accountable, locally conditioned creative practical.”** The repo supports the architecture and a narrow numerical experiment. It does not yet support useful personalization, competitive creative quality, business lift, production privacy or exclusive ownership of the general idea.

## Top three actions for this turn

1. **Give a fresh checkout something useful before ML setup.** Separate `npm ci && npm start` exploration from optional teacher reproduction. The new homepage must explain the experiment without a model, Python or WebGPU; the lab must show explicit missing-artifact/device states rather than imply generation succeeded.
2. **Ship one inspectable proof, not another capability claim.** Provide a genuine recorded teacher/raw-student/delivered comparison with provenance and the failed conditioning result. Keep recorded, procedural, test-double and live-model outputs visibly distinct. Do not manufacture an attractive AI result to fill a screenshot.
3. **Repair the edit → rebuild instructions.** The current dialog always downloads Tiny-SD and reuses the default dataset filename, but datasets deliberately refuse overwrite. Generate commands from the selected graph and pass a new dataset path consistently to collection and training. Keep the last source recoverable.

These three priorities are addressed by the accompanying implementation. Remaining acceptance gates and the research backlog below should not be confused with demonstrated results.

## What is actually present

| Capability | Repository evidence | Honest showcase claim |
| --- | --- | --- |
| Typed authoring representation | `src/core/graph.js:45` validates version, node/parameter schemas, port types, cycles, reachability and image output; `src/core/graph.js:111` hashes canonical graph content. | A bounded typed DAG and source language, not arbitrary graph execution or a general multimodal editor. |
| Model-backed teacher path | `ml/angelos_ml/providers.py:34` loads pinned local models without remote code or implicit download; `ml/angelos_ml/dataset.py:44` renders complete graphs and splits profiles before sampling. | Real local teacher adapters exist. The earlier report records Tiny-SD execution, not executed FLUX/LTX/CUDA coverage (`docs/DIFFUSION-EXECUTION.md:20`). |
| Learned browser student | `scripts/train.js:23` invokes diffusion training; `web/app.js:446` validates artifacts and initializes the GPU renderer. | An earlier learned 32×32 student ran in actual Chrome WebGPU. Historical parity/timing is documented, not rerun here (`docs/DIFFUSION-EXECUTION.md:44`). |
| Quality boundaries | `docs/DIFFUSION-EXECUTION.md:82` records 40 teacher samples, failed correct-UPL advantage and failed raw brand fidelity. The selected local metrics also record `improvedWithCorrectUpl: false`. | Learned weights and numerical parity are real evidence; useful personalization and campaign-quality output remain unproven. Exact brand composition is a separate correction. |
| Local accountability | `src/core/store.js:390`, `src/core/store.js:408`, `src/core/store.js:452` bind snapshots, model identity, exposure and click windows; `src/core/store.js:359` handles consent changes. | A synthetic local attribution/permissions experiment, not verified attention, a randomized experiment or production user ownership. |
| Authoring UI | `web/app.js:215` edits parameters; `web/app.js:256` applies validated source; `web/app.js:196` rearranges nodes. | Source editing plus parameter inspection and an arrangeable graph. Dragging nodes is not visual wiring. |
| Learning loop | `src/server.js:115` exports mature rows and explicitly says no trainer/promotion is scheduled; `scripts/train.js:35` selects a preview after a training sanity gate. | Recorded reviews and eligible examples exist. No trained preference critic, CTR-driven optimizer or automatic production promotion. |
| Text generation and route split | `generate_text` and the `/` versus `/lab` split were added after baseline inspection; see `docs/TEXT.md`. | Deterministic `prompt` is separate from local Qwen inference. Language generation is a teacher intermediate, not a browser text student. |

## P0 — Critical demo blockers

### 1. Fresh-clone experience and recovery depend on the developer's machine

**Evidence.** `README.md:20` puts Python, a model download, dataset generation and training before first launch. Yet `src/server.js:67` already permits an absent preview pointer. `.gitignore:9` excludes both data and artifacts; a new checkout does not inherit the working local student. `web/app.js:455` handles a missing artifact, but other initialization errors reach the outer catch before handlers are installed (`web/app.js:485`). The lab cannot be considered portable merely because it works with today's cached weights.

**Implement now.** Give README and homepage two distinct paths: explore without ML; reproduce the real teacher/student experiment explicitly. Make missing, incompatible, corrupt and unsupported-GPU states persistent and actionable, with graph/source navigation still usable. Show graph compatibility separately from model availability and GPU availability. Keep the CPU/procedural reference labeled as such; do not silently substitute it for the student.

**Acceptance gate.** In a separate fresh directory, with no `data/`, `artifacts/`, `.venv` or model cache, start the Node app and visit `/` and `/lab`. Both explain their state and allow their intended navigation. No inference/download occurs merely by loading a page. Repeat with missing/corrupt/stale artifacts and without WebGPU; the result is an honest recoverable state, not an inert page. Record the actual environment and result; dependency installation has not been validated by this audit.

### 2. The demonstrated graph-edit workflow leads to a preventable rebuild dead end

**Evidence.** Changing graph content correctly invalidates its student (`web/app.js:263`). The build dialog hardcodes `download tiny-sd` and default collection/training commands (`web/index.html:61`). `scripts/dataset.js:14` defaults to `artifacts/<campaign>.dataset.npz`; `ml/angelos_ml/dataset.py:111` refuses an existing dataset or manifest. `scripts/train.js:16` defaults back to that same path. All existing presets use the Aurora workflow, so switching a teacher can collide with an old corpus. The README mentions an alternate output (`README.md:78`), but the primary UI action does not provide it.

**Implement now.** Derive the explicit model-download list from the active graph, including the text model when integrated. Supply a fresh, non-overwriting dataset path and use that exact path as training's third positional argument. Derive preview-output directories similarly. Show the saved source path and selected device. Include a downloadable/copyable source backup and a deliberate restore action; no server-side shell execution from graph input.

**Acceptance gate.** After changing a parameter, copy the shown commands and produce a compatible replacement artifact without deleting an existing dataset. Repeat after loading a different teacher preset; no Tiny-SD-only instructions remain for another model. Invalid source leaves saved source unchanged; restoring the prior source restores its hash and allows its matching artifact. The command-producing UI now derives models and fresh dataset paths from the saved graph; a complete replacement training run remains a separate validation step.

### 3. Route migration, screenshot capture and public sharing need different boundaries

**Evidence.** The baseline serves the lab at `/` through an explicit static allowlist (`src/server.js:121`). Browser scenarios navigate to `/` and screenshot into ignored `test-results/` (`test/browser/laboratory.spec.js:8`, `.gitignore:11`). The test server isolates data but still consumes the repository's default artifacts (`scripts/test-server.js:6`); Chrome is explicitly required (`playwright.config.js:7`). Host/Origin checks accept only loopback origins (`src/server.js:98`). This is not an app that can safely be made public by changing its bind address.

**Implement now.** Coordinate `/lab` navigation changes with the existing browser tests; add a separate homepage/no-artifact smoke scenario. Make screenshot capture wait for a declared ready state and attach the graph/artifact/seed/output-mode metadata. Put approved share assets in a deliberate distributable location, not solely ignored run directories. If shipping a static homepage, make its assets/links portable and its lab CTA explicitly local-only or unavailable in the hosted build.

**Acceptance gate.** A cold visit and reload work at both routes, desktop and narrow viewport. The homepage still communicates when ML/GPU is absent. A screenshot or recording can be viewed outside this machine with its mode/resolution/provenance caption. A static showcase works under its intended hosting path without calling local profile APIs or presenting a dead live-demo link. Public sharing does not expose the unauthenticated ledger, artifact mutation or training machinery. Actual public hosting is a separate authorized action, not required to finish this audit.

## P1 — Credibility gaps

### 4. Make evidence travel with the claim, including negative results

**Evidence.** The generative teacher canvas intentionally displays an offline placeholder (`web/app.js:95`), while native PNGs exist only in ignored `artifacts/teacher-native/`. The recorded warm benchmark measures 32×32 inference plus readback, not high-resolution generation or full delivery latency (`docs/DIFFUSION-EXECUTION.md:63`). The report records a correct-UPL result worse than shuffled UPLs and a failed learned brand gate (`docs/DIFFUSION-EXECUTION.md:85`). Browser parity depends on learned artifacts and can explicitly skip if they are unavailable (`test/browser/diffusion.spec.js:19`); the laboratory suite also reads local parity files (`test/browser/laboratory.spec.js:25`).

**Implement now.** Add one compact evidence panel: native teacher, native raw student, explicit delivered composition, model/revision, graph hash, artifact identity, UPL, teacher and student seeds, sample split, dimensions, execution date and device. Use only genuine retained outputs with verified matching metadata. Record that different teacher/student processes are not an exact pixel reconstruction merely because their seed numbers match. Put “recorded experiment,” “32×32 enlarged,” “conditioning not demonstrated” and “exact brand overlay” next to the relevant images/metric, not only in the footer. If a matched set is unavailable, say so rather than assembling a misleading comparison.

**Next milestone.** Produce an evidence bundle with checksums, relative paths, regeneration commands and a machine-readable run summary distinguishing executed, failed and skipped checks. Separate model-free tests from artifact-dependent hardware checks. A designated evidence run must fail or report incomplete when its real artifact/hardware prerequisites are absent; a skipped test is not a GPU pass.

**Acceptance.** Another person can locate the exact sample and its provenance without `/Users/main/Angelos`, distinguish recorded/live/procedural output from the caption alone, and reproduce the eligible experiment or see precisely what dependency is missing. New text-generation claims require a genuine local-weight run with pinned model/tokenizer revision, explicit sampling parameters, output, hardware and failure behavior. An adapter, mock or template does not satisfy that gate. Old JavaScript/Python/browser pass counts are not promoted to today's release results.

### 5. Ownership, consent and security language must match the actual boundary

**Evidence.** The server already has valuable local protections: static allowlisting, bounded JSON requests, loopback Host/Origin checks and CSP (`src/server.js:14`, `src/server.js:92`). Providers forbid remote code and implicitly downloaded models (`ml/angelos_ml/providers.py:34`). These do not create authenticated users or tenant isolation. Every stored profile is designated synthetic (`src/core/store.js:238`); current training builds independent synthetic vectors (`ml/angelos_ml/dataset.py:23`). Exposure qualification is client-supplied evidence, not attested attention (`src/server.js:155`, `src/core/store.js:419`).

Two specific claim corrections matter:

- **Reviews are not necessarily exposure-qualified.** `web/app.js:423` requires a saved, rendered candidate but no impression; `src/core/store.js:538` stores a review without requiring an exposure record. Keep candidate/operator review distinct from audience feedback on an exposed creative. `README.md:95` currently instructs exposure before a review, which is stronger than enforcement. Either label the distinction or intentionally enforce the stronger contract; do not imply it exists now.
- **“Human reviews” and “Observed CTR” need provenance/context.** `web/app.js:319` renders these labels; `src/server.js:87` totals operator and synthetic reviews. The store already separates review sources and pending/matured exposures (`src/core/store.js:512`). Expose those distinctions, use “synthetic sandbox”/“operator review,” and show an empty denominator as no data rather than 0% performance. A single-candidate propensity of one (`src/server.js:160`) is not causal lift evidence.

**Implement now.** Carry the local-synthetic scope into homepage, lab and captured assets. Explain separately what runs in the browser, what stays on the local server, and what an explicitly authorized model download contacts. Preserve the warning that record deletion does not unlearn weights (`README.md:122`). Avoid “privacy guaranteed,” “secure production platform,” “user-owned identity network” or “brand-safe AI” as demonstrated capabilities. A graph/model hash proves consistency, not independent provenance or authorship attestation (`ml/angelos_ml/train_diffusion.py:1`).

**Publication check, not an allegation.** `docs/MODEL-COVERAGE.md:7` discloses inspection of user-provided local language/catalog materials and retains filenames and inventory counts; `docs/RESEARCH-OPERATIONS.md:7` maps an inspected action registry. `docs/EXECUTION.md:11` records the conceptual reference and private repository context. The audit has not examined those external materials or established copying. Before public release, the owner should review permission to publish these reference-derived details and distinguish independent Angelos implementation from upstream inspiration. Do not erase truthful provenance to make a stronger independence claim. Do not frame a catalog replacement exercise as the product's original contribution.

**Acceptance.** Public copy has no stronger consent/exposure/security claim than the implemented contract. The owner approves the publication scope and asset provenance; authorship/contributor credits are factual, and upstream models/tools are attributed. The inspected root has no project `LICENSE` file: decide and state intended source/asset distribution terms rather than calling the project open source by implication. This is a release-readiness gap, not a legal conclusion or a substitute for checkpoint-specific permissions review.

### 6. “Author a graph” needs one complete, recoverable user action

**Evidence.** The canvas supports selection/arrangement but its edges are rendered, not visually editable. Parameters are raw JSON inputs (`web/app.js:215`); positions are held only in a `Map` (`web/app.js:20`). Structural edits require source. The typed graph has strong validation, but no visual add/connect/delete/undo flow is established by this snapshot. A user cannot infer the actual authoring capability from the polished pipeline diagram alone.

**Implement now.** Be explicit: “inspect and edit source/parameters” until visual structural authoring exists. Provide one guided edit with its expected graph-hash change and rebuild consequence. Preserve draft text and the last valid source on failures. Use parameter-specific controls for the showcased node rather than asking a first-time visitor to discover JSON quoting/ranges. Pending text generation should make the distinction between prompt construction and model inference obvious.

**Next milestone.** Implement one end-to-end visual editing slice, not a broad editor framework: insert an allowed node, connect type-compatible ports, edit parameters, undo, export/import and restore. Keep view/layout metadata separate from the semantic graph hash and fail invalid connections before save. Reuse the existing typed registry rather than duplicating validation in the UI.

**Acceptance.** A new user completes the chosen graph edit, understands why the old student becomes stale, and can undo it without losing a working demonstration. The next visual slice round-trips the same semantic graph through source and canvas, and layout-only changes do not invalidate learned artifacts. Do not delay the first honest showcase for subgraphs, collaboration or a full asset browser.

## Founder story and defensible product thesis

**Choose the first audience.** The present evidence is best suited to a technical collaborator or design partner evaluating an architecture, not a marketer expecting production-ready personalized ads. `README.md:3` leads with the mechanism and `README.md:15` immediately expands into a capability audit. Neither supplies a concise founder motivation, specific initial customer problem or a bounded request to the reader. Do not invent the founder's biography or customer validation.

**Implement now.** Put this sequence in the homepage and first README screen: the problem; the specific architectural bet; one genuine result; its most important failure; how to explore; what feedback/collaboration is wanted. Explain UPL on first use as a versioned conditioning representation, with the current four declared synthetic features—not a learned understanding of a person. Link deep model research after the proof and constraints, not as a substitute for either. Name the author/contact only using confirmed information. Include a viewable proof asset and a portable lab/exploration link, not a machine-specific path.

**Defensible today:** an inspectable implementation that separates authoring, teacher execution, training, evaluation and delivery, while preserving consent-aware outcome identity. These are engineering choices the repository can substantiate. **Not defensible today:** “first,” a proven moat, ownership of graph-based AI in general, FLORA parity, better personalization, lower campaign-wide cost or improved CTR. The repository alone establishes none of those.

**The falsifiable bet:** for one tightly bounded creative family, a campaign-specific student can meet an agreed quality/brand threshold and deliver relevant conditional variation at lower total delivery burden than cached approved variants or invoking the teacher. If caching wins, the project should retain its useful authoring/provenance work rather than assume every campaign needs learned weights. Provenance and portable contracts are useful only if a real author can use them; a static canvas is not the whole product.

## Ordered milestones, not a feature wishlist

| When | Scope | Exit gate |
| --- | --- | --- |
| **This turn: honest shareable demo** | Parent completes the route split, artifact-independent entry, selected-graph rebuild commands, claim corrections and one genuine proof asset. Integrate `generate_text` without overstating execution. | Parent checks the P0 scenarios and reports pass/fail/pending by environment. Homepage/README/captures agree about what is live, recorded and unsupported. Audit-only edits do not close these items. |
| **Next: reproducible authorship proof** | Portable evidence bundle, explicit artifact-dependent validation, one recoverable visual authoring slice, confirmed author/upstream credits and publication scope. | A second person can inspect the proof, change one graph and reproduce or precisely diagnose the result without the original machine. |
| **Then: one useful campaign experiment** | One approved creative family; teacher/student/cached-variant baseline comparison; profile/family holdouts; independent brand, relevance and quality review; cold and warm delivery costs. | Predeclare quality/brand requirements and compare under the same resolution, distribution and hardware budget. Demonstrate meaningful correct-conditioning advantage and measured utility/cost tradeoff, or record failure. Noise MSE alone cannot pass. |
| **Later, only if warranted** | Preference learning/causal traffic experiments; wider modalities; pretrained/latent students and compression; authenticated multi-tenant serving and deletion/retraining lineage. | Each depends on the useful-campaign evidence or a separately justified need. Real personal data/public service require explicit authorization and a new threat/consent design. No automatic nightly training, ad spend or public API exposure is implied. |

**Stop rule:** do not spend this turn expanding the catalog, polishing invented output, claiming new test results from old reports, or deploying the local API publicly. The highest-value result is a small demonstration whose inputs, learned behavior, limits and provenance another person can inspect and believe.
