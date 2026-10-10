# UPL v2 local semantic encoder

This implementation encodes **declared stable facts of synthetic adults** using a
fixed local embedding checkpoint. Recent intent, provenance, consent revisions,
and campaign scope remain separate. It does not collect browser histories,
connect advertising identities, infer missing facts, or train a campaign model.
The design background and research limitations are in [RESEARCH-UPL.md](RESEARCH-UPL.md).

## Pinned encoder contract

| Field | Value |
| --- | --- |
| `id` | `qwen3-embedding-0.6b-v1` |
| `modelId` | `Qwen/Qwen3-Embedding-0.6B` |
| `revision` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| `dimension` | 128, 256, 512, or 1024; explicitly requested |
| `pooling` | `last-token`: final nonpadding hidden state |
| `normalization` | `l2`, after taking the first requested number of coordinates |
| `templateVersion` | `angelos.upl-text.v1` |

The official checkpoint is a 1024-wide, Matryoshka-capable embedding model, not
the existing Qwen text-generation node. Its supported dimensional reduction is
prefix truncation **followed by normalization**, not padding, random projection,
or expanding four demo coordinates. A changed model revision requires a new
encoder ID and downstream compatibility checks. A changed text template also
changes the embedding contract. These choices follow the
[official pinned card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md),
[pinned pooling configuration](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/1_Pooling/config.json),
and [Qwen's evaluation implementation](https://github.com/QwenLM/Qwen3-Embedding/blob/main/evaluation/qwen3_embedding_model.py).
The publisher's [model API](https://huggingface.co/api/models/Qwen/Qwen3-Embedding-0.6B)
identified this revision when checked on 2026-10-08. No inference-time lookup of
`main` is performed. The model card declares Apache-2.0; retain publisher notices
when provisioning or redistributing its files.

## CLI and Python interfaces

From the repository root:

```sh
PYTHONPATH=ml .venv/bin/python -m angelos_ml.upl --stdin --device cpu < request.json
node scripts/upl.js --stdin --device cpu < request.json
npm run --silent upl -- --stdin --device cpu < request.json
```

The worker consumes one bounded
JSON request and emit **only one UPL bundle** on stdout. Diagnostics use stderr;
validation/loading/inference failures exit with code 2 and no success payload.
The Node launcher uses the repository virtual environment and forces offline
Hub/Transformers flags. Inputs larger than 262144 UTF-8 bytes are rejected.

Python callers can use `encode_request(request, device="cpu", cache_dir=None)`.
`validate_ingredients`, `prepare_request`, `semantic_text`, `encoder_metadata`,
and `load_local_encoder` are independently available. `LocalSemanticEncoder`
freezes model weights and uses inference mode. Regression tests use offline
fixtures. A separate, explicitly provisioned CPU smoke execution is recorded
below; it does not establish that the resulting semantics improve personalization.

### Request shape

Required request fields are `ingredients` and `dimension`; `ingredientsJson` is
the **only** optional field. Repositories, model revisions, custom code, dtype,
and arbitrary templates cannot be supplied in the request.

```json
{
  "dimension": 512,
  "ingredients": {
    "format": "angelos.upl-ingredients.v2",
    "synthetic": true,
    "adult": true,
    "factsSchema": "angelos.facts.v2",
    "facts": {"ageBand": "25-34", "interests": ["hiking", "painting"]},
    "provenance": [
      {"field": "ageBand", "source": "synthetic:declared", "occurredAt": 100, "receivedAt": 110},
      {"field": "interests", "source": "synthetic:declared", "occurredAt": 100, "receivedAt": 110}
    ],
    "eventsSchema": "angelos.events.v1",
    "events": [],
    "factRevision": 1,
    "eventRevision": 0,
    "consentVersion": 1,
    "campaignId": "synthetic-campaign",
    "asOf": 200
  }
}
```

Times in this minimal fixture are epoch milliseconds, not ISO date strings.
Production transport should supply `ingredientsJson` as described below.

### Validation and temporal scope

- Unknown keys are rejected at every schema level. Duplicate JSON keys, NaN,
  infinities, unsafe integers, booleans masquerading as numbers, real-data flags,
  and nonadult/missing age bands fail before model loading.
- `ageBand` is required and must be one of `18-24`, `25-34`, `35-44`, `45-54`,
  `55-64`, or `65+`. Other facts may be absent or explicitly null.
- `region` is `NA`, `EU`, `APAC`, or `OTHER`; `openness` and `energy` are in
  `[0,1]`. `interests`, `goals`, and `preferredStyles` contain at most 16 unique
  strings each. `preferredLanguage` is free-form declared text, not inferred.
  Fact strings, language, and event topics are bounded to 80 UTF-16 code units,
  matching JavaScript string-length bounds; C0 controls, DEL and unpaired
  surrogates are rejected. Text must already be nonempty and trimmed according
  to JavaScript whitespace rules; it is never silently trimmed or normalized.
- Provenance has exactly one latest entry for **each present fact**, including
  explicit nulls: `{field, source, occurredAt, receivedAt}`. It is not a history
  log. Missing, duplicate, or unknown fact entries are rejected.
- Each of at most 64 events has exactly `{id, kind, topic, occurredAt,
  receivedAt, expiresAt, source, campaignId}`. Kinds are `view`, `click`, `save`,
  and `dismiss`; IDs are unique. All events must belong to this campaign and
  satisfy `expiresAt > asOf`. Invalid events are rejected, not silently filtered.
- All provenance/event observation times satisfy
  `0 <= occurredAt <= receivedAt <= asOf`. Revisions/times are JavaScript-safe
  integers. Fact and consent revisions start at one; event revision may be zero.
  IDs and source labels use `[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}`.
- `events: []` is valid, including when the store suppresses events because
  telemetry consent is off. The store must apply consent, revocation, expiry,
  and purpose/campaign authorization **before** invoking this worker. A consent
  version is an exposure-time binding, not proof that authorization was granted.
  The worker validates the supplied snapshot; it cannot query consent history.

### Exact serialized-byte hash, not cross-language reserialization

Main sends `{ingredients, ingredientsJson: canonical(ingredients), dimension}`.
Its JavaScript canonical form recursively sorts object keys, preserves array
order, and uses `JSON.stringify` for primitive values. Python strictly parses
and deeply validates **both** representations, requires them to agree, then
sets `context.ingredientsHash = SHA256(UTF8(ingredientsJson))`.

The hash binds the exact serialized snapshot, including facts, provenance,
event scope, revisions, and cutoff. It is **not** a vector hash, anonymous user
identifier, signature, or certificate of consent. The main backend verifies it
against the bytes it submitted. Comfy preserves it; it need not reconstruct
ingredients from the reduced output bundle.

If `ingredientsJson` is omitted, Python uses sorted-key JSON with compact
separators, UTF-8 text, and finite numbers only, then hashes those exact bytes.
Python and JavaScript may serialize `1.0`, negative zero, and small exponents
differently. Do not recompute this fallback hash in JavaScript and assume equal
bytes. The worker accepts equivalent JSON serialization with different spacing
but hashes the supplied spacing verbatim; canonicality is the main caller's
contract. Array order remains significant. Tests compare Node-generated hashes
with Python using Unicode, negative zero, and exponent-form numeric facts.

### Semantic template and output

The text is exactly `angelos.upl-text.v1`, a newline, `Declared profile facts:`,
a newline, then Python canonical JSON of the validated **facts object only**.
Present numeric openness/energy values are rendered consistently as floats,
with negative zero collapsed to positive zero; missing and null fields remain
distinct. The supplied facts in the output are not rewritten. No opaque IDs,
campaign IDs, provenance, events, timestamps, revisions, or consent enter the
semantic text. This is a document-style facts template, not an LLM-generated
summary or a claim that Qwen was trained on this particular task.

Tokenization uses the pinned tokenizer, its normal special-token processing,
left padding, and no chat template. No manual EOS duplication is added.
Inputs over 4096 tokens fail before transformer inference: no silent truncation.
Last-nonpadding pooling handles left and right padding. A zero/nonfinite vector
fails rather than producing a fabricated replacement. Output norm validation
uses absolute tolerance `1e-4`; coordinates may be **negative**.

The bundle contains exactly:

```text
{format: "angelos.upl-input.v2", synthetic: true, adult: true,
 factsSchema: "angelos.facts.v2", facts,
 encoder: {id, dimension, modelId, revision, pooling, normalization, templateVersion},
 upl: [dimension finite signed normalized coordinates],
 context: {campaignId, factRevision, eventRevision, consentVersion, asOf, ingredientsHash},
 recentIntent: [{kind, topic, occurredAt, expiresAt}]}
```

Recent intent preserves validated input event order and negative `dismiss`
feedback. Changing only events/context changes the snapshot hash and/or recent
intent, but not the stable semantic text. Coordinates have no named personality
meaning. Untrusted declared text remains data; no tools or generated instructions
are executed. The embeddings are not guaranteed anonymous.

## Cache-only loading

### Local laboratory and snapshots

In `/lab`, save richer fictional audience facts, add a synthetic intent event,
then explicitly encode at 128, 256, 512 or 1024 dimensions. The default is 512.
The successful public bundle can be copied into **Angelos Parse UPL JSON** in
ComfyUI; copying excludes internal profile and snapshot identifiers. A separate,
clearly labeled legacy handoff continues to support existing four-input weights.

The loopback-only API exposes:

| Route | Purpose |
| --- | --- |
| `GET /api/upl/ingredients?profileId=…&campaignId=…` | Permitted facts, latest provenance and at most 64 unexpired campaign events |
| `POST /api/upl/fact` | `{profileId, campaignId, field, value}`; update a typed declared fact |
| `POST /api/upl/event` | `{profileId, campaignId, kind, topic}`; record one synthetic intent ingredient |
| `POST /api/upl/encode` | `{profileId, campaignId, dimension}`; offline worker result plus stored immutable snapshot |

Fact, campaign-event and consent revisions are independent. Event revisions
exported in a bundle count only that campaign's events. Facts remain separate
from the encoder revision. Events expire after seven days by default; revoking
personalization/telemetry or removing a campaign expires its permitted history,
so regranting permission cannot resurrect those events. Profile deletion cascades
through events and snapshots. Previously copied files cannot observe revocation.

Encoding rechecks facts, events, consent and expiry after inference; concurrent
changes reject the result rather than attaching an embedding to newer state.
Exposures require a matching complete encoder contract and freeze the complete
v2 bundle for later feedback. Qualified attributed clicks can also add a separate
`Ad engagement` intent event; old clicks and exhausted intent capacity never
prevent an otherwise valid click from being recorded. A manually added `click`
intent event does **not** create an exposure, conversion label or reward signal.

The HTTP editing forms are for a single local operator, not a distributed event
ingestion service: they assign mutation IDs server-side. Store-level methods
support explicit event IDs and fact revision checks; a production collector would
also need authenticated clients, caller-provided idempotency/revision contracts,
bounded retention compaction and policy enforcement beyond this synthetic demo.

### Provisioning and execution

The runtime uses PyTorch and the repository's installed Transformers API
(tested with Transformers 5.18.0); no dependency/package files are changed.
CPU is the default and uses float32 weights/inference. Explicit `mps`/`cuda`
requests require that device and never fall back to CPU. OOM/unsupported
operations propagate without changing precision, context budget, or model.

The expected Hub cache layout is:

```text
<cache-dir>/models--Qwen--Qwen3-Embedding-0.6B/snapshots/
  97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/
    config.json
    tokenizer_config.json
    tokenizer.json
    model.safetensors
```

Supply the complete official snapshot, not an independently edited set of these
minimum checked files. `--cache-dir` identifies the **Hub cache root**, not the
snapshot itself. Defaults honor `HF_HUB_CACHE`, then `HF_HOME/hub`, otherwise
`artifacts/huggingface/hub` in this repository. Loading checks the fixed snapshot
and basic architecture before calling `AutoTokenizer`/`AutoModel` on the local
path with `local_files_only=True`, `trust_remote_code=False`, and safetensors.
Custom `auto_map` code is rejected. The cache-path/config check is not an
independent cryptographic attestation of every weight byte; cache provisioning
must remain trusted and controlled.

Missing/incomplete weights yield an actionable error naming the expected cache
path and revision. **This runtime never downloads weights and has no download
command.** Provisioning requires a separately authorized operation.
Do not substitute the existing Qwen generation checkpoint, a different revision,
zeros, random vectors, or a 4D projection. No paid service is involved.

## Trainable adapter: untrained, not generator integration

`UPLConditioningAdapter` in `ml/angelos_ml/upl_adapter.py` is a bounded PyTorch
MLP taking three explicit float feature batches:

```python
from angelos_ml.upl_adapter import UPLConditioningAdapter

adapter = UPLConditioningAdapter(
    semantic_dim=512, recent_dim=8, context_dim=8,
    num_tokens=4, token_dim=256, hidden_dim=256,
    feature_schema="my-campaign.recent-context.v1", seed=2026,
)
tokens = adapter(semantic, recent, context)
```

Shapes are `[batch, semantic_dim]`, `[batch, recent_dim]`, and
`[batch, context_dim]`; output is `[batch, num_tokens, token_dim]`. Semantic
inputs must already be L2-normalized. Batch alignment, width, finite values,
dtype, and device are checked; there is no implicit transfer or coercion.
Initialization uses an isolated seeded CPU RNG. Parameters and all three input
branches receive gradients. The model is capped at eight million parameters,
16 output tokens, and batches of 256. No optimizer/training job runs by itself.

The **caller-defined** `feature_schema` must specify how recent topics, recency,
missingness and authorized campaign/placement context become feature tensors.
No such learned temporal/topic encoder is supplied by this module; it does not
pretend event strings or campaign identifiers already have learned coordinates.
`specification()` records encoder identity, shapes, feature schema and seed,
with `initialization="random-untrained"`, `pretrained=false`, and
`validatedGenerator=null`. These are initialization/provenance declarations,
not an automated measurement of subsequent training success. Persist this
specification with separately trained state dictionaries; loading weights alone
cannot establish compatible feature semantics.

**These tokens are not automatically valid FLUX, SANA, CLIP, DDPM, or Comfy
conditioning.** A generator-specific bridge, actual training data/objective,
compatible artifact version, and held-out evaluation remain required. Downstream
v2 width-aware training, reward and diffusion contracts are being integrated
separately; their final compatibility matrix belongs to the integration record.
Accepting wider tensors is not evidence of trained conditioning or HD alignment.
Legacy four-coordinate generator/reward weights cannot consume this bundle by
changing a dimension constant or taking its first four values. Separate v2
training-contract work may accept wider inputs, but compatible weights still
need training and evaluation. Keep the deterministic 4D regression distinct.

## Tests and remaining evidence

### Downstream compatibility

| Boundary | Current support |
| --- | --- |
| Store, local API, browser handoff | Rich facts, expiring scoped intent, immutable full v2 bundles, exact encoder compatibility |
| ComfyUI plugin and pack importer | Both legacy and semantic bundles; typed fact/intent strings and an opaque embedding block, never an invented CLIP conditioning tensor |
| Backend-independent graph JSON | `angelos.graph.v2` binds the full semantic encoder and checks feature indices against its width; old textual syntax stays v1 |
| Python diffusion, reward and RL primitives | Versioned wide-vector models/checkpoints and frozen feedback contracts; native32 regression experiments, not an HD model |
| Python dataset training | Explicit frozen per-sample bundles, full encoder binding and stable person-disjoint split validation; no synthetic four-number fallback |
| Legacy graph teachers/compiler and WebGPU student | Explicitly reject unsupported semantic v2 execution instead of silently consuming four coordinates |
| HD generation | Remains a separate prompt-conditioned baseline; the model-specific UPL adapter has not been trained or evaluated |

V2 reward and diffusion experiments currently use the **semantic vector branch**.
They retain recent intent as frozen provenance but do not pretend it is learned
temporal conditioning. The token adapter exposes that future branch explicitly;
its generator bridge, feature construction and training remain separate work.
Imported ComfyUI teacher packs are not automatically converted into a training
dataset, and importing examples does not manufacture ratings or click labels.

```sh
PYTHONPATH=ml .venv/bin/python -m unittest discover -s ml/tests -p 'test_upl*.py' -v
```

The encoder and adapter modules have **23 focused tests** (17 encoder/CLI and
six adapter). The coordinated UPL-pattern run also included 14 separate
training-contract tests: **37 passed** before the final trim/DEL parity update.
Those training tests are separate work, not evidence that this adapter is trained.

Tests use mocked local-loading fixtures and a tiny randomly initialized real
Qwen3 transformer, never downloaded/pretrained weights. They cover strict
schemas, nullable facts, provenance, temporal/campaign exclusions, exact byte
hash interoperability, padding/pooling and prefix normalization, frozen encoder
inference, missing-cache errors, CLI stdout, adapter shapes, gradients, weight
updates and RNG isolation. They do **not** measure semantic quality, production
latency, pretrained memory requirements, learned personalization, privacy risk,
human image quality, observed CTR lift, or conversion lift. Those remain
separate experiments with appropriate authorized data and held-out controls.

## Recorded real CPU smoke execution

On **2026-10-08**, the coordinating operator explicitly provisioned the public
pinned checkpoint (approximately 1.21 GB; reported download time 22 seconds)
as a separate authorized operation. The actual **JavaScript runner → Python CPU
encoder → store snapshot** path then succeeded for one synthetic adult with a
synthetic `save` event. The inspected ignored artifact `artifacts/upl-smoke.json`
records **512 coordinates**, **267 negative coordinates**, norm
**0.9999999803845473**, and **21110.284417 ms** for the complete worker including
cold loading and inference. It records the exact revision in the table above.

This is actual local pretrained-model execution, distinct from mocked/random
unit fixtures. It is not a sustained-throughput benchmark, trained adapter run,
semantic-quality evaluation, real-person experiment, or evidence of CTR lift.
There was no paid job, and no automatic-download code was added to the runtime.
See [UPL-EXECUTION.md](UPL-EXECUTION.md) for the coordinating execution record.
