# ComfyUI → Angelos programs and HD teacher packs

## Workflow to program

The lab’s **Convert a ComfyUI graph to a program** tool accepts a limited, explicit subset of ComfyUI **API-format JSON**. File selection and translation stay in the browser. You can download readable `campaign.comfy.ang` source and compile it back into checked API JSON. Neither direction executes nodes, loads weights, downloads models, trains a student or changes the built-in Aurora recipe.

The `comfy_program v1` dialect is a separate, typed Comfy backend program, **not** the existing `campaign` student-training language or a CUDA compiler. Keeping checkpoint, sampler, negative-conditioning and decoder operations explicit avoids falsely replacing arbitrary Comfy workflows with a different image pipeline. Port types come from the reviewed catalog; `node_1[1]` means output slot 1 of node 1. It is a CLIP output for `CheckpointLoaderSimple`, not a MODEL or VAE. Type errors fail conversion.

Accepted classes: `CheckpointLoaderSimple`, `CLIPTextEncode`, `EmptyLatentImage`, `KSampler`, `VAEDecode`, `ImageScale`, `SaveImage`, `PreviewImage`. The catalog follows the [official ComfyUI core node contracts](https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py), checked 2026-10-09, with narrower limits:

- API object keyed by positive numeric node IDs; 1–64 nodes, at most 256 KiB. Canvas/layout JSON, request envelopes, custom nodes (including Angelos UPL nodes), FLUX/LTX-specific loaders and other classes fail explicitly. Continue using those workflows in ComfyUI with the separate image-pack export path below.
- Every execution input must be present. No unknown inputs, implicit defaults, dropped branches or disconnected nodes. At least one SaveImage/PreviewImage sink is required. Cycles, missing references and incompatible output slots fail. Optional `_meta.title` is omitted as display-only metadata; all execution inputs are preserved.
- Empty latent dimensions: 16–4096, multiples of 8; batch 1–8. These are translation limits, not a GPU memory budget or model-resolution recommendation. An ImageScale operation remains an explicit resize, not proof of native HD generation.
- KSampler: 1–100 steps; CFG 0–100; denoise 0–1. Samplers: `euler`, `euler_ancestral`, `heun`, `dpmpp_2m`. Schedulers: `normal`, `karras`, `exponential`, `simple`. Seeds must be nonnegative JavaScript safe integers; larger Comfy 64-bit seeds are rejected rather than rounded. Duplicate JSON keys and nonfinite numbers are rejected.
- Checkpoint names remain literal strings, never resolved or downloaded. Translation checks graph structure, not checkpoint existence, licensing, compatibility, memory needs or image quality. Review the program before running it yourself in a trusted Comfy installation. Prompt text is data; no code evaluation occurs.

Local CLI, with fresh output paths:

```sh
npm run comfy:program -- import examples/aurora-comfy-api.json --output artifacts/aurora.comfy.ang
npm run comfy:program -- compile artifacts/aurora.comfy.ang --output artifacts/aurora-api.json
```

The example’s `YOUR_LOCAL_CHECKPOINT.safetensors` is a placeholder. It has not been generated or quality-tested; choose a trusted model and suitable settings yourself. Generated source contains statements such as:

```text
comfy_program v1
node_1 = CheckpointLoaderSimple(ckpt_name: "YOUR_LOCAL_CHECKPOINT.safetensors");
node_2 = CLIPTextEncode(text: "A backpack beside a lake", clip: node_1[1]);
```

This excerpt is incomplete; the supplied API example includes positive/negative conditioning, sampler, decoder and output. Importing finished images for distillation is a different operation from converting a graph into a program. No automatic program-to-student training bridge is claimed.

## HD teacher packs

**Implemented scope:** manually installed custom nodes export actual ComfyUI IMAGE tensors and versioned metadata to local files. **Workflow authoring, model loading and execution stay in ComfyUI.** This bridge is not a workflow execution endpoint in Angelos and does not call its API or train automatically.

For installation and click-by-click wiring, start with the [custom-node README](../integrations/comfyui_angelos/README.md). Install only into an existing trusted ComfyUI environment; no cloud account, model download or automatic installation is needed. The bridge uses the supported V1 node API for a simple standalone package, based on the [official node properties](https://docs.comfy.org/custom-nodes/backend/server_overview), [custom node discovery](https://docs.comfy.org/custom-nodes/backend/lifecycle), [hidden input](https://docs.comfy.org/custom-nodes/backend/more_on_inputs), and [IMAGE datatype](https://docs.comfy.org/custom-nodes/backend/images_and_masks) documentation, reviewed 2026-10-08. The [core SaveImage source](https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py) supplies the UI file-descriptor convention; [folder_paths](https://github.com/Comfy-Org/ComfyUI/blob/master/folder_paths.py) supplies the configured output root.

## Node and file contract

| Node | Inputs | Outputs / behavior |
| --- | --- | --- |
| `AngelosAudience` | `preset`, `facts_json`, `synthetic` | `ANGELOS_AUDIENCE` metadata and its JSON STRING. Only synthetic adult facts with the existing `structured-demo-v1` UPL ordering. No real-person consent is created or inferred. |
| `AngelosCampaign` | `audience`, `campaign_id`, `version`, `reference_prompt` | `ANGELOS_CAMPAIGN` metadata plus the unchanged prompt STRING for a compatible Comfy text encoder. Campaign hash binds all four inputs, including audience contents. |
| `AngelosExportCreative` | Final `IMAGE`, `campaign`; hidden `PROMPT`, `EXTRA_PNGINFO`, `UNIQUE_ID` | Output node: native-dimension PNGs and manifest; Comfy `ui.images` descriptors and a STRING `manifest_path` relative to the Comfy output root. |
| `AngelosParseUPL` | `upl_json` STRING | Validated `ANGELOS_AUDIENCE`, canonical audience JSON STRING. Unhashed wire input described below. |
| `AngelosUPLFactString` | `audience`, `field`: ageBand/region/preferredLanguage/interests/goals/preferredStyles | Actual STRING from an explicit fact; list facts joined by `, `. Missing optional facts fail, never inferred. |
| `AngelosUPLFactFloat` | `audience`, `field`: openness/energy | Actual FLOAT from an explicit fact. |
| `AngelosUPLVectorFloat` | `audience`, `index` within declared dimension | FLOAT `opaque_coordinate`, without inferred trait semantics; 0–3 for v1, up to 1023 for v2. |
| `AngelosUPLEmbeddingBlock` | v2 `audience` | `ANGELOS_EMBEDDING` containing the full validated audience bundle; not CLIP/Comfy `CONDITIONING`. No model adapter is included. |
| `AngelosUPLRecentIntentText` | v2 `audience` | STRING of explicit `kind: topic` lines, or empty STRING for an empty event list. No inferred intent/score. |
| `AngelosScalarToInt` | FLOAT value/input bounds, INT output bounds, rounding | INT affine range mapping; rejects out-of-range values, no silent clipping. Nearest rounds ties toward positive infinity. |
| `AngelosScalarMap` | FLOAT value/input bounds/output bounds | FLOAT affine range mapping, no clipping. Input bounds strictly increase; output bounds are ordered. Example `[0,1]` → `[2,7.5]` maps `0.5` to `4.75`; wire only to a compatible model-valid CFG/strength parameter. |
| `AngelosScalarCompare` | FLOAT value/threshold, operator | Actual BOOLEAN from one of `>=`, `>`, `<=`, `<`, `==`. |
| `AngelosPromptTemplate` | template, text, value, integer_value, enabled | STRING using only `{text}`, `{value}`, `{integer}`, `{enabled}`. No indexing, attribute access, evaluation or format specifications. |
| `AngelosChoosePrompt` | BOOLEAN condition, two STRING prompts | Selected actual STRING, not metadata-only labeling. |

### UPL wire input and typed conditioning

The preserved **v1** UI Copy JSON contract is an **unhashed seven-field envelope**:

```json
{"format":"angelos.upl-input.v1","synthetic":true,"adult":true,"factsSchema":"angelos.facts.v1","facts":{"ageBand":"35-44","region":"NA","openness":0.25,"energy":0.2},"encoder":{"id":"structured-demo-v1","dimension":4},"upl":[0.4,0,0.25,0.2]}
```

Only the independently versioned `angelos.facts.v1` and `structured-demo-v1` dimension-4 combination is supported in v1. Unsupported learned encoders/dimensions fail explicitly; no generic learned-vector semantics are assumed. The four-coordinate formula exactly follows `src/core/store.js`'s `encodeFacts`, with the ordering below. Every fact is mandatory, and coordinates must equal the declared encoder's result, not merely pass a supplied hash. Numeric `0` and `0.0` are equivalent on input. JSON size ≤16 KiB; duplicate keys, extra fields (including supplied hashes), wrong versions, nonfinite/out-of-range/non-numeric coordinates, missing facts, minors and non-synthetic declarations are rejected. Only fictional adult test inputs are authorized by this bridge; real-profile authorization is not implemented.

UI code can use `JSON.stringify(envelope)` directly. It must adapt authorized synthetic data into the exact envelope, not paste a whole profile, exposure snapshot or hashed canonical audience. Python validates first, then builds the established `angelos.comfyui-audience.v1` with `preset: custom` and its own hash. The parser's canonical JSON output is a diagnostic/export representation, not wire input. The normalized audience is serialized unchanged into both `manifest.audience` and `manifest.campaign.audience`; **teacher-pack v1 and its hash/binding contract do not change**. Workflow provenance retains original parser JSON and submitted connections, not proof of conditioning execution.

Actionable wiring (see [standalone README](../integrations/comfyui_angelos/README.md) for widget conversion and full examples):

```text
ParseUPL.audience -> FactFloat(energy).fact_value -> ScalarCompare(>= 0.5).condition
                 -> Campaign.audience                |
                                             ChoosePrompt.prompt
                                                     |
                                             Campaign.reference_prompt
                                             Campaign.prompt -> CLIPTextEncode.text
                                             Campaign.campaign -> ExportCreative.campaign
Existing model-compatible text-conditioning/sampler/decoder -> ExportCreative.images
```

For formatting, connect `FactString(region)` to `PromptTemplate.text` and `FactFloat(energy)` to `PromptTemplate.value`, then wire the resulting STRING through Campaign to CLIPTextEncode. Numeric facts can instead feed compatible FLOAT ports, ScalarToInt can feed compatible INT ports, and ScalarCompare can feed BOOLEAN ports. Scalar helpers accept finite non-boolean numbers within ±1,000,000; integer helpers require actual bounded integers. Model-specific sampler/parameter constraints remain your responsibility; do not vary fixed-step distilled schedules blindly. Prompts and formatted results are bounded to 4096 UTF-8 bytes. Explicit rule-based prompt selection is not learned preference inference, and reference metadata alone never establishes that generation consumed the STRING.

## UPL v2 wire and pack contract

The same parser dispatches on `format`. V1 remains unchanged and never accepts an opaque embedding. V2's exact top-level keys are `format`, `synthetic`, `adult`, `factsSchema`, `facts`, `encoder`, `upl`, `context`, `recentIntent`. There is **no input audience hash**; `context.ingredientsHash` is producer provenance, not the audience hash. Use the complete JSON returned by the authorized synthetic-adult producer, not a partially populated illustration or an exposure/profile object.

| Wire field | Exact v2 requirement |
| --- | --- |
| `format`, `synthetic`, `adult` | `angelos.upl-input.v2`, literal `true`, literal `true`. |
| `factsSchema` | `angelos.facts.v2`, independently versioned from the encoder. |
| `facts.ageBand` | Required adult enum as in v1. |
| Optional facts | `region`: NA/EU/APAC/OTHER; `openness`/`energy`: finite non-boolean `[0,1]`; `interests`/`goals`/`preferredStyles`: lists of ≤16 unique strings; `preferredLanguage`: string. Each optional field may be omitted or null (unknown); ageBand may not. No other fact fields. |
| Fact strings and event topics | Nonblank, trimmed, no control characters or DEL, ≤80 UTF-16 code units. No automatic trimming, inference or truncation. |
| `encoder` | Exactly the seven fields in the JSON below; only dimensions 128, 256, 512, 1024. |
| `upl` | Exactly the declared number of signed finite non-boolean numbers in `[-1,1]`; absolute L2-norm error ≤`1e-4`. No four-dimensional fallback, clipping or renormalization. |
| `context.campaignId` | ASCII `[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}`, exactly equal to Campaign's ID when bound/exported. |
| `context.factRevision`, `eventRevision`, `consentVersion` | JS-safe integers; fact/event ≥0, consent ≥1. No latest-profile lookup or inferred authorization. |
| `context.asOf`, `ingredientsHash` | Nonnegative JS-safe epoch-millisecond integer; lowercase 64-hex SHA-256 string. |
| `recentIntent` | List of ≤64 objects with exactly `kind`, `topic`, `occurredAt`, `expiresAt`. Kind is view/click/save/dismiss; safe-integer epoch-millisecond times satisfy `0 <= occurredAt <= context.asOf < expiresAt`. |

Required encoder descriptor (choose a supported dimension):

```json
{"id":"qwen3-embedding-0.6b-v1","dimension":128,"modelId":"Qwen/Qwen3-Embedding-0.6B","revision":"97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3","pooling":"last-token","normalization":"l2","templateVersion":"angelos.upl-text.v1"}
```

The descriptor alone is not a parseable bundle: retain the producer's actual vector, full facts, context and recent-intent fields. This immutable revision was verified against the [official model metadata](https://huggingface.co/api/models/Qwen/Qwen3-Embedding-0.6B), with [pinned model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3/README.md), on 2026-10-08. The official model demonstrates last-token pooling and L2 normalization; the bridge validates the declared contract but never executes that model. Qwen text embeddings are not CLIP conditioning.

Raw v2 input and canonical wire serialization are bounded to 65,536 bytes; the existing duplicate-key, finite-number, nesting and plain-JSON checks still apply. Missing/extra schema or encoder fields, unsupported pins, dimensions, campaigns and expired/future events at the declared `asOf` fail explicitly. No wall-clock freshness or current ledger consent is asserted by offline import. The input cannot enable a download, supply a model path or request inference.

### V2 typed wiring

The existing fact nodes now expose explicit richer facts: list fields become comma-separated STRINGs. Connect, for example, `FactString(interests)` or `RecentIntentText` → `PromptTemplate.text` → `Campaign.reference_prompt` → `Campaign.prompt` → `CLIPTextEncode.text`. Numeric optional facts retain FLOAT output; mapping/comparison nodes can emit FLOAT, INT or BOOLEAN parameters. Missing facts fail extraction rather than use embedding coordinates as substitute traits. Recent-intent text preserves event order and emits `kind: topic` lines; empty event lists return an empty STRING. Its output is bounded to 32 KiB; the separate template/final-prompt 4096-byte limit still applies and fails rather than truncates oversized prompts.

`AngelosUPLEmbeddingBlock` emits only the custom `ANGELOS_EMBEDDING` socket, carrying the entire validated audience bundle with its encoder/context/hash. It does not emit a model tensor or Comfy `CONDITIONING` and cannot connect to a normal CLIP conditioning socket. This distinction follows Comfy's [custom datatype contract](https://docs.comfy.org/custom-nodes/backend/more_on_inputs). An explicit, separately validated model adapter would be needed for learned-vector conditioning; none is included. The opaque coordinate FLOAT node can read any actual coordinate, but coordinate names never become personality/interest traits.

### V2 persistence and verification limits

V2 uses these new version tags without changing v1:

- `angelos.comfyui-audience.v2`: all nine wire fields preserved, except `format` becomes this audience tag; adds `sha256`. No `preset` field or four-coordinate reduction.
- `angelos.comfyui-campaign.v2`: same campaign keys as v1, with the full v2 audience and its bound `context.campaignId`.
- `angelos.comfyui-teacher-pack.v2`: same top-level/image/provenance keys as the v1 table below, but requires matching audience/campaign v2. All original workflow and image bytes remain intact.
- `angelos.comfyui-import.v2`: same receipt fields as v1 plus `embeddingExecutionVerified: false` and `ingredientsHashVerified: false`.
- `angelos.review-template.v2`: same blank review-record fields as v1, with unreduced UPLs, plus a top-level `audience` containing the full v2 bundle. No generated quality/brand/click labels.

Audience `sha256` covers **the entire audience object except its hash**, using the existing Python canonical serialization below. This includes every fact, coordinate, encoder field, context revision/timestamp/ingredients hash and recent-intent entry. Campaign `bindingHash` includes that complete hashed audience; the import receipt additionally binds exact manifest bytes. The parser computes hashes itself after strict validation, so ordinary JS `JSON.stringify` is sufficient for wire input. Import revalidates using the same shipped standard-library `upl_v2.py` contract: it loads only that fixed checkout file, not a path or module from a pack, and does not import ComfyUI/PyTorch/Transformers. Keep the full checkout for the CLI and all four Python files for standalone plugin installation.

These checks prove consistency, **not that the supplied vector was produced from the supplied facts by Qwen**. The bridge does not have original ingredients/provenance or an encoder execution oracle; it validates the ingredients-hash format and preserves it, but cannot recompute or attest it. `ingredientsHash` binds the producer's original canonical JS ingredient bytes, **not** a Python reserialization; the main application verifies returned facts/context/recent intent against its captured ingredients before copying. That upstream check is not independently repeated or claimed by this offline bridge/importer. A malicious producer can recompute consistency hashes. Context/event changes remain visible in the full-bundle hash; this hash is still a configuration identifier, not a stable person/profile ID or proof of profile-disjoint training. Signed vectors never become synthetic CTR labels.

### Export files

File layout under `folder_paths.get_output_directory()`:

```text
angelos/aurora-<new uuid>/
  manifest.json
  workflow.json
  workflow-ui.json       (only when the client supplied it)
  image-000.png
  image-001.png          (when input batch > 1)
```

Directories are fresh, not reused. Workflow input never supplies an output path. V1 campaign IDs are ASCII identifiers beginning with a letter, at most 64 characters; only letters, digits, `_` and `-` follow. V2 accepts the bounded context ID described above. Directory slugs lowercase IDs, replace characters outside `[a-z0-9_-]` with `-`, and retain at most 64 characters; UUID identity avoids collisions. Versions are bounded ASCII version strings, never path components. Symlinked `output/angelos` is rejected. All writes use new files; ordinary write failures remove only the newly created incomplete pack. A process/power interruption may leave an incomplete directory: importers must require a complete valid manifest and verify every listed file.

### HD image gate

- Dense floating PyTorch tensor, NHWC `[batch,height,width,3]`, finite values strictly inside the inclusive range `[0,1]`. No integer, complex, RGBA, mask or latent substitution.
- Batch 1–8; each spatial side 1–4096; **total batch pixels ≤ 4096²**. Eight 1920×1088 images fit; eight 4096×4096 images do not.
- Normal exports require `(width ≥ 1920 and height ≥ 1080)` **or** `(width ≥ 1080 and height ≥ 1920)`. These are minimum-size constraints, not a required exact aspect ratio. Larger/square images qualify if they meet one orientation's bounds.
- No resize, crop, fit, normalization, clipping, procedural fallback or fake image replacement. Export preserves all input rows/columns. Native 1920×1088 remains 1920×1088; do the intended eight-row crop in the original workflow if delivery must be exactly 1080 rows.
- Conversion is RGB8: float32 values, `floor(value * 255 + 0.5)`, then uint8. This preserves geometry but quantizes the float raster. PNG compression is lossless relative to those RGB8 bytes. Source float tensors are not saved or mutated.
- `inputDevice` records where the input tensor resided. Copying pixels to CPU for PNG serialization is not a CPU renderer and does not prove which hardware produced them.

An upstream workflow may have loaded an existing file or upscaled a smaller image. The bridge cannot attest otherwise from an IMAGE socket: **HD size is not proof of native generation, trained weights, CUDA execution or image quality.** Validate the workflow's actual model/latent/decoder path independently. Preserve the original graph; the exporter does not compile it or turn it into campaign weights.

## Manifest schema: `angelos.comfyui-teacher-pack.v1`

This is the established parent-import contract. JSON names below are literal. All paths inside a pack are relative filenames; `manifest_path` returned to ComfyUI instead includes the `angelos/<directory>/` prefix.

| Field | Meaning |
| --- | --- |
| `format` | Exactly `angelos.comfyui-teacher-pack.v1`. |
| `kind` | `hd-teacher-pack` for normal node exports; `test-fixture` for explicit helper tests. |
| `sourceKind` | `comfyui-local-render` normally; `comfyui-test-fixture` for tests. Describes the export source, not verified upstream execution. |
| `status`, `testOnly` | `complete` and boolean fixture indicator. Importers must reject fixture packs from real training/evaluation evidence. |
| `packId`, `createdAt` | New UUID hex and UTC export timestamp. Neither is a training/model ID. |
| `campaign` | `format: angelos.comfyui-campaign.v1`, `id`, `version`, `referencePrompt`, full `audience`, `bindingHash`. |
| `audience` | Identical copy of `campaign.audience`, provided for easy dataset conversion. |
| `imageContract` | `inputLayout: NHWC`, tensor dtype/device strings, `pngMode: RGB`, `pngBitDepth: 8`, `spatialTransform: none`, quantization description, `hdSizeGatePassed`. |
| `images[]` | `index`, `file`, `width`, `height`, PNG-file `sha256`, decoded row-major interleaved RGB8 `pixelSha256`. One entry per actual input batch element. |
| `provenance.workflow` | `{file: workflow.json, sha256: <exact saved bytes hash>}`. Contains the entire supplied API graph, not only exporter ancestors. |
| `provenance.apiPromptSha256` | Hash of the API prompt using the bridge's JSON serialization below. |
| `provenance.uiWorkflow` | `{file: workflow-ui.json, sha256: ...}` or `null`. Preserves the UI graph separately for ComfyUI's workflow loader. |
| `provenance.exportNodeId`, `imageInput` | Hidden node ID and the declared `[upstreamNodeId, outputIndex]` IMAGE connection. This validates a graph reference, not the actual runtime ancestry. |
| `provenance.seedFields` | Observed top-level API-node input fields named `seed` or ending `_seed`, with node ID, field, value and encoding. Not a replay guarantee. |
| `provenance.checkpointReferences` | Recognized checkpoint/UNet/VAE/CLIP/LoRA loader fields, with node ID/class, original value and `verified: false`. Values are metadata only; never opened as files. |
| Verification flags | `modelHashesVerified`, `workflowExecutionVerified`, `upstreamNativeGenerationVerified`, `hardwareExecutionVerified`, `allRandomnessCaptured` are all false. The bridge does not independently verify any of these. |
| `training`, `promotion` | Both false. Neither action happens on export. |

### Audience encoding and bindings

Audience fields: `format: angelos.comfyui-audience.v1`, `synthetic: true`, `adult: true`, `preset`, `facts`, `factsSchema: angelos.facts.v1`, `encoder: {id: structured-demo-v1, dimension: 4}`, `upl`, `sha256`.

UPL order is `[ageBandIndex/5, regionIndex/3, openness, energy]`. Adult age ordering is `18-24`, `25-34`, `35-44`, `45-54`, `55-64`, `65+`; region ordering is `NA`, `EU`, `APAC`, `OTHER`. All four facts are required; openness/energy are numeric non-booleans in `[0,1]`. Custom facts accept no other fields. Presets use explicit fictional facts, not external catalogs or demographic inference. The exporter rebuilds and compares metadata, including hashes and derived UPL, rather than trusting a mutable Python dictionary from another node.

These are offline synthetic labels, not real profile identifiers, exposure snapshots, consent grants, advertising calls or impressions. An audience hash identifies this exact metadata configuration, **not a stable person/profile identity**; changing facts, encoder or preset can change it. Hash-disjoint splits alone cannot establish profile-disjoint evaluation across updates. The existing consent/attribution system must remain separate if another component later imports and serves creatives.

Audience `sha256` hashes the audience object without `sha256`; campaign `bindingHash` hashes the campaign object without `bindingHash`, including the complete hashed audience. The serializer is Python `json.dumps(ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)` encoded as UTF-8, with no trailing newline. This is **not an RFC 8785/JCS claim**; JavaScript numeric serialization can differ (`0.0` versus `0`). File hashes should always be verified over original bytes. Python importers can reuse this explicit serialization to verify metadata hashes; other languages must reproduce it or verify the persisted workflow/image files without silently claiming metadata-hash parity.

### Complete workflow provenance, with explicit limits

`workflow.json` is an envelope, **not** directly a frontend workflow:

```json
{
  "apiPrompt": {"...": "complete original ComfyUI API node map"},
  "extraPngInfo": {"workflow": {"...": "original frontend graph"}},
  "exportNodeId": "9"
}
```

The example above illustrates fields, not an executable template. Use a working model-specific ComfyUI workflow and add the export/conditioning nodes you need instead of trusting a static graph tied to an unknown model/node version. `PROMPT` contains submitted configuration, not every intermediate tensor or an execution trace. Cached, unexecuted or dynamically expanded nodes are not attested. All supplied API nodes and all JSON-compatible `EXTRA_PNGINFO` are retained, with no secret-field stripping or partial-graph substitution. `workflow-ui.json`, when supplied, is the ordinary UI object that can be reopened in ComfyUI. API clients without that object get an explicit `null` UI descriptor.

Metadata limits: full provenance envelope ≤2 MiB, maximum nesting depth 32, maximum 50,000 JSON values, maximum string length 65,536 UTF-8 bytes, object key 256 bytes, API node ID/class 128 bytes, API graph 1–1024 nodes. Campaign prompts are capped at 4096 UTF-8 bytes. Duplicate facts keys, nonfinite numbers, tensors and other non-JSON objects are rejected. Unsupported/oversized provenance fails rather than silently truncating the graph. The complete prompt must contain this export node and its IMAGE connection.

Seed integers are copied into `seedFields[].value` as decimal strings (`encoding: decimal-integer`) to avoid 64-bit seed loss in JavaScript; linked or other JSON values use `unresolved-json` and are not evaluated. The original workflow remains intact, including numeric literals. Seeds can miss hidden/custom randomness, batch offsets, scheduler behavior and hardware nondeterminism. A checkpoint name/path is not a content hash; the exporter neither loads nor hashes model files. No model hash, registry revision or CUDA execution claim is fabricated.

PNG files contain no prompt/workflow metadata chunks. Required sidecars are always written, independently of ComfyUI's normal PNG `disable_metadata` option; opting into this output node opts into sidecar provenance. Full workflows can expose local paths, prompts or secrets from other nodes: use synthetic inputs, inspect the pack before sharing, and never send a private workflow merely because the image is public.

## Import a pack for review

The local importer is implemented in `ml/angelos_ml/comfy.py`. From your Angelos checkout, with the existing `.venv/bin/python` environment and Pillow available, run:

```sh
mkdir -p artifacts
npm run comfy -- import /physical/ComfyUI/output/angelos/aurora-REPLACE_WITH_PACK_UUID --output artifacts/imported-aurora-001
```

Replace the source with the actual exported **directory**, not its manifest or a PNG. The output parent must already exist; the destination must be new and outside the source pack. The npm wrapper resolves relative output paths from the Angelos checkout. It does not install dependencies or launch ComfyUI.

**macOS/Linux physical paths only:** source and destination paths must contain no symlink components or `..` traversal. This rejects even benign aliases such as macOS `/tmp` and `/var`; use physical paths such as `/private/tmp` when applicable. Obtain an existing directory's physical path with `cd /your/directory && pwd -P`; for a new output, resolve its existing parent first and append the new directory name. If `artifacts` is a symlink, select another physical output parent. The importer requires POSIX descriptor APIs and fails explicitly on unsupported platforms; the standalone exporter can still be installed in ComfyUI on Windows.

The importer validates the entire pack before publishing: recognized v1/v2 format, complete status, matching audience/campaign versions, synthetic adult audience, facts/UPL/campaign bindings, workflow hashes, RGB8 dimensions and PNG-file/pixel hashes. V2 additionally preserves the full context/recent-intent/encoder bundle and rejects a mismatched campaign; v1 still rejects opaque encoders. It rejects unknown pack files, traversal, symlinks, hardlinks, nonregular files and existing output destinations. Test-fixture packs are rejected by default; `--allow-test-fixture` is solely for explicit offline fixtures and never turns them into real generation evidence.

A successful import creates a **new read-only pack** containing the unchanged exported files plus:

- `import.json`: the hash-bound import receipt; also printed to stdout. Verification remains consistency checking, not authenticity or model/hardware/quality attestation.
- `review-template.json`: an `unreviewed` draft with one record per PNG, bound image/campaign/UPL identifiers, and **null** `source`, `quality` and `brand` fields. It supplies no fabricated labels, click evidence or ledger entries.

Files use mode `0444` and the directory `0555`: a write-once application convention, not protection against the owner changing permissions. Copy the review template to a separate writable location before supplying actual human quality/brand review scores and their source; do not modify the evidence pack or insert the blank draft directly into an exposure/click ledger.

**Import never executes workflow JSON, opens checkpoint references, downloads weights, assigns scores, trains or promotes a model.** It preserves image bytes without resizing. Workflow execution stays in your independently trusted ComfyUI installation. This is an HD teacher/review pack, not a native-32 NPZ or campaign model; conversion into a campaign-training dataset and subsequent training remain separate future work.

Each PNG inherits the pack's campaign reference prompt and audience configuration; batch items are separate creatives, not separate profiles. A reference prompt proves conditioning only when the actual generation path uses it, not merely because it appears in metadata. Workflow hashes are provenance, not compiler IR hashes, verified model identities or replay guarantees. The review `profileId` groups exact synthetic audience configurations, not stable people or ledger identities. Changed facts change its hash, so configuration-disjoint splits cannot establish person/profile-disjoint evaluation across updates. Keep human review and CTR candidate optimization separate; importing is not an impression, click, review or promotion.

The bridge has no network client, endpoint registration, shell execution, arbitrary code evaluation, model registry lookup or cloud integration. It also **cannot sandbox ComfyUI or other custom nodes**: operators must trust their independently installed ComfyUI workflows. The safety boundary is not accepting arbitrary submitted workflows into Angelos for execution.

## Validation scope

`ml/tests/test_comfyui_bridge.py` loads the node package without ComfyUI and injects only a temporary `folder_paths` module. Tests exercise typed UPL parsing/extraction, version/dimension/semantic binding, JavaScript-style unhashed numbers, all age/region encoder combinations, safe template rejection, actual prompt selection, node registration, audience/prompt binding, native landscape/portrait/1088-row persistence, RGB rounding/hashes, complete workflow preservation, UI descriptors, fixed output containment, no overwrite, failure cleanup, metadata limits and malformed/nonfinite/oversized input rejection.

The helper `export_teacher_pack(..., test_only=True)` permits small fixture images solely for Python tests. The UI node has no such input or argument. The flag must be an actual bool; it never relaxes batch, pixel, metadata, synthetic-adult or range checks. Fixture folders and PNG filenames begin `TEST_ONLY-`; manifests use `kind: test-fixture`, `sourceKind: comfyui-test-fixture`, and `testOnly: true`, even for HD-sized test inputs. Tests of the normal UI path necessarily use supplied synthetic pixels; their passing proves contract behavior, not pretrained generation.

No live ComfyUI session, loaded model, native-generation quality, GPU success, training job or cloud execution is implied by these offline tests. For manual acceptance, restart your real ComfyUI, wire an already working local HD workflow, queue once, inspect the image preview and full pack, and reopen `workflow-ui.json` if present. Confirm dimensions and upstream generation provenance independently before calling the resulting content native high-quality 1080p.
