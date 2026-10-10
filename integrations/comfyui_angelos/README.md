# Angelos nodes for your existing ComfyUI

Use ComfyUI to generate the creative. These nodes parse synthetic adult UPLs into **typed prompt/parameter inputs**, bind campaign metadata, and **export the actual connected image**, without calling Angelos, loading models, training, or using cloud services.

## Install manually

Use your existing trusted ComfyUI installation and its Python environment (Python 3.11+, PyTorch, NumPy and Pillow). No Angelos server or separate model installation is required. Stop ComfyUI first and choose **one** option; replace the ComfyUI path with your actual installation. Do not overwrite an existing node installation.

macOS/Linux, symlink your checkout (replace both absolute paths):

```sh
ANGELOS_DIR="/absolute/path/to/Angelos"
COMFYUI_DIR="/absolute/path/to/ComfyUI"
test ! -e "$COMFYUI_DIR/custom_nodes/comfyui_angelos" && \
  test ! -L "$COMFYUI_DIR/custom_nodes/comfyui_angelos" && \
  ln -s "$ANGELOS_DIR/integrations/comfyui_angelos" "$COMFYUI_DIR/custom_nodes/comfyui_angelos"
```

Or copy this whole directory, including **all four Python files: `__init__.py`, `nodes.py`, `upl.py`, `upl_v2.py`**, into a **new** `ComfyUI/custom_nodes/comfyui_angelos` directory. Keep this README with the copy. On Windows use File Explorer; do not copy the entire Angelos repository. Restart ComfyUI, refresh the browser, and search Add Node for **Angelos**. No installation is performed by opening this README or by this task.

## Parse a UPL and actually condition generation

Add **Angelos Parse UPL JSON** and paste this unhashed wire envelope:

```json
{"format":"angelos.upl-input.v1","synthetic":true,"adult":true,"factsSchema":"angelos.facts.v1","facts":{"ageBand":"35-44","region":"NA","openness":0.25,"energy":0.2},"encoder":{"id":"structured-demo-v1","dimension":4},"upl":[0.4,0,0.25,0.2]}
```

This v1 wire accepts **only `structured-demo-v1`, dimension 4**, with exactly the four facts shown. Learned vectors must use the separate v2 wire below; arbitrary encoders are still rejected. All v1 coordinates must be finite numbers in `[0,1]` matching the declared facts. V1 JSON is limited to 16 KiB; missing/extra fields, duplicate keys, false/non-boolean synthetic/adult flags and supplied hashes are rejected. These declarations are for fictional adult test data, not a way to authorize real people.

**Prompt selection, step by step:**

1. Connect Parse UPL's `audience` to **Angelos UPL Fact → FLOAT**; select `energy`.
2. Connect `fact_value` to **Angelos Scalar → BOOLEAN (Compare)**, threshold `0.5`, operator `>=`.
3. Connect `condition` to **Angelos Choose Prompt**. Set `when_true` to your complete energetic fictional campaign prompt and `when_false` to your complete quiet prompt. These are your explicit rules, not inferred preferences.
4. Connect Choose Prompt's `prompt` to **Angelos Campaign**'s `reference_prompt`; also connect Parse UPL's `audience` to Campaign's `audience`. For editable widgets, right-click the node and convert the named widget to an input before connecting.
5. Connect **Campaign's `prompt` STRING to `CLIPTextEncode.text`**, then connect that encoder's conditioning into your existing model-compatible sampler path. Keep its CLIP connection and negative conditioning intact. Connect Campaign's `campaign` plus the final decoded IMAGE to Export Creative as described below.

With the example, energy `0.2` chooses `when_false`. Changing both `facts.energy` and `upl[3]` to `0.9` chooses `when_true`; changing only one fails validation. **Metadata alone does not condition the image.** The same actual STRING must feed generation; the export does not prove that the sampler used it.

**Prompt formatting:** connect **Angelos UPL Fact → String** (`region`) to **Angelos Safe Prompt Template**'s `text`, and Fact → Float (`energy`) to `value`. Template `A fictional Aurora campaign for region {text}, creative energy {value}` produces a STRING you wire through Campaign to CLIPTextEncode as above. Allowed placeholders are only `{text}`, `{value}`, `{integer}`, `{enabled}`; `{{` and `}}` escape braces. Attribute/index access, conversion flags, format specifications and code evaluation are forbidden. Placeholder values are never recursively interpreted.

**Typed parameter ports:** Fact → Float emits a real `FLOAT`; **Angelos Scalar → Int** maps an explicit bounded input range to an integer range with nearest/floor/ceil rounding. For example, `[0,1]` → `[4,8]` maps `0.25` to INT `5`. Connect only to a compatible INT input and choose bounds valid for your model; this example is **not** a recommended denoising-step range. A distilled model may require fixed steps. Scalar → Boolean emits Comfy `BOOLEAN`, usable by Choose Prompt or another matching Boolean input. Widget values can be converted to inputs to accept these wires.

**FLOAT range controls:** **Angelos Scalar → FLOAT (Explicit Mapping)** (`AngelosScalarMap`) maps a FLOAT with explicit input/output ranges and no clipping. Connect Fact → FLOAT to its `value`; `[0,1]` → `[2,7.5]` maps `0.5` to `4.75`, which can connect to a compatible CFG FLOAT input. For a strength input, set an appropriate output range such as `[0.2,0.8]`. Both ranges must be ordered (input strictly increasing); out-of-range inputs and nonfinite/bool values fail. Values and bounds are limited to ±1,000,000. These are wiring examples, not settings recommendations: use controls valid for the model, and keep fixed guidance fixed when required.

**Opaque coordinate access:** **Angelos UPL Coordinate → FLOAT (Opaque)** reads an index within the actual vector dimension (`0..3` for v1, up to `1023` for v2) as a FLOAT. It assigns no trait name and performs no latent-to-personality inference. Use explicit fact nodes for semantic facts.

For a UI Copy JSON button, use ordinary `JSON.stringify` on the seven-field envelope above; do **not** hash it in JavaScript. Python validates facts/encoder/vector semantics, then computes its canonical audience hash. Full Angelos profiles/snapshots and the parser's hashed canonical JSON output are not this input format: explicitly adapt only these fields from an already authorized synthetic fixture. Fact schema and encoder ID remain separate version bindings. This conversion never fetches a profile or creates consent.

The existing `angelos.comfyui-teacher-pack.v1` stays unchanged: parsed metadata uses the normal `ANGELOS_AUDIENCE` socket and canonical audience format with `preset: custom`. The pack retains the actual reference prompt, full facts, encoder, UPL and hash; full workflow provenance preserves the parser input and declared conditioning wires, not verified execution evidence.

## UPL v2: richer facts and opaque embeddings

The same **Angelos Parse UPL JSON** node accepts the exact `angelos.upl-input.v2` JSON copied from the synthetic-adult UPL producer. Do not relabel a v1 vector as v2 or hand-edit a generated embedding. Required top-level keys are exactly `format`, `synthetic`, `adult`, `factsSchema`, `facts`, `encoder`, `upl`, `context`, `recentIntent`; no caller-supplied audience hash. V2 uses `angelos.facts.v2`, requires an adult `ageBand`, and permits optional `region`, `openness`, `energy`, `interests`, `goals`, `preferredStyles`, `preferredLanguage`.

Only `qwen3-embedding-0.6b-v1`, model `Qwen/Qwen3-Embedding-0.6B`, revision **`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`**, pooling `last-token`, normalization `l2`, template `angelos.upl-text.v1`, and dimensions **128/256/512/1024** are accepted. Coordinates are signed finite numbers in `[-1,1]`, with L2 norm within `1e-4` of one. The bridge never clips, renormalizes, reduces to four coordinates, loads the encoder or downloads weights. The pin is a required declared identity, not evidence that the encoder ran.

V2 JSON is bounded to 64 KiB. Optional facts may be omitted or explicitly `null` (unknown); adult `ageBand` is always required and non-null. Lists of interests/goals/styles have at most 16 unique strings; fact text and event topics are trimmed, control-free and at most 80 UTF-16 code units. `context` requires `campaignId`, nonnegative safe-integer `factRevision`/`eventRevision`, positive safe-integer `consentVersion`, epoch-millisecond `asOf`, and lowercase 64-hex `ingredientsHash`. At most 64 `recentIntent` entries carry exactly `kind` (`view`, `click`, `save`, `dismiss`), `topic`, `occurredAt`, `expiresAt`; timestamps must satisfy `occurredAt <= asOf < expiresAt`. This is snapshot validity, not a live consent/expiry lookup. Facts/encoder/context/intent/vector are all covered by the computed audience hash. The original ingredients are not supplied, so the bridge preserves the producer's JS-byte `ingredientsHash` unchanged: it cannot independently recompute it or verify vector semantics.

**Usable conditioning paths:**

- Parse UPL → **UPL Fact → STRING** (`interests`, `goals`, `preferredStyles`, `preferredLanguage`, `region` or `ageBand`) → Safe Prompt Template → Campaign prompt → `CLIPTextEncode.text`. List facts are joined with `, `, not inferred from coordinates.
- Parse UPL → **UPL Recent Intent → STRING** → Safe Prompt Template's `text` → Campaign prompt → `CLIPTextEncode.text`. This emits explicit `kind: topic` lines; no event is silently dropped or scored. If a large result exceeds the template's 4096-byte prompt budget, formatting fails instead of truncating the evidence.
- Explicit numeric facts → FLOAT/INT/BOOLEAN helpers → compatible parameter/selection ports as above. Missing or null optional facts fail extraction; there are no synthetic defaults.
- Parse UPL → **UPL Opaque Embedding Bundle (Not CLIP)** emits `ANGELOS_EMBEDDING`: the complete validated audience bundle, **not `CONDITIONING`, CLIP tokens or a tensor compatible with arbitrary models**. Only a separately implemented matching consumer could use this socket. No learned embedding-to-model adapter is bundled. Individual coordinates remain opaque even if wired into explicit numeric rules.

Set Campaign's ID to **exactly** `context.campaignId`; mismatch is rejected. V2 automatically exports `angelos.comfyui-teacher-pack.v2` with campaign/audience v2, without changing v1 exports. The full context, recent intent, facts, encoder and unreduced vector survive export/import. Audience hashes identify configurations, not stable people; they do not prove profile-disjoint evaluation. The CLI importer handles both versions and retains v2's full audience in its blank review template. See the [v2 schema](../../docs/COMFYUI.md#upl-v2-wire-and-pack-contract) in the checkout; this link is optional in a standalone copy.

## Wire the workflow

1. Open a **known-working local workflow** with weights you already own and trust. Keep its model loader, text encoders, samplers, decoder and finishing nodes in ComfyUI.
2. Add **Angelos Audience (Synthetic Adults)**. Select `quiet`, `balanced`, or `bold`; keep `facts_json` as `{}` and `synthetic` true. For custom facts, select `custom` and use the example below. These labels are fictional test profiles, not inferred traits of real people.
3. Connect its `audience` output to **Angelos Campaign**. Set a campaign ID such as `aurora`, version `v1`, and your full positive reference prompt.
4. Connect Campaign's `prompt` STRING to the workflow's positive text encoder where compatible. For a text widget, use ComfyUI's conversion-to-input option if necessary. Keep the negative prompt separate. If your workflow uses another prompting system, configure it there and treat the reference prompt as a label—not proof that the sampler consumed it.
5. Connect Campaign's `campaign` output and the **final decoded/composited IMAGE** to **Angelos Export Creative (HD Teacher Pack)**. Use the final RGB IMAGE socket, not LATENT/MASK/MODEL or a preview filename.
6. Configure the original workflow to generate at least `1920×1080`, or portrait `1080×1920`. `1920×1088` is also accepted and remains 1088 rows; this exporter does **not** crop it. Each side is capped at 4096, batch size at 8, and the **whole batch** at 16,777,216 pixels.
7. Queue in **ComfyUI**. The exporter displays the saved images and returns a relative `manifest_path`. Each execution creates a fresh directory under ComfyUI's configured output directory: `angelos/aurora-<uuid>/`.

Custom fictional facts:

```json
{"ageBand":"35-44","region":"NA","openness":0.25,"energy":0.2}
```

Accepted ages: `18-24`, `25-34`, `35-44`, `45-54`, `55-64`, `65+`. Regions: `NA`, `EU`, `APAC`, `OTHER`. Openness/energy are finite numbers from 0 to 1. No names, email addresses or arbitrary personal fields are accepted. Setting `synthetic=false` is an error, not an authorization switch.

## What you get

- Unresized RGB PNGs, `manifest.json`, and a complete `workflow.json` provenance envelope.
- `workflow-ui.json` when ComfyUI supplied the frontend workflow; open this file in ComfyUI to inspect/reuse the graph. Otherwise only the API prompt is available in the envelope.
- Campaign/prompt/audience/UPL binding, file/content hashes and explicitly unverified seed/checkpoint references.

Copy the **entire pack directory** for future dataset conversion. This is an HD teacher pack, **not** a native-32 training artifact, a compressed campaign model, or proof of native high-resolution generation. An upstream upscale is still an upscale: the export node only proves it did not resize the tensor it received. Check your original latent dimensions/decoder and resize nodes yourself.

To validate and import a pack for review on macOS/Linux, use the existing Angelos checkout and its `.venv/bin/python` environment with Pillow:

```sh
mkdir -p artifacts
npm run comfy -- import /physical/ComfyUI/output/angelos/aurora-REPLACE_WITH_PACK_UUID --output artifacts/imported-aurora-001
```

Replace the source directory; the destination must not exist and its parent must already exist. Use physical source/output paths with **no symlink components or `..`**; even macOS `/tmp` and `/var` aliases are rejected. Resolve existing directories with `pwd -P`; relative output is based on the Angelos checkout. Unlike this portable ComfyUI plugin, the importer requires macOS/Linux POSIX APIs. It creates a new read-only copy plus `import.json` and a blank `review-template.json` with null source/quality/brand fields. Copy the template elsewhere before editing. Import does not execute workflows, supply labels, download models or train. Campaign-training dataset conversion remains separate future work; see the checkout's [import guide](../../docs/COMFYUI.md#import-a-pack-for-review).

The PNG is an RGB8 raster, not lossless float-tensor storage; conversion rounds to 8-bit without clipping invalid values. PNGs have no embedded workflow chunks. Provenance stays in sidecars even when ComfyUI's normal PNG metadata embedding is disabled. Inspect sidecars before sharing: full workflows can contain prompts, local model names/paths and other node settings.

ComfyUI workflows/custom nodes execute in **ComfyUI**, which you must trust independently. Angelos neither submits nor executes workflow JSON through this bridge. No API request, automatic trainer, downloader, recurring job or cloud provisioning is included. See the full contract in [`../../docs/COMFYUI.md`](../../docs/COMFYUI.md) in the Angelos checkout; that link may not exist in a copied standalone installation.

## Troubleshooting and offline validation

- **Nodes missing:** ensure `custom_nodes/comfyui_angelos/__init__.py` exists; inspect ComfyUI's import log and use its Python environment. Restart after installation.
- **HD gate:** fix upstream generation/crop dimensions. There is no UI fixture bypass and no automatic fit/upscale.
- **Invalid pixels:** resolve NaN, infinity, non-RGB or out-of-range values upstream; the bridge does not repair or replace images.
- **Missing PROMPT/unique ID:** queue inside ComfyUI, rather than calling the node's export method without hidden execution inputs.
- **No `workflow-ui.json`:** the client supplied no UI graph, common with API clients. Do not mistake API-format JSON for a frontend graph.
- **Unchanged export:** each queued output-node execution requests a fresh export; upstream ComfyUI nodes may still use their cache. A new pack does not establish new model inference.

From the Angelos checkout, run tests without ComfyUI or model weights:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=ml .venv/bin/python -m unittest discover -s ml/tests -p test_comfyui_bridge.py -v
```

The fixtures validate persistence and contracts, not visual quality or GPU execution. Official references reviewed before implementation: [node properties/V1 support](https://docs.comfy.org/custom-nodes/backend/server_overview), [hidden inputs](https://docs.comfy.org/custom-nodes/backend/more_on_inputs), [IMAGE tensors](https://docs.comfy.org/custom-nodes/backend/images_and_masks), and [core SaveImage implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py).
