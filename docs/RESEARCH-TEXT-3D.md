# Text, understanding, documents and 3D alternatives

Research checked 2026-10-06. These are proposed teachers, not installed Angelos adapters. “Direct” means the published model supports the input/output task, not that its quality matches a proprietary service. Permissively licensed weights are identified separately from orchestration code and dependency licenses.

## Text and understanding

| Capability | Concrete alternative and coverage | License and important limits |
| --- | --- | --- |
| Copywriting, prompt expansion, translation, structured text generation | **Direct:** `Qwen/Qwen3-8B`; use its non-thinking path for ordinary copy. | Apache-2.0 weights. Validate JSON against a schema; generating plausible JSON is not a typed contract. No demonstrated parity with every proprietary text model. [Model card](https://huggingface.co/Qwen/Qwen3-8B). |
| Reasoning and planning | **Direct:** the same Qwen3 checkpoint with thinking enabled; keep reasoning budget separate from final-output budget. | Same license; planning output is data, never executable graph code. Benchmark campaign constraints rather than treating a model-family name as a guarantee. [Model card](https://huggingface.co/Qwen/Qwen3-8B). |
| Image captioning, visual Q&A, OCR, multiple-image understanding | **Direct:** `Qwen/Qwen3-VL-8B-Instruct`; explicitly supply images and text to a vision-language adapter. | Apache-2.0 weights. OCR and spatial reasoning are probabilistic, not exact logo/text validators. [Model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). |
| Video captioning and temporal understanding | **Direct:** Qwen3-VL with timestamped video frames. **Composed** if audio understanding is also requested: add an audio model rather than silently dropping the soundtrack. | Same checkpoint license. Sampling policy and maximum duration become part of the adapter contract; sparse frames can miss short events. [Model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). |
| Documents, page images, tables and document-grounded answers | **Composed:** Docling parsing, local page assets and Qwen3-VL/Qwen3 reasoning. | Docling code MIT; its optional models have separate licenses. A document reference is not a text string or arbitrary URL fetch. Preserve page and bounding-box citations. [Docling](https://github.com/docling-project/docling). |
| Deep research / web-enabled reasoning | **Composed, partial:** Qwen3 plus retrieval, source snapshots, citation verification and a restricted tool-calling coordinator. `smolagents.ToolCallingAgent` is one Apache-2.0 orchestration option. | A checkpoint alone does not replace a deep-research service. External retrieval is a separate permissioned effect. Do not enable its CodeAgent, shell tools or default unrestricted toolboxes. Search coverage, source licensing and factuality remain evaluation gaps. [smolagents](https://github.com/huggingface/smolagents). |
| Audio understanding and transcription | See [audio research](RESEARCH-AUDIO.md). | Transcribing words is not equivalent to understanding music, ambient sounds, emotion or simultaneous audio/video. |

### UPL encoding is a separate application capability

`Qwen/Qwen3-Embedding-0.6B` is an Apache-2.0 candidate for encoding a canonical, consent-filtered facts document. This is **not** an identified Rhizome inference category and must not inflate its coverage count. Text similarity is not evidence of useful personalization, calibrated personality dimensions, or fair demographic clustering. Keep facts, canonicalization version, encoder revision, pooling, dimension and normalization independently versioned. Do not expose cross-advertiser raw personal facts through a prompt by default. [Embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

## 3D

The inspected language declares a 3D gate with text and image-reference inputs, but its generated 477-entry model inventory contains no 3D-output checkpoint. Include this declared capability in the design; do not pretend the snapshot demonstrates a working 3D provider.

| Capability | Concrete alternative and coverage | License and important limits |
| --- | --- | --- |
| Text → 3D asset | **Direct:** `microsoft/TRELLIS-text-large` (1.1B), with smaller text-base or larger text-xlarge variants available. | MIT weights; output representation/export must be explicit. This is native text conditioning, not a text-to-image pipeline mislabeled native text-to-3D. [Checkpoint](https://huggingface.co/microsoft/TRELLIS-text-large), [official implementation](https://github.com/microsoft/TRELLIS). |
| Single image → 3D asset | **Direct:** `microsoft/TRELLIS-image-large`; evaluate TRELLIS.2 for higher-detail materials and geometry. | TRELLIS models and most code are MIT. TRELLIS.2 code/model are MIT with separately licensed rendering dependencies. [TRELLIS](https://github.com/microsoft/TRELLIS), [TRELLIS.2](https://github.com/microsoft/TRELLIS.2). |
| Multiple image references → 3D | **Partial:** TRELLIS-image multi-image conditioning. | Official implementation is a tuning-free inference method, not a separately trained multiview model. Conflicting views and geometry consistency require testing. [Upstream multi-image release note](https://github.com/microsoft/TRELLIS). |
| Text + image → textured/PBR asset | **Composed/partial:** image editing or generation followed by TRELLIS.2. | Not native arbitrary joint text/multiview conditioning. TRELLIS.2 requires Linux/NVIDIA CUDA; upstream specifies at least 24 GB GPU memory. It is not a demonstrated Mac or browser teacher. [TRELLIS.2](https://github.com/microsoft/TRELLIS.2). |

An asset has a typed representation: mesh, Gaussian representation, radiance field, or PBR mesh bundle. These are not interchangeable “3D URLs.” Geometry generation does not replace rigging, animation, accurate CAD, or product-dimension validation. A 3D teacher can render a supervised image/video target; the current raster student cannot output a mesh.

## Required adapter and evaluation work

1. Pin exact weight revisions, tokenizer/processor revisions, dependencies and licenses. No runtime downloads or remote Python execution from graph input.
2. Define text, bounded image lists, timestamped video, audio, document pages and mesh bundles as distinct ports. Advertise actual model capabilities rather than guessing from the primary modality.
3. Record truncation, sampling, token budgets, stop conditions, decoding settings, seeds and retrieved evidence in teacher provenance. Separate deterministic preprocessing from stochastic inference.
4. Test text schemas, citation-to-page correctness, OCR on owned brand assets, temporal event ordering and refusal to silently discard unsupported inputs. Test multimodal prompt injection and private-data leakage.
5. For 3D, validate finite vertices, face indices, materials, external-reference rejection and safe export; compare multi-view renders and geometry on held-out products. Reject active payloads in generated documents and assets.
6. Keep all these teachers on the offline teacher side until measured. Browser execution of a small campaign diffusion student does not imply browser execution of every source model.
