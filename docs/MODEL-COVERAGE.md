# Full creative-model capability audit

Research date: **2026-10-06**. This replaces the three-model list as the **planning scope**, not as the runtime allowlist. The existing [implemented adapters](MODELS.md) remain exactly what they were. No new weights, adapters or quality benchmarks are implied by this report.

## What was actually inspected

Read-only inspection of the user-provided local Rhizome language, generated model/parameter registries, gate IO, action definitions, catalog generator and evaluation cache. Only original functional descriptions and aggregate audit results are retained here; no private implementation, provider catalog, credentials, prices or former-employer assets are incorporated into Angelos.

| Local inventory | Result | Interpretation |
| --- | --- | --- |
| `lang/src/gen/models.ts` | 477 entries; 221 distinct display names | Includes modality variants, versions, provider aliases and utilities; not 477 separate model architectures. |
| `lang-ts/src/gen/models.ts` | 466 entries; 216 display names | Divergent snapshot, not the canonical complete inventory. No TS-only slugs in this comparison. |
| `eval/harness/models-cache.json` | 356 endpoints; 21 capability labels | A different inventory, **not** a subset of the language catalog. Ten endpoints across four display names occur in neither generated catalog. |
| Parameter definitions | 170 parameter keys | Important features are hidden here, not in the coarse model mode. |
| Language gate registry / app IO metadata | 18 gates / 26 node definitions | Includes non-model graph structure and app-only nodes. A catalog or IO declaration is not evidence of executable support. |
| Deterministic action registry | 38 actions | All accounted for in [operations research](RESEARCH-OPERATIONS.md). |
| Published 3D-output models in either generated catalog | 0 | The 3D gate exists, but the snapshot does not bind a working provider. Research covers the declared capability separately. |

The catalog generator prioritizes text whenever a model accepts text, and keeps only its first output modality. Consequently, “text-to-image” hides editing and reference-conditioned generation; “text-to-video” hides motion transfer, audio conditioning and video editing. Upscalers and LoRA utilities also occur under `noMode`/`other`. Building a registry from those mode labels alone would repeat the original mistake.

The endpoint cache does not prove every exposed capability executes in the local evaluation harness. In particular, its document path is not general PDF ingestion and its video endpoint selection does not establish full mixed/audio-conditioned coverage. This report targets the **declared creative capability surface**, not an assertion of working Rhizome service parity.

## Reading the research

| Area | Detailed alternatives and contracts |
| --- | --- |
| Raster/vector images, editing, reference conditioning, restoration, masks, LoRA | [Image research](RESEARCH-IMAGE.md) |
| Generation, temporal controls, video editing, reenactment, enhancement | [Video research](RESEARCH-VIDEO.md) |
| Speech, music, sound effects, ASR, audio understanding, avatars, lip sync | [Audio research](RESEARCH-AUDIO.md) |
| Language, vision/video understanding, documents, retrieval and 3D | [Text and 3D research](RESEARCH-TEXT-3D.md) |
| All 38 deterministic actions, asset nodes, routing, layers and export | [Operations research](RESEARCH-OPERATIONS.md) |

**Direct** means the upstream implementation accepts the required kind of conditioning and returns the relevant output. **Composed** means multiple explicit operations are needed. **Partial** means important controls, identity, consistency, fidelity or scope remain unmatched. **Gap** means no verified suitable replacement was established. None of these labels means “matches the proprietary model's quality.”

License descriptions are checkpoint-specific research notes, not legal clearance. Keep permissively licensed code/weights, custom restricted weights, noncommercial weights and unknown dependency terms distinct. An Apache license on inference code cannot relicense an attached checkpoint. Do not assume the right to train or distribute a distilled student follows merely from being able to download teacher weights.

## Capability coverage

### Recommended evaluation stack

These are my proposed defaults to evaluate, **not a measured ranking or an installation plan**. Every component still needs an immutable intake manifest and real-weight testing. The detailed reports include alternatives, dependency issues and controls not covered by these headlines.

| Job | First candidate(s) | Why this is a different capability |
| --- | --- | --- |
| Text planning / visual understanding | [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) / [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Text and multimodal understanding are not image generators. |
| Image generation / reference editing | [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) / [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | Generation, initial-image sampling and reference editing need separate modes. |
| Spatial image controls | [Qwen-Image ControlNet Union](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union) with its supported Qwen base | Edge/depth conditioning is not just another prompt. |
| Layer decomposition | [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) | Emits manipulable RGBA layers rather than a flattened image. |
| Vector generation / tracing | [OmniSVG 1.1 8B](https://huggingface.co/OmniSVG/OmniSVG1.1_8B) / [VTracer](https://github.com/visioncortex/vtracer) | Native vector generation differs from raster tracing; review training-data terms. |
| Video generation / masked reference editing | [Wan2.2](https://github.com/Wan-Video/Wan2.2) / [Wan2.1-VACE-14B](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B) | A generation checkpoint alone does not supply VACE's control/reference interface. |
| Image-conditioned joint audio/video | [MOVA-720p](https://huggingface.co/OpenMOSS-Team/MOVA-720p) | Generates synchronized modalities; a music mux is not equivalent. |
| Video restoration / frame interpolation | [SeedVR2-3B](https://huggingface.co/ByteDance-Seed/SeedVR2-3B) / [Practical-RIFE 4.25](https://github.com/hzwer/Practical-RIFE) | Spatial detail and temporal samples are independent operations. |
| Segmentation / background extraction | [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) / [SAM 2.1](https://github.com/facebookresearch/sam2) | Masks are not physical transparency mattes. |
| Speech / transcription / understanding | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | Three different input/output objectives, not a single generic audio model. |
| Music | [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) | Music composition is not speech or event-conditioned Foley. |
| Sound effects | [MOSS-SoundEffect v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0), compare [Dasheng-AudioGen](https://huggingface.co/mispeech/Dasheng-AudioGen) | Permissively declared text-to-SFX candidates; neither establishes direct synchronized video-to-audio equivalence. |
| Talking performance | [InfiniteTalk](https://huggingface.co/MeiGen-AI/InfiniteTalk) | Portrait/video + speech animation; inspect its entire dependency bundle. |
| 3D assets | [TRELLIS text/image](https://github.com/microsoft/TRELLIS), [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) | Mesh/material outputs need their own asset and rendering contracts. |

Do not force a permissive badge onto the harder categories. **Direct synchronized Foley, generic audio-driven scenes, strict local lip sync, native multishot, retakes, precise camera/identity controls and soft matting have restricted or partially matched candidates.** MOSS/Dasheng improve text-to-SFX coverage, but do not close direct Foley coverage. LTX, Hunyuan and specialist models remain useful research alternatives where their exact licenses permit the intended use; they are not interchangeable with unrestricted open-source weights. See the license registers before selecting them.

### All 21 cache categories are accounted for

Counts are capability-label occurrences in the 356-endpoint cache, not numbers of independent models; some endpoints carry several labels. Each category has a dedicated research destination. The detailed tables distinguish direct, composed and partial alternatives rather than hiding deficits under an overall “supported” flag.

| Cache category | Occurrences | Required distinction | Research |
| --- | ---: | --- | --- |
| Text → image | 51 | Raster generation, typography and vector output are different contracts. | [Image](RESEARCH-IMAGE.md) |
| Image → image | 51 | Editing, style, restoration and geometry control are not interchangeable. | [Image](RESEARCH-IMAGE.md) |
| Multiple images → image | 29 | Reference roles, count limits and identity preservation. | [Image](RESEARCH-IMAGE.md) |
| Inpainting | 3 | Explicit mask, protected region and boundary behavior. | [Image](RESEARCH-IMAGE.md) |
| Outpainting | 1 | Canvas expansion, position and preserved source pixels. | [Image](RESEARCH-IMAGE.md) |
| Image upscaling | 4 | Precision restoration versus hallucinated detail. | [Image](RESEARCH-IMAGE.md) |
| Text → video | 41 | Silent generation versus audiovisual and multi-shot generation. | [Video](RESEARCH-VIDEO.md) |
| Image → video | 44 | Initial-frame conditioning versus subject reference. | [Video](RESEARCH-VIDEO.md) |
| Video → video | 27 | Editing, reenactment, retake and enhancement. | [Video](RESEARCH-VIDEO.md) |
| Video upscaling | 2 | Spatial restoration versus temporal interpolation. | [Video](RESEARCH-VIDEO.md) |
| Frames → video | 12 | This bucket mixes reference ingredients and frame morphing; it cannot be treated uniformly as temporal keyframes. | [Video](RESEARCH-VIDEO.md) |
| First + last frame → video | 17 | Endpoint-constrained generation, not ordinary I2V. | [Video](RESEARCH-VIDEO.md) |
| Mixed inputs → video | 14 | Typed image/video references and controls, not an untyped list. | [Video](RESEARCH-VIDEO.md) |
| Text → audio | 4 | Separate speech, music and sound effects. | [Audio](RESEARCH-AUDIO.md) |
| Audio → text | 2 | Transcription versus general sound understanding. | [Audio](RESEARCH-AUDIO.md) |
| Audio + video → video | 4 | Lip sync versus broader reference-conditioned editing. | [Audio](RESEARCH-AUDIO.md), [video](RESEARCH-VIDEO.md) |
| Audio + image → video | 4 | Talking-head/body animation versus general audiovisual reference generation. | [Audio](RESEARCH-AUDIO.md) |
| Audio → video | 2 | Motion conditioning requires visual identity/scene assumptions to be explicit. | [Audio](RESEARCH-AUDIO.md) |
| Text → text | 16 | Copy, reasoning, structured output and research/tool use. | [Text](RESEARCH-TEXT-3D.md) |
| Image → text | 16 | Visual understanding, OCR and document-page reasoning. | [Text](RESEARCH-TEXT-3D.md) |
| Video → text | 16 | Timestamped temporal understanding; soundtrack is separate. | [Text](RESEARCH-TEXT-3D.md) |

### Capabilities that those 21 labels miss

The catalog names, gate contracts and parameters require these additional distinctions. They are not optional extras invented to expand a model list:

- **Images:** identity/character consistency, multi-subject/product/style references, Canny/depth guidance, explicit camera angles, layered RGBA decomposition, vector/SVG generation, raster vectorization, background/mask extraction, portrait/skin refinement, region selection and search-grounded editing/generation.
- **Video:** pose/motion transfer, performance reenactment and replacement, camera controls, temporal keyframes and morphing, continuation/extension, local retake, multi-shot planning, synchronized audio, background removal and frame-rate enhancement.
- **Audio/workflows:** multilingual speech, music and SFX; audio understanding, transcription/timestamps, lip sync and avatars; video-conditioned foley and styled subtitle rendering. Do not call a waveform mux operation learned audiovisual generation.
- **Training/assets:** LoRA training and separate validated adapter import/loading; uploaded sources, reusable references, layers, routers, collections and export effects.
- **Understanding/3D:** document ingestion, retrieval-grounded research, declared text/image/reference-to-3D gate support. A geometry model is not required merely because a VLM can discuss a picture in 3D terms.

See the area-specific tables for candidates and explicit gaps, and [the action ledger](RESEARCH-OPERATIONS.md) for all 38 deterministic actions.

Provider parameters still need a separate conformance layer: reference count/roles, seed, dimensions, duration, frame rate, prompt expansion, reasoning/token budgets, structured response formats, camera controls, region selection, audio retention/generation, codec/container and output alpha. Hosted moderation, web search and provider-side postprocessing do not automatically come with an open checkpoint. Unsupported controls must fail explicitly; similar parameter names are not evidence of identical behavior. Private/preview catalog labels are not assertions that the corresponding products are publicly available.

## Required architecture correction

The graph should declare **capabilities and typed ports**, then resolve a reviewed implementation. Do not add a giant model-name dropdown to `generate_image` and call it coverage.

1. **Versioned rich values:** text; RGB/RGBA image with color space and alpha semantics; mask; ordered image references with roles; timestamped frames; video plus optional audio; PCM audio with sample rate/channels; document pages; structured geometry/material bundle; adapter artifact. Preserve dimensions, duration and resource bounds.
2. **Explicit conditioning:** initial image, multiple subject/style/product references, mask, depth/edges/pose, first/last frames, temporal keyframes, source video, source audio and regional/time-interval controls. A model must reject unsupported conditioning rather than silently ignore it.
3. **Model manifest:** capability IDs, exact upstream revision and artifact digests, code/weight/dependency licenses, loader, parameter schema, supported backend/dtypes, preprocessing, native limits, resource policy, seed behavior and tested status. Research candidates must live outside the executable allowlist until these are resolved.
4. **Typed host effects:** asset decoding, tokenization, retrieval, document parsing, codec work and font shaping are explicit. Compile compatible tensor regions; never claim retrieval or arbitrary Python has become a CUDA kernel.
5. **Separate stochastic execution from training:** teacher graphs produce reproducible, provenance-bound datasets. Student distillation is a separate optimization process. Training/loading a LoRA is also separate from inference and from compiling the graph.
6. **Format-specific students:** use a pretrained image diffusion/flow backbone as the next serious image-student baseline, adapted to a campaign/UPL and then step-distilled if useful. A single 32px scratch U-Net cannot absorb high-quality image, audio, video, vector and 3D outputs. Video/audio students need their own architectures and measurements. Exact typography, logos, QR codes and structured assets may retain deployment modules.
7. **Honest browser contract:** WebGPU is required for the deployed campaign student, not proof that every offline teacher fits in a browser. Model export, operator support, memory, download size, quantization error and cold/warm latency require independent tests.

## Integration order and exit gates

This is an ordering of engineering work, not a claim of completed adapters:

1. **Asset and capability contracts first.** Add mask/reference/time-aware ports and validate mutually exclusive modes. Negative tests must reject silent input loss, incompatible shapes, unsupported outputs and arbitrary downloads/code.
2. **One strong image vertical.** Text planner + image generation + reference editing + masks + exact composition. Generate a useful-resolution licensed corpus with held-out identities/products; preserve native outputs. Do not spend another round optimizing a 32px demonstration while calling it ad quality.
3. **Image control and repair.** Geometry guidance, identity/style references, layers, background removal, precision/generative enhancement and SVG. Test brand/product/text correctness separately from aesthetics.
4. **Video and sound.** Timestamped video IO, endpoint/keyframe/reference distinction, editing and motion controls; then speech/music/foley, synchronized output and avatars. Verify temporal identity, boundary preservation, lip synchronization and audio drift.
5. **Documents, retrieval and 3D.** Permissioned ingestion/retrieval and safe structured assets. These cannot bypass consent or become arbitrary graph-directed network access.
6. **Student quality gates before acceleration.** Compare pretrained campaign adaptation, distilled generation and cached approved creative. Evaluate held-out campaign quality, conditioning and brand retention. Only then optimize steps/precision/export and measure actual browser performance.

Every adapter needs an offline contract test **and** a separately reported real-weight smoke test. A fake pipeline tests argument routing only. Record exact model/device/precision, peak memory, latency, sample outputs and failures. Promote nothing on the basis of a literature table.

## Current implementation boundary

Angelos presently has registered Tiny-SD, FLUX.2 Klein 4B and LTX-Video adapters, with live evidence only for the documented Tiny-SD path. Its graph does not yet expose the full conditioning supported even by those upstream models. The browser student is genuine diffusion but is a small, freshly trained 32×32 model whose reported UPL-conditioning audit failed; it is not a demonstrated personalized ad generator. See [measured execution](DIFFUSION-EXECUTION.md).

This audit changes the scope and proposed design. It does **not** silently turn research candidates into implemented support, weaken the runtime allowlist, import Rhizome code, or download dozens of checkpoints.
