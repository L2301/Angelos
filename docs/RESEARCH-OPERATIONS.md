# Non-model graph operations

Research checked 2026-10-06. A complete replacement graph needs more than pretrained models. These are proposed clean-room implementations, not claims that Angelos already supports them.

## Deterministic action coverage

The inspected language action registry declares 38 actions. The following table accounts for all 38 by function, without importing its private implementation. It describes implementation choices, not numerical parity with undocumented provider behavior.

Implementation building blocks:

- **Tensor:** PyTorch with Kornia geometry, color and filtering operators; Apache-2.0 Kornia code. Device-resident implementations can target CUDA; operator-specific export, precision and gradient behavior need testing. [Kornia](https://kornia.readthedocs.io/en/latest/), [license](https://github.com/kornia/kornia).
- **CUDA:** CV-CUDA provides a separate Apache-2.0 NVIDIA-specific operator library. Its layout/dtype/operator restrictions must be checked individually; do not imply that every operator exports through PyTorch or has gradients. [Operator contracts](https://cvcuda.github.io/CV-CUDA/modules/python/operators.html), [license](https://github.com/CVCUDA/CV-CUDA).
- **Media:** FFmpeg/libav for timing, containers and selected filters. Some filters have GPU variants; others require host work or explicit transfers. LGPL by default, with GPL/nonfree implications depending on the build and dependencies. [Filter reference](https://www.ffmpeg.org/ffmpeg-filters.html), [build licensing](https://ffmpeg.org/legal.html).
- **Typography:** HarfBuzz shaping plus a reviewed font rasterizer, font assets and GPU alpha compositing. HarfBuzz uses an MIT-style license; fonts and rasterizers need their own checks. [HarfBuzz](https://github.com/harfbuzz/harfbuzz).
- **QR:** Nayuki's MIT-licensed encoder followed by exact rasterization, not diffusion. [QR Code generator](https://github.com/nayuki/QR-Code-generator).

| # | Functional action | Proposed implementation and boundary |
| --- | --- | --- |
| 1 | Image color grading | Tensor color transforms / explicit LUT. |
| 2 | Video color grading | The same transform per frame, preserving timestamps. |
| 3 | Stitch videos | Media concatenation; normalize timeline and format. |
| 4 | Split video | Media trimming with declared boundary semantics. |
| 5 | Change image aspect ratio | Tensor crop/pad/resize; no hidden generative fill. |
| 6 | Ken Burns motion | Timestamped tensor affine camera path. |
| 7 | Extract video frames | Media decode with selected timestamps. |
| 8 | Split text | Bounded deterministic string operation. |
| 9 | Find and replace text | Bounded literal replacement, not arbitrary code. |
| 10 | Concatenate text | Explicit ordering and delimiter. |
| 11 | Rotate image | Tensor rotation with interpolation/border specification. |
| 12 | Flip image | Tensor index transform. |
| 13 | Image color filter | Declared tensor color transform. |
| 14 | Tint image | Tensor blend in a declared color space. |
| 15 | Select/filter a color | Tensor threshold/mask with defined tolerance. |
| 16 | Blur image | Tensor Gaussian or explicitly chosen kernel. |
| 17 | Duplicate image | Immutable asset reference or tensor copy. |
| 18 | Video frame grid | Frame selection and tensor tile composition. |
| 19 | Watermark video | Alpha compositing per frame. |
| 20 | Boomerang video | Media forward/reverse sequence, specified endpoint handling. |
| 21 | Reverse video | Frame/timestamp reversal; bound buffering. |
| 22 | Video long exposure | Tensor temporal accumulation, explicit weighting. |
| 23 | Video color filter | Per-frame tensor transform. |
| 24 | Video effect | Typed allowlisted effects; an unspecified generic effect is not a parity claim. |
| 25 | Speed up video | Media timestamp rescaling and audio policy. |
| 26 | Slow down video | Timestamp rescaling; learned interpolation is a separate node. |
| 27 | Duplicate video | Immutable reference or timestamped sequence copy. |
| 28 | Green-screen video | Tensor chroma key; not semantic segmentation. |
| 29 | Separate audio from video | Media demux. |
| 30 | Merge audio into video | Media mux, explicit offsets/mixing policy. |
| 31 | Side-by-side composition | Tensor placement plus timeline alignment. |
| 32 | Add shape to image | Exact tensor rasterization and alpha blending. |
| 33 | Generate shape image | Exact rasterization into an empty image. |
| 34 | Add text to image | Typography asset then tensor composite. |
| 35 | Generate text image | Typography into a transparent image. |
| 36 | Generate QR code | QR encoder and exact pixel module layout. |
| 37 | Resize video | Tensor resize or compatible media GPU filter. |
| 38 | Change video aspect ratio | Explicit crop/pad/resize and metadata. |

Subtitles combine timestamped ASR/alignment from [audio research](RESEARCH-AUDIO.md) with typography; transcoding and audio/video merging are media operations, even when exposed in a model menu. Precision interpolation and learned motion interpolation must have different semantics. A generic “video effect” or “color filter” requires parameter-level specification before equivalence can be tested.

## Non-generative node categories

| Category | Clean-room replacement |
| --- | --- |
| Uploaded image, video, audio and document sources | Validated, content-addressed assets with modality, shape, duration, provenance and access policy. No arbitrary graph-directed URL requests. |
| Reusable character/product/style/brand reference | An asset bundle plus optional descriptive text, not a separate foundation model. Preserve reference identity and permission. |
| Collections and text splitting | Typed lists, bounded fan-out, stable ordering, explicit seed derivation and execution budgets. |
| Router | Typed selection/control flow; never silently choose a different model. |
| Layers | Ordered alpha compositing with transforms, typography and explicit blend/color-space rules. |
| Reusable subgraphs/techniques | Versioned functions over typed ports. An upscale technique resolves to an explicit precision or generative model. |
| Export and publish | Permissioned effect outside the tensor graph, with explicit destinations and sanitized assets. |
| Result/source/UI-only nodes | Alias, result binding or editor metadata; comments, grouping and ghost placeholders are not models. |
| Arbitrary-code node present in app IO metadata | Deliberately **not supported** as arbitrary execution. Replace known operations with an allowlist; do not import a former-employer code executor. |

## GPU and distillation consequences

Compile device-compatible tensor regions; keep tokenization, IO, document parsing, font shaping, codec work and tool calls as explicit boundaries. GPU scheduling a graph is not turning that graph into one network's weights, and CUDA execution is not proof of differentiability. Discrete routing, QR generation and retrieval remain meaningful parts of a teacher even when gradients cannot pass through them.

Train the campaign student from teacher outputs separately. Evaluate exact logos, legal copy, QR decode success and layout constraints; retain a deployed deterministic compositor when the student cannot satisfy them. A stochastic model should never be the sole integrity mechanism for a QR destination or required text.

Acceptance tests need color-space/alpha fixtures, interpolation and crop conventions, multilingual glyph coverage, audio/video synchronization, variable frame rates, temporal boundary behavior, bounded memory and device-transfer traces. Only claim GPU support after profiling the actual path. None of these proposed library integrations was installed or benchmarked by this research pass.
