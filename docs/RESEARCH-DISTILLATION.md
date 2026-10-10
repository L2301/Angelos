# Campaign-model distillation and compression: research and implementation decision

**Evidence cutoff: 2026-10-08. Research only; no training, inference, model-weight downloads, cloud provisioning, or production qualification performed.**

## 1. Decision first

**Keep the arbitrary-graph → complete-output dataset → campaign-specific learned generator architecture. Change the quality target and student initialization, not that separation.** The existing native-32 diffusion model is a valid execution/regression baseline, but it is not the final deliverable. The current target is genuinely impressive **1920×1080 generation**, or generation at **1920×1088 followed by an explicit eight-row crop**, not enlarging a 32×32 or 1024-class generated image.

Recommended implementation order:

1. **Continue the generic RL/candidate-selection path now.** Preserve a separately trained human-quality/brand critic, a separately calibrated CTR estimator, exposure-time attribution, consent, and candidate-policy probabilities. These interfaces should work with both the small regression baseline and future high-resolution generators. A successful synthetic RL test is not evidence of 1080p quality or real CTR lift.
2. **Qualify pretrained compact latent students before training anything from scratch.** Compare SANA/SANA-Sprint 0.6B and 1.6B against FLUX.2 Klein 4B on the same native-1080p campaign suite. Prefer the smallest candidate that passes, not the candidate with the best quoted one-step benchmark. SANA-Sprint is the leading compactness candidate; Klein 4B is the practical generation/editing comparator already represented in the local teacher registry. Neither is proven here to be the highest-quality choice for this campaign. See the [SANA-Sprint paper](https://arxiv.org/abs/2503.09641), [official SANA model zoo](https://github.com/NVlabs/Sana/blob/main/docs/model_zoo.md), and [official Klein release](https://github.com/black-forest-labs/flux2).
3. **Adapt a pretrained, preferably undistilled base with LoRA on native-resolution complete graph outputs.** Freeze the codec and initially the text encoder; learn campaign conditioning and transformer adapters. This is immediately compatible with rendered black-box supervision. LoRA lowers adaptation/storage overhead, not the base model's parameter count. The original [LoRA paper, 2021-06-17; ICLR 2022](https://arxiv.org/abs/2106.09685) and [official SANA LoRA training source](https://github.com/NVlabs/Sana/blob/main/train_scripts/train_dreambooth_lora_sana.py) support this starting point.
4. **Only then reduce denoising steps if necessary.** A campaign-adapted latent base provides a compatible score/velocity teacher for progressive/consistency or DMD-family distillation. Alternatively, test a released Sprint/few-step student directly, with its correct objective and scheduler. Preserve a 2–4-step quality option rather than making one step a product requirement.
5. **Quantize/export after quality and conditioning pass.** Retain browser inference as the deployment architecture. A pretrained DiT plus codec needs a new versioned browser runtime; it cannot be loaded into the current five-convolution artifact. CUDA qualification is a development stage, not proof of WebGPU deployment.

**Do not replace this plan with “DMD the arbitrary graph.”** Standard DMD needs compatible noised-distribution scores; the graph executor exposes final renders, not the score of the graph's composed output distribution. Sample-based latent fine-tuning, conditional feature-distribution matching, and explicitly coupled output regression remain available without such an oracle. A particularly relevant new alternative is [RDM/iRDM, 2026-07-02 preprint](https://arxiv.org/abs/2607.02375), but its large-batch research recipe is not automatically a cheap campaign-training solution.

The user has confirmed **no paid jobs**. This document does not connect accounts or authorize spending. Any future paid run would require new explicit authorization, including a total cap, GPU/time cap, storage/egress allowance, cancellation behavior, and an operator-approved dataset. No cloud job is required to finish the generic RL contracts.

## 2. What the current code actually establishes

Inspection on 2026-10-08 covered the working tree, not just historical plans. Other implementation work is concurrent; locations describe the inspected snapshot, not a promise that those files remain unchanged. Only this new document is owned by this review.

| Inspected component | Observed behavior | Consequence |
| --- | --- | --- |
| `ml/angelos_ml/dataset.py`, `build_dataset` | Executes `RasterTeacher` at native teacher resolution, including downstream deterministic operations/branding, then area-downsamples the complete raster to `[N,3,32,32]`. Defaults: 16 training UPLs, 4 held-out UPLs, 2 seeds each. Native PNG retention is optional. | Already whole-graph output learning, not node-by-node distillation. The current NPZ representation discards the detail needed for the new target. Retain/rebuild full-resolution targets rather than enlarging these arrays. |
| `ml/angelos_ml/train_diffusion.py`, `load_dataset` | Checks graph/encoder/model provenance, image shapes, profile-disjoint splits and diversified seeds. | Preserve these invariants in a separately versioned latent/high-resolution dataset contract. |
| `ml/angelos_ml/train_diffusion.py`, `train_diffusion` | Fresh initialization, sampled timestep/noise, epsilon MSE, Adam, finite-gradient checks, held-out improvement over untrained. No pretrained latent student in this path. | Valid conditional density-estimation baseline; not white-box score distillation and not high-resolution qualification. |
| `ml/angelos_ml/diffusion.py` | Native-32 U-Net with five convolutions; four UPL channels, time and coordinates; 64-level cosine training schedule; 16-step DDIM. | “32” is spatial resolution, not 32 inference steps. Step acceleration and larger output resolution are different projects. |
| `src/browser/diffusion.js`, `validateDiffusion` | Fixed architecture, shapes, epsilon semantics and schedule validation for `angelos.diffusion.v1`. | A scheduler swap, arbitrary channel increase, latent codec, or SANA checkpoint is not a compatible artifact update. |
| `ml/angelos_ml/providers.py`, `PipelineRegistry`; `ml/angelos_ml/models.py` | Allowlisted local pipelines and immutable teacher revisions; render-oriented interfaces. Klein is registered, not evidence that its real weights have run. | A new score/trajectory adapter must be explicit. Owning model files does not mean the current provider API supplies a score oracle. |
| `ml/angelos_ml/compile.py`, `compile_teacher` | Separate deterministic-graph compilation path. | Compilation preserves an executable computation; it does not learn a smaller probabilistic replacement for the whole graph. |
| `src/core/store.js`; `src/server.js` | Exposure qualification, frozen snapshot/artifact/creative identities, consent, delayed click labels and operator/synthetic review sources. The inspected exposure route uses `single-candidate-v1` with propensity 1. Reviews include quality and brand ratings. | Good attribution foundation. Existing deterministic single-candidate logs cannot identify the value of an arbitrary alternative selection policy. Preserve sources and add candidate support/probabilities in the RL implementation. |

The recorded [execution report](DIFFUSION-EXECUTION.md) describes 22,707 trained parameters, 40 teacher examples and a measured WebGPU denoising loop. It also reports **failed useful-conditioning evidence**: shuffled-UPL noise loss was slightly better than correct-UPL loss, and raw branding needed an exact compositor. These are documented prior measurements, not rerun results from this review. Improving noise MSE, pixel variation or browser parity does not establish attractive images, personalization or CTR improvement.

The current architecture remains useful as a tiny reproducible contract/parity fixture. It must not be presented as a nearly complete 1080p model that merely needs upscaling.

## 3. Separate four different meanings of compression

| Axis | What changes | What does not follow automatically |
| --- | --- | --- |
| Sampler acceleration | Fewer network evaluations, learned longer jumps, better numerical integration or cached computation. | Fewer parameters, smaller downloads, or lower peak activation memory. Count guidance passes and decoder cost, not just nominal steps. |
| Architectural compression | Fewer blocks/channels, a smaller student, cheaper attention or a more compressed latent representation. | Preserved quality, identical controls, or a faster browser implementation without suitable kernels. |
| Numerical/storage compression | Lower-bit weights/activations, compressed artifact storage, quantization-aware calibration. | Native low-bit execution. Dequantizing all weights to FP32 on load may save transfer bytes but not runtime memory/compute. |
| Whole-graph behavioral distillation | A learned generator approximates the distribution of final campaign outputs over a declared input domain. | Exact program equivalence, access to intermediate editing controls, all output modalities, or arbitrary out-of-distribution graph changes. |

LoRA is a fifth, complementary tool: **parameter-efficient adaptation**. A small adapter still needs its large base unless that base is separately compressed. Merging a LoRA removes separate adapter operations but does not reduce the base tensor dimensions.

The foundational [DDPM paper, 2020-06-19; NeurIPS 2020](https://arxiv.org/abs/2006.11239) supports training a denoiser from image samples; [DDIM, 2020-10-06; ICLR 2021](https://arxiv.org/abs/2010.02502) changes the sampling construction; [latent diffusion, 2021-12-20; CVPR 2022](https://arxiv.org/abs/2112.10752) moves the learned generative process into an autoencoder representation. These are distinct operations, not interchangeable labels for a single compression trick.

### Access levels determine which objectives are valid

Let `G(c, ξ)` be the entire campaign graph, `c` its supported inputs/UPL, and `ξ` all teacher randomness. Its target distribution is `p_G(x | c)`, after all declared composition and postprocessing.

| Available supervision | Valid starting points | Missing information |
| --- | --- | --- |
| Rendered outputs plus conditions | Conditional denoising/flow training, pretrained LoRA adaptation, conditional GAN/MMD-style matching, preference learning. | No exact graph score or denoising trajectory. |
| Reproducible paired generator inputs and final outputs | Function/endpoint regression, perceptual regression, deterministic-coupling datasets. | A numeric seed alone is not a shared continuous latent or a teacher trajectory. |
| A compatible trained denoiser/velocity field | Progressive distillation, consistency distillation, DMD/SiD/TDM, reflow through that model. | Its score describes that model's distribution, not automatically the original heterogeneous graph. |
| Compatible intermediate features | Feature-level KD or block-pruning recovery. | Cross-model feature alignment is not defined merely because both models generate images. |
| Human ratings, comparisons and matured clicks | Separate quality/brand and CTR models; contextual candidate selection; later generator post-training. | Sparse feedback is not a differentiable image score or unbiased evidence for unobserved candidates. |

Even if every graph node is differentiable, differentiating the renderer gives a **Jacobian**, not generally `∇x log p_G(x | c)`. Multiple random sources, generated text, model-specific codecs, frame extraction, compositing and non-invertible operations make a final graph score especially nontrivial. A last image node's denoiser does not become the score of the branded/composited output by assumption.

For an “arbitrary multimodal graph,” freeze the graph and assets per campaign and declare the **output contract** being learned. A raster student can learn a rendered view of text/video/3D-containing graphs; it does not thereby emit editable text, temporal video, audio, SVG topology or 3D geometry. Those require explicit typed students/heads and modality-specific datasets/losses. Preserve unsupported-node/type failures and exact external constraint enforcement.

## 4. Literature map: what is established and what transfers

Dates below are first arXiv submission dates unless a proceedings/release date is explicitly used. “Preprint record; venue not verified” means no acceptance claim is made here, not a claim that the work has never been accepted. Official repositories establish implementation availability, not successful reproduction or unrestricted licensing. Benchmark numbers across different datasets, resolutions, backbones and hardware are deliberately not used as a universal ranking.

### 4.1 Trajectory-preserving and consistency methods

| Method and dated primary source | Verified mechanism | Implication for Angelos |
| --- | --- | --- |
| **Progressive distillation**, 2022-02-01; ICLR 2022. [Paper](https://arxiv.org/abs/2202.00512), [proceedings PDF](https://openreview.net/pdf?id=TIdIXIpzhoI). | Train one deterministic student step to replace two teacher steps; repeat step halving. Parameterization matters at very low step counts. | Good controlled experiment once a useful campaign denoiser exists. Usually reduces steps, not backbone size. Not available from final renders alone without an additional coupling/teacher construction. |
| **Consistency Models**, 2023-03-02; ICML 2023. [Paper](https://arxiv.org/abs/2303.01469), [proceedings](https://proceedings.mlr.press/v202/song23a/song23a.pdf). | Enforce common endpoint predictions along a probability-flow trajectory; support distillation from a score teacher and standalone consistency training from data. | Important distinction: consistency *training* can use graph samples; consistency *distillation* needs appropriate teacher dynamics. One-step capability does not guarantee one-step campaign quality. |
| **LCM**, 2023-10-06; preprint record, venue not verified. [Paper](https://arxiv.org/abs/2310.04378), [official code](https://github.com/luosiallen/latent-consistency-model). | Distills guided probability-flow solutions in a pretrained latent model; practical few-step generation. | Reuse compatible latent model/codec/conditioning. An LCM adapter is not a compact independent browser generator and cannot attach to arbitrary architectures. |
| **CTM**, 2023-10-01; ICLR 2024. [Paper](https://arxiv.org/abs/2310.02279), [official Sony code](https://github.com/sony/ctm). | Learns maps between arbitrary times, incorporates score information, denoising and adversarial training. | Richer trajectory modeling and step flexibility, with more training machinery. CTM and TCD are different methods. |
| **TCD**, 2024-02-29; preprint record, venue not verified. [Paper](https://arxiv.org/abs/2402.19159), [official code](https://github.com/jabir-zheng/TCD). | Semi-linear trajectory consistency using exponential-integrator structure, plus controlled stochastic sampling. | Better-targeted few-step latent distillation, not graph-output score estimation or parameter pruning. Scheduler and compatible adapter/base are part of the method. |
| **SDXL-Lightning**, 2024-02-21; preprint record, venue not verified. [Paper](https://arxiv.org/abs/2402.13929). | Progressive plus adversarial distillation; releases full U-Net and LoRA variants. | Useful precedent for adapting a foundation model and then reducing steps. Full SDXL dependencies remain; step-specific checkpoints cannot be freely mixed with unrelated schedules. |
| **Hyper-SD**, 2024-04-21; NeurIPS 2024. [Paper](https://arxiv.org/abs/2404.13686), [author project](https://hyper-sd.github.io/). | Segmented trajectory consistency, human-feedback learning and score distillation; supports multiple low-step settings. | Strong evidence that distillation and preference learning are compatible. It does not replace actual campaign reviews/CTR or supply arbitrary graph scores. |
| **sCM**, 2024-10-14; ICLR 2025 oral. [Paper](https://arxiv.org/abs/2410.11081). | Stabilizes continuous-time consistency through parameterization, architecture and objective changes. | Relevant foundation for SANA-Sprint, but not a one-line epsilon-loss substitution in the native-32 trainer. |
| **LogCD**, CVPR 2026 proceedings. [Accepted paper](https://openaccess.thecvf.com/content/CVPR2026/papers/Xie_LogCD_Local-to-global_Consistency_Distillation_for_Few-step_Image_Generation_CVPR_2026_paper.pdf). | Local then global consistency distillation, with latent perceptual consistency; targets a unified 2–4-step model. | Recent support for prioritizing few-step quality over compulsory one-step sampling. Needs a compatible LDM/RFM teacher; deployment/code readiness must be separately checked. |

### 4.2 Distribution matching, adversarial objectives and sample-only alternatives

| Method and dated primary source | Verified mechanism | Access/caution |
| --- | --- | --- |
| **DMD**, 2023-11-30; CVPR 2024. [Paper](https://arxiv.org/abs/2311.18828), [accepted PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yin_One-step_Diffusion_with_Distribution_Matching_Distillation_CVPR_2024_paper.pdf). | Approximate reverse-KL gradient from target and generated-distribution scores, with paired regression regularization. | Requires a teacher score and a learned fake score. Does not require identical student/teacher architecture in principle, but compatible sample/noise/conditioning spaces are essential. |
| **DMD2**, 2024-05-23; NeurIPS 2024 oral. [Paper](https://arxiv.org/abs/2405.14867), [official training code](https://github.com/tianweiy/DMD2). | Removes DMD's paired-regression dataset requirement, stabilizes fake-score tracking with two-timescale updates, adds real-data adversarial training, and trains on inference-like multi-step states. | “No expensive paired dataset” is not “no teacher score,” “no real data in the published GAN recipe,” or “cheap on any GPU.” The repository declares a noncommercial share-alike license; review the exact code/checkpoint terms before reuse. |
| **SiD**, 2024-04-05. [Paper](https://arxiv.org/abs/2404.04057), [official code](https://github.com/mingyuanzhou/SiD). **Few-step SiD**, 2025-05-19. [Paper](https://arxiv.org/abs/2505.12674). Venue status not verified here. | Uses score identities for model-based score matching; later work matches a mixture of step outputs and studies guidance/diversity. | Data-free means no real/teacher-rendered corpus in that setting, not score-oracle-free. Useful later for an accessible campaign latent teacher. |
| **ADD**, 2023-11-28. [Paper](https://arxiv.org/abs/2311.17042). **LADD**, 2024-03-18. [Paper](https://arxiv.org/abs/2403.12015). Preprint records; venues not verified. | ADD combines score distillation and adversarial fidelity; LADD uses pretrained latent diffusion features for adversarial distillation. | A discriminator and a human preference critic have different jobs. The released-method training cost is additional to rendering a graph dataset. |
| **Diffusion2GAN**, 2024-05-09; ECCV 2024. [Paper](https://arxiv.org/abs/2405.05967), [author project](https://mingukkang.github.io/Diffusion2GAN/), [full paper](https://mingukkang.github.io/Diffusion2GAN/static/paper/diffusion2gan_arxiv.pdf). | Learns deterministic noise–output pairs using latent perceptual regression and conditional adversarial training. | Endpoint data is a valuable black-box-compatible idea. The published model also exploits pretrained diffusion components; generic graph adaptation is an inference, not a demonstrated arbitrary-DAG result. Save a real coupling, not just matching integer seeds. |
| **TDM**, 2025-03-09; ICCV 2025. [Paper](https://arxiv.org/abs/2503.06674), [accepted PDF](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Luo_Learning_Few-Step_Diffusion_Models_by_Trajectory_Distribution_Matching_ICCV_2025_paper.pdf). | Score-based trajectory distribution matching with sampling-step-aware targets, combining trajectory and distribution approaches. | Especially relevant for 2–4 steps; still not rendered-black-box-only distillation. TDM is not TCD. |
| **Decoupled DMD**, 2025-11-27; preprint record, venue not verified. [Paper](https://arxiv.org/abs/2511.22677). | Separates classifier-free-guidance augmentation from distribution-matching regularization; used by Z-Image's few-step approach. | Helps explain practical guided DMD behavior. A guided field should not casually be described as the exact score of the whole graph's empirical distribution. |
| **DP-DMD**, 2026-02-03; ICML 2026 per [official repository](https://github.com/Multimedia-Analytics-Laboratory/dpdmd). [Paper](https://arxiv.org/abs/2602.03139). | First step uses a teacher-derived prediction objective to preserve diversity; later steps use DMD. | Promising when a seed/candidate optimizer needs meaningful variation. Still teacher-field dependent. “No teacher-generated reference samples” does not remove this dependency. |
| **RDM/iRDM**, 2026-07-02; preprint. [Paper](https://arxiv.org/abs/2607.02375), [author project](https://alan-lanfeng.github.io/rdm/), [official code](https://github.com/vita-epfl/RDM). | Matches generated/reference features under frozen encoders using improved MMD estimation and a balanced multi-encoder objective; studies one-step Klein post-training. | The strongest newly identified fit to **sample-only** final-graph distribution matching. No online diffusion teacher score is required by that objective. However, large generated batches, encoder compute, conditional matching and independent quality checks matter; it is not a demonstrated low-budget 1080p campaign recipe. |

**Distribution matching is a family; DMD is a specific member.** It would be equally wrong to say that all distribution matching needs a teacher score or that DMD needs only rendered images. GAN/MMD/feature-distribution objectives can use samples; DMD/SiD use learned score fields. RDM's evidence also warns against trusting one feature encoder: low feature distance need not imply convincing images.

### 4.3 Flow matching, reflow and single-stage few-step generation

| Method and dated primary source | Mechanism | Recommendation |
| --- | --- | --- |
| **Rectified Flow**, 2022-09-07; ICLR 2023. [Paper](https://arxiv.org/abs/2209.03003). | Regresses velocities along noise–data interpolation; iterative reflow uses model-induced endpoint couplings to straighten transport. | Base flow training is sample-only compatible. Actual reflow needs a defined model-generated coupling. Merely drawing unrelated noise and rendered outputs is initial flow matching, not a reflow procedure. |
| **Flow Matching**, 2022-10-06; ICLR 2023. [Paper](https://arxiv.org/abs/2210.02747). | Simulation-free regression of conditional probability-path vector fields. | Natural objective for pretrained latent flow adaptation to final graph samples. Preserve each checkpoint's time direction, latent scaling and prediction conventions. |
| **InstaFlow**, 2023-09-12; ICLR 2024. [Paper](https://arxiv.org/abs/2309.06380). | Applies reflow to text-conditioned generation, then distills a more learnable noise-to-image mapping. | Foundational whole-generator acceleration, not evidence that reflow is free or reduces the backbone's parameter count. |
| **Shortcut Models**, 2024-10-16; ICLR 2025. [Paper](https://arxiv.org/abs/2410.12557), [accepted PDF](https://openreview.net/pdf?id=OlzB6LnXcS). | Adds step-size conditioning and self-consistency bootstrapping to learn short and long moves in one training phase. | Interesting future standalone student. It changes conditioning/training semantics and is not a drop-in pretrained SANA adapter. |
| **Inductive Moment Matching**, 2025-03-10; ICML 2025. [Paper](https://arxiv.org/abs/2503.07565), [proceedings](https://proceedings.mlr.press/v267/zhou25c.html), [official code](https://github.com/lumalabs/imm). | Distribution-level moment matching with inductive bootstrapping; trains one/few-step generators without the conventional pretrained teacher/student setup. | Valid research alternative on graph samples, but “can train from scratch” is not a reason to discard pretrained image knowledge for this small campaign dataset. |
| **MeanFlow**, 2025-05-19; NeurIPS 2025 oral per [author explanation](https://www.cs.cmu.edu/~csd-phd-blog/2026/mean-flows/). [Paper](https://arxiv.org/abs/2505.13447). | Learns average rather than instantaneous velocity using an identity involving derivatives/JVPs; no prerequisite distillation. | Important one-step generative modeling result, not automatically knowledge distillation. A new MeanFlow-from-scratch project is lower priority than qualifying pretrained 1080p students. |

### 4.4 Architecture, pruning, quantization and latent compression

| Method / artifact and source | What is actually compressed | Fit and limitations |
| --- | --- | --- |
| **BK-SDM**, 2023-05-25 first preprint; ECCV 2024 camera-ready. [Paper](https://arxiv.org/abs/2305.15798), [official repository](https://github.com/Nota-NetsPresso/BK-SDM). | Removes residual/attention blocks and recovers quality with output/feature distillation. Actual smaller U-Net, orthogonal to step reduction. | This is the verified name. No separately verified diffusion method called “BKSD” was identified; treat that spelling as ambiguous/shorthand, not a second established method. |
| **Tiny-SD / Segmind distill-sd**, 2023-era release lineage; model-card exact release day not established here. [Model card](https://huggingface.co/segmind/tiny-sd), [publisher's training repository](https://github.com/segmind/distill-sd). | Repository reports 323,384,964 parameters for its tiny U-Net versus 859,520,964 for the original. Uses denoising, teacher-output and block-feature losses; based on BK-SDM, with Realistic Vision V4 as teacher. | Not Nota's identical checkpoint, not the local 22,707-parameter student, and not “tiny” when codec/text encoder/download are counted. Valuable existing teacher/regression comparator, not the first choice for impressive native 1080p in 2026. |
| **Diff-Pruning**, 2023-05-18; NeurIPS 2023. [Paper](https://arxiv.org/abs/2305.10924), [accepted PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/35c1d69d23bb5dd6b9abcd68be005d5c-Paper-Conference.pdf), [official code](https://github.com/VainF/Diff-Pruning). | Structured network pruning with diffusion-timestep-aware importance and recovery training. | Needs weights/gradients and compatible execution shapes. Not the unrelated NLP “diff pruning” method. Pruning before proving campaign quality makes diagnosis harder. |
| **Q-Diffusion**, 2023-02-08; ICCV 2023. [Paper](https://arxiv.org/abs/2302.04304). | Timestep-aware post-training quantization and split treatment of skip-path activation distributions. | Calibrate across the actual campaign conditions and sampler timesteps. Generic LLM quantization assumptions need not transfer. |
| **SVDQuant**, 2024-11-07; ICLR 2025 spotlight. [Paper](https://arxiv.org/abs/2411.05007), [proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f34f0630c33be15b8c89426bb8056798-Abstract-Conference.html). | Uses a high-precision low-rank branch to absorb outliers while quantizing the residual; co-designs execution to avoid branch overhead. | Useful later for latent transformer deployment. Its optimized NVIDIA execution is not an existing WebGPU kernel. Count low-rank/scales/unquantized tensors in memory budgets. |
| **SANA**, 2024-10-14; ICLR 2025. [Paper](https://arxiv.org/abs/2410.10629), [accepted PDF](https://openreview.net/pdf?id=N8Oj1XhtYZ). | Efficient linear-attention DiT and 32× spatial-compression autoencoder; released 0.6B/1.6B-class backbones. | Particularly attractive for a compact high-resolution student. Latent compression and fewer pixels are not the same: a learned decoder produces the actual high-resolution raster. Decoder detail loss must still be evaluated. |
| **SANA-1.5**, 2025-01-30 first preprint. [Paper](https://arxiv.org/abs/2501.18427), [official model zoo](https://github.com/NVlabs/Sana/blob/main/docs/model_zoo.md). | Studies depth growth and compute scaling, including a larger 4.8B model. | A useful teacher/quality comparator, not inherently a smaller browser student than SANA 0.6B/1.6B. |
| **SANA-Sprint**, 2025-03-12; preprint record, venue not verified. [Paper](https://arxiv.org/abs/2503.09641), [official implementation guide](https://github.com/NVlabs/Sana/blob/main/docs/sana_sprint.md). | Converts a pretrained flow parameterization for continuous-time consistency distillation and combines sCM with latent adversarial distillation; 1–4-step operation. | Combines a compact backbone with step acceleration. The conversion is described as training-free, **not the complete distillation process**. Codec, conditioning and scheduler remain required. |

### 4.5 Preference/RL and video developments

| Method and dated primary source | Contribution | Relevance / boundary |
| --- | --- | --- |
| **DDPO**, 2023-05-22; ICLR 2024. [Paper](https://arxiv.org/abs/2305.13301), [author project](https://rl-diffusion.github.io/). | Treats stochastic denoising transitions as policy actions and optimizes terminal rewards. | A later generator-training route. The present deterministic DDIM replay does not provide ordinary nondegenerate transition log-probabilities for a copied PPO objective. |
| **Diffusion-DPO**, 2023-11-21; CVPR 2024. [Paper](https://arxiv.org/abs/2311.12908). | Adapts direct preference optimization to diffusion likelihood surrogates with winner/loser image pairs. | Useful if actual matched comparisons are collected. Independent 1–5 ratings or an unclicked impression are not automatically valid paired preferences. |
| **Self Forcing**, 2025-06-09; NeurIPS 2025 spotlight. [Paper](https://arxiv.org/abs/2506.08009), [official code](https://github.com/guandeh17/Self-Forcing). | Autoregressive video training on self-generated context; few-step rollouts, KV caching and truncated gradients permit video-level supervision. | Relevant to a future video student, not a 2D image upscaler. Its distribution-matching variants still need compatible video teachers; it does not unify arbitrary text/audio/video graph outputs by itself. |
| **DMDR**, 2025-11-17; revised 2026-07-03, ECCV 2026 acceptance reported by authors. [Paper](https://arxiv.org/abs/2511.13649). | Combines reward-tilted distribution matching and RL, using DMD as an alignment regularizer. | Supports retaining preference optimization after distillation. Assumes generative-training access; not a replacement for unbiased campaign outcome logging. |
| **TDM-R1**, 2026-03-08; preprint record, venue not verified. [Paper](https://arxiv.org/abs/2603.07700), [official repository](https://github.com/Luo-Yihong/TDM-R1). | Learns surrogate rewards for non-differentiable feedback and optimizes few-step generators along their trajectories. | Relevant to eventual human/CTR-driven generator updates; a learned surrogate can still be exploited and is not actual click evidence. |
| **RTDMD**, 2026-05-25; preprint. [Paper](https://arxiv.org/abs/2605.26108), [official repository](https://github.com/Harahan/RTDMD). | Ambient-consistent DMD followed by hybrid stochastic-step policy gradients and differentiable final-step reward optimization. | Recent concrete bridge between few-step distillation and RL. Requires its score models, stochastic process and reward assumptions, not just the existing render endpoint. |
| **Geometry-Aware Distillation**, 2026-06-01; ICML 2026 per paper. [Paper](https://arxiv.org/abs/2606.01651). | Aligns sensitivity to input noise using Jacobian-vector behavior. | Particularly relevant to seed/latent candidate optimization: visual quality alone can hide a flattened variation landscape. Needs meaningful continuous noise coordinates; finite differences over arbitrary hashed integer seeds are not this objective. |

For video, evaluate temporal consistency, motion, long-horizon drift, first-frame latency and memory—not just per-frame image preference. Extracting a frame from LTX into the current image dataset does not train a video generator. Image-specific 1080p feasibility must not be extrapolated into 1080p video feasibility.

## 5. Strongest practical student shortlist for the new target

### Ranking is conditional on measured campaign quality

| Candidate | Role in a concrete trial | Why not automatically the winner |
| --- | --- | --- |
| **SANA 0.6B → campaign LoRA → Sprint-style few-step student** | Compact deployment track. Begin from a pretrained base; compare released Sprint at 2 and 4 steps before funding custom distillation. | Limited scene/text fidelity may dominate latency gains. The released Sprint cards describe 1024-based multi-aspect training, not a demonstrated Angelos 1920×1088 result. |
| **SANA 1.6B / SANA-Sprint 1.6B** | Quality-first compact-track default if the 0.6B model fails. Test the same native output contract, then choose based on blinded review. | Larger download/working set. Higher quality than 0.6B on this campaign is a hypothesis, not a measured result. |
| **FLUX.2 Klein base 4B + LoRA; Klein distilled 4B comparator** | Immediately relevant open-weight flow family for generation and multi-reference editing. Base is preferable for broad adaptation; distilled model provides a ready four-step comparison. | Still 4B denoiser parameters plus codec and text encoder. Not directly browser-ready here, and a four-step distilled model should not be treated as an untouched long-trajectory score teacher. |
| **Z-Image-Turbo 6B** | Optional high-quality few-step reference if the approved budget allows a fourth family. | Larger than the compact track, not first browser target. Official card specifies eight NFEs; do not confuse scheduler-loop length with model evaluations. [Publisher card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), [official repository](https://github.com/Tongyi-MAI/Z-Image), [2025-11-27 report](https://arxiv.org/abs/2511.22699). |
| **Tiny-SD / BK-SDM / SDXL-Lightning** | Historical controls, cheaper experiments, or a measured fallback with explicitly lower quality. | No basis for declaring them the strongest current native-1080p student. A smaller low-resolution prior is not made competitive merely by accepting a large output-size argument. |

Klein's official release is dated **2026-01-15**; distinguish 4B Apache-2.0 variants from the differently licensed larger family members. The base lacks step/guidance distillation; the four-step variant is already distilled. Do not silently use a 9B checkpoint under the 4B admission decision. See [official release/model table](https://github.com/black-forest-labs/flux2), [base card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) and [distilled card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).

SANA cards declare Apache-2.0 for the model while also pointing to **Gemma text-encoder terms**. Admit each component, dataset and generated-training-output use separately; a pipeline-level badge does not establish every dependency's terms. [0.6B Sprint card](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers), [1.6B Sprint card](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers).

### Verified immutable model pins, not weight-execution claims

Public Hub API metadata and small JSON configuration files were read on 2026-10-08, without retrieving tensors. These snapshots predate the cutoff. The Klein distilled revision also matches the inspected local registry. **No pin below means the weights were downloaded, checked tensor-by-tensor, or run.**

| Proposed identity | Immutable revision | Primary snapshot |
| --- | --- | --- |
| `sana-base-600m-1024` | `28f3af7689de15f3883d5863059a2fca0aa9b829` | [Sana_600M_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers/tree/28f3af7689de15f3883d5863059a2fca0aa9b829) |
| `sana-base-1600m-1024-bf16` | `e2b3c0cbffebcd09d83805e88b9f5f106afc74ac` | [Sana_1600M_1024px_BF16_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers/tree/e2b3c0cbffebcd09d83805e88b9f5f106afc74ac) |
| `sana-sprint-600m` | `aa76e7f4f4928f378716b6716a2130fba3caf5b1` | [Sana_Sprint_0.6B_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/tree/aa76e7f4f4928f378716b6716a2130fba3caf5b1) |
| `sana-sprint-1600m` | `19683c58b7ea290e55cedd8950ae1d86ada7ef96` | [Sana_Sprint_1.6B_1024px_diffusers](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers/tree/19683c58b7ea290e55cedd8950ae1d86ada7ef96) |
| `flux2-klein-base-4b` | `a3b4f4849157f664bdbc776fd7453c2783562f4d` | [FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/tree/a3b4f4849157f664bdbc776fd7453c2783562f4d) |
| Existing `flux2-klein-4b` teacher | `e7b7dc27f91deacad38e78976d1f2b499d76a294` | [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294) |

A pinned repository is only the first level of the admission contract. Later record each loaded file's digest, tensor shape/dtype, code/library/container revisions, scheduler configuration, codec scale, conditioning preprocessing and adapter parent hash. Do not follow `main` at runtime or accept arbitrary graph-supplied repositories/code.

### Concrete SANA/Sprint architecture contract

The proposed first deployment family is **SANA linear DiT + frozen DC-AE decoder + campaign-conditioned embeddings/adapters**, not the old RGB U-Net. Verified Sprint 0.6B configuration: 28 transformer layers, 36 attention heads of dimension 32, 32 input/output latent channels, patch size 1, caption channels 2304 and guidance embeddings. Sprint 1.6B uses 20 layers and 70 attention heads of dimension 32. These are different complete model configurations; do not infer that the shallower model is smaller. [Pinned 0.6B config](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/blob/aa76e7f4f4928f378716b6716a2130fba3caf5b1/transformer/config.json), [pinned 1.6B config](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers/blob/19683c58b7ea290e55cedd8950ae1d86ada7ef96/transformer/config.json).

The inspected Sprint codec configuration is `AutoencoderDC`, 32 latent channels, 32× spatial reduction and scaling factor `0.41407`; its provenance names the **SANA 1.1** codec. The inspected ordinary 600M base names the **SANA 1.0** codec. Equal dimensions/scaling do not prove interchangeable latent coordinates. Use the exact matched codec when fine-tuning or distilling; moving between codec versions needs a validated conversion or re-encoding path. [Pinned Sprint codec](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/blob/aa76e7f4f4928f378716b6716a2130fba3caf5b1/vae/config.json), [pinned base codec](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers/blob/28f3af7689de15f3883d5863059a2fca0aa9b829/vae/config.json).

The base 600M scheduler specifies flow prediction, flow sigmas and a multistep DPM solver. The pinned Sprint index specifies **`SCMScheduler` with `prediction_type: trigflow` and `sigma_data: 0.5`**. Do not substitute the existing epsilon DDIM schedule or assume the original base loss remains correct for Sprint. Pin a library revision that actually implements this checkpoint contract. [Base scheduler](https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers/blob/28f3af7689de15f3883d5863059a2fca0aa9b829/scheduler/scheduler_config.json), [Sprint scheduler](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/blob/aa76e7f4f4928f378716b6716a2130fba3caf5b1/scheduler/scheduler_config.json).

**Critical native-resolution trap:** the inspected Diffusers Sprint pipeline enables `use_resolution_binning` by default. It can generate at a nearby 1024-class aspect bucket and resize the decoded image to the requested size. Therefore, `height=1088, width=1920` alone is not evidence of native generation. Disable binning for this qualification, audit all resizing, and capture actual denoiser/decoder shapes. [Official pipeline source](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/sana/pipeline_sana_sprint.py).

Proposed acceptance contract, with dimensions written width×height:

| Field | Required value / invariant |
| --- | --- |
| Native generative raster | `1920×1088`, not a lower-resolution raster resized after decoding. Direct `1920×1080` is acceptable only for a model whose actual spatial constraints permit it. |
| SANA latent layout | NCHW `[batch,32,34,60]`; patch size 1 gives 2,040 spatial tokens. This arithmetic follows the inspected codec/config, not an executed quality result. |
| Delivered raster | Center crop four rows from top and bottom: crop rectangle `(left=0, top=4, width=1920, height=1080)`. No geometric resampling. Store the crop in provenance. |
| Codec operations | Native latent decode is part of the generator, not an RGB upscale. If decoder tiling is needed, retain full latent resolution and test tile seams. |
| Conditions | Versioned campaign/UPL encoder, prompt preprocessing, token order/masks, asset/reference bindings and supported domain. Initially use real cached prompt embeddings for a finite prompt inventory; continuous UPL projection is an additional trained/validated component, not interpolation assumed equivalent to text encoding. |
| Sampling | Exact model-specific schedule, prediction type, time scaling, guidance embedding/value, noise generator and NFE count. Compare 2 and 4 Sprint steps; qualify the base separately. |
| Precision | GPU training/reference dtype is explicit. A BF16 checkpoint needs validated FP16/FP32 conversion or an explicitly supported browser representation; do not assume identical BF16 WebGPU execution. |
| Exact graphics | Preserve the learned full-graph image target, but keep audited brand/text/mask guarantees in a declared compositor when needed. Record raw and delivered quality separately; compose using crop-aware coordinates. |
| Deployment artifact | New format/backend capability ID; base/component and LoRA hashes, tensor manifest, runtime version, quality gates and numerical fixtures. Existing v1 remains a separate baseline. |

As rough **denoiser-weight arithmetic only**, 0.6B FP16 parameters are about 1.2 GB, 1.6B about 3.2 GB and 4B about 8 GB in decimal units. Four-bit packing alone would quarter these figures, before scales, high-precision branches, codec, text encoder, activations and runtime overhead. These are not measured VRAM requirements or mobile-browser promises. Precomputing campaign embeddings can remove the text encoder from a restricted serving path; it does not remove the image decoder or support arbitrary unseen prompts for free.

The Klein base config instead identifies `Flux2Transformer2DModel`, 5 dual-stream plus 20 single-stream layers, 24 heads of dimension 128 and a different packed latent/conditioning contract. Reusing SANA's latent shape logic for Klein would be invalid. [Pinned Klein transformer](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/blob/a3b4f4849157f664bdbc776fd7453c2783562f4d/transformer/config.json), [matched codec](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/blob/a3b4f4849157f664bdbc776fd7453c2783562f4d/vae/config.json).

## 6. Practical campaign-training sequence

### Phase A: qualify the prior, do not train the foundation

Use a bounded, approved GPU benchmark to compare the pretrained candidates at the actual target resolution and native aspect ratio. Test prompts representing the full graph's final content, not merely the last image node's prompt. Include products, subjects, fine textures, lighting, layout, multiple concepts, required typography and brand zones. A family that cannot produce convincing base images or reconstruct critical details through its codec should fail before adapter training.

The released Sprint pipeline is useful for this inference comparison even if the training track starts from undistilled SANA. Starting with a high-quality prior and a 2–4-step model is substantially more plausible than learning general image structure from a few dozen campaign images. This is an engineering recommendation, not a measured sample-efficiency guarantee.

### Phase B: collect complete high-resolution graph behavior

Keep profile-disjoint splits **before generation**, immutable graph/model/encoder provenance and node-level randomness. Retain native 1920×1088 targets, their delivered crops, captions/conditions and any required exact overlays; do not reuse the native-32 arrays as the only supervision. If a teacher cannot produce the target natively, label any resized/super-resolved training images honestly; they do not prove that the final generator meets the no-upscale requirement.

Use stratified coverage of UPL regions, graph branches and rare required layouts. Expand profile/seed counts through approved pilot stages, not an unpriced massive generation run. Preserve failures and rejected examples for audits rather than silently selecting favorable outputs. Hold out profiles and, where relevant, products/subjects/templates so the evaluation tests more than seed memorization. Near-duplicate assets and generated captions must not leak across splits.

**Information sufficiency matters:** if the graph uses changing text, references, documents or scene controls that are absent from the student condition, no loss can recover the missing conditional behavior. Either freeze those inputs into campaign identity, expose them through a typed condition, or explicitly marginalize them as stochastic variation. A new graph hash requires new compatibility evidence; it is not an input the present campaign model automatically understands.

### Phase C: pretrained LoRA adaptation on graph outputs

First train the undistilled compact latent backbone using its ordinary denoising/flow target on encoded final images. Freeze the codec/text encoder initially; compare modest adapter ranks such as 16 and 32 as **proposed pilot hyperparameters**, not upstream recommendations. Verify exact target module names from the admitted architecture rather than copying SD U-Net adapter names onto a DiT.

Retain useful general priors through a modest replay/regularization set if rights and budget permit. Compare campaign adapter against the same base without adaptation, with the same evaluation prompts and seeds. Record whether the student learns UPL/layout behavior rather than merely the campaign's average visual style. Copy/brand precision remains a separate hard gate.

Applying ordinary flow-matching LoRA training directly to an already few-step consistency checkpoint can destroy its few-step behavior. Prefer **base adaptation → qualified campaign teacher → step distillation**, or use the released few-step model's correct post-training recipe and evaluate it explicitly. A base LoRA is not automatically compatible with a separately distilled checkpoint, especially where codec, normalization or guidance structure differs.

For “one model per campaign,” either publish a merged campaign checkpoint or publish an immutable shared-base plus campaign-adapter composition with a single logical artifact identity. The latter can amortize browser download/cache cost, but a tiny adapter file is not a standalone tiny generator. Report first-campaign total bytes and incremental campaign bytes separately. Compare total offline dataset/adaptation/distillation cost against expected serving savings before funding another compression stage; multiplying candidates per decision also multiplies rendering work unless a measured batching/cache strategy avoids it.

### Phase D: optional step distillation with a real oracle

Once the campaign base is useful, it supplies a same-space teacher for consistency/progressive distillation or DMD2/TDM. For risk control, first compare a moderate step reduction and retain a higher-step reference. This new teacher only approximates the original graph; the final student must still be evaluated against held-out complete graph targets and independent human criteria. The chain introduces both graph-to-base and base-to-fast-student errors.

If rendered outputs remain the only permissible teacher access, keep sample-based adaptation. Consider conditional RDM-style or adversarial regularization as a separate budgeted experiment rather than claiming DMD2 has become black-box. A learned graph-score surrogate is possible, but training/validating it is an additional stage, not an oracle obtained from a renderer API.

### Phase E: browser qualification

Keep graph compilation, teacher execution, training, evaluation and deployment separate. Add a capability-negotiated latent-model artifact/runtime rather than weakening existing validation. Test transformer operators, conditioning, codec, scheduler, FP16 behavior, quantization kernels, allocation limits, cancellation and device-loss handling.

CUDA/MPS outputs used for development should be visibly labeled with backend and actual resolution. CPU scheduling/tokenization/reference calculations are not GPU generation. A backend that cannot run the full required model must fail clearly; no server-rendered image should masquerade as browser inference. Preserving the browser architecture means developing and proving its new model backend, not asserting that every compact checkpoint already fits it.

## 7. Sourceable objectives and optional adaptive losses

The equations below are design guidance, **not implemented changes**. `sg` denotes stop-gradient. Weights, timestep distributions and normalizations must be recorded; a superficially similar MSE can optimize a different target under different parameterizations.

### 7.1 Sample-only denoising or flow adaptation: the default

For full graph output `x`, encoded condition `c`, noise `ε`, and a diffusion latent `z0 = E(x)`:

`zt = αt z0 + σt ε`

`Ldenoise = E[w(t) || εθ(zt,t,c) − ε ||²]`.

This is valid with rendered outputs alone. The local baseline uses the corresponding RGB epsilon loss. A pretrained latent model contributes a learned image prior, not a new requirement for graph score access. See [DDPM](https://arxiv.org/abs/2006.11239) and [latent diffusion](https://arxiv.org/abs/2112.10752).

For a simple noise-to-data linear flow convention:

`zt = (1−t) ε + t z0`, with `t ∈ [0,1]`;

`Lflow = E[w(t) || vθ(zt,t,c) − (z0−ε) ||²]`.

This is the sample-based construction from [Flow Matching](https://arxiv.org/abs/2210.02747) / [Rectified Flow](https://arxiv.org/abs/2209.03003), not an assertion that every admitted checkpoint uses this time direction. For a data-to-noise convention the sign/trajectory changes. SANA-Sprint's trigflow/consistency contract must be handled by its actual training parameterization rather than this generic illustration.

For the **existing epsilon baseline only**, an optional Min-SNR ablation uses `SNRt = αt²/σt²` and `wε(t) = min(SNRt, γ)/SNRt`, with, for example, `γ=5` as a pilot. This formula is parameterization-specific; do not copy it unchanged into flow/consistency training. It is a training-balance experiment, not a high-resolution quality solution. [Min-SNR, 2023-03-16; ICCV 2023](https://arxiv.org/abs/2303.09556), [official implementation](https://github.com/TiankaiHang/Min-SNR-Diffusion-Training).

### 7.2 White-box output/feature KD when shapes and semantics align

For compatible teacher/student prediction targets `hT` and `hS` at the **same** noisy sample, time and condition:

`LKD = E[λdata ||hS−target||² + λout ||hS−sg(hT)||² + Σℓ λℓ ||Pℓ fS,ℓ−sg(fT,m(ℓ))||²]`.

`Pℓ` is an explicit learned/fixed feature projection and `m(ℓ)` an explicit layer correspondence; omit the feature term if neither is justified. This is the output/feature-distillation family exemplified by [BK-SDM](https://arxiv.org/abs/2305.15798) and [Segmind's implementation](https://github.com/segmind/distill-sd), not a feature-matching objective available on arbitrary rendered graphs. Choose epsilon, clean-latent or velocity targets consistently.

### 7.3 DMD's actual score requirement

For a differentiable generator `x = gθ(z,c)` and noising `xt = αt x + σt ε`, an idealized noised-distribution objective is:

`LDM = Et[wt KL(pθ,t(·|c) || pT,t(·|c))]`.

Under the score identity, its generator gradient takes the form:

`∇θ LDM = E[wt (sθ(xt,t,c) − sT(xt,t,c))ᵀ αt ∂gθ(z,c)/∂θ]`.

In practice, the fake score `sθ` is estimated by a separately trained model and the target score by a compatible teacher. Appropriate detach/weighting conventions are required; this is a gradient identity, not instructions to backpropagate an arbitrary KL scalar through both score estimators. [DMD paper](https://arxiv.org/abs/2311.18828), [DMD2 paper](https://arxiv.org/abs/2405.14867).

The DMD fake-score network is **not** the human-quality critic and **not** a CTR predictor. Replacing `sT` with a CLIP scalar, click probability, rendered PNG or the Jacobian of a compositing graph does not implement this objective. Reward terms can be combined with valid distillation, but their gradients and statistical meanings stay distinct.

### 7.4 Sample-only conditional distribution matching

Let `a = Φ(gθ(z,c))`, `b = Φ(xG)` for a frozen image encoder and graph sample `xG` with matching condition support. A kernel objective can use:

`MMD² = E[k(a,a′)] + E[k(b,b′)] − 2E[k(a,b)]`.

Average across several appropriate feature spaces and evaluate with independent encoders/human review. This is a sourceable alternative to score-based DMD; [RDM/iRDM](https://arxiv.org/abs/2607.02375) is direct recent evidence. Do not describe a small-batch, single-encoder implementation as a faithful reproduction of its large-batch recipe.

**Match conditionally.** Matching the unconditional image marginal can succeed while the model ignores UPL. Use matched discrete condition strata or a justified joint condition–image kernel, with sufficient samples. Finite frozen features can miss text, identity or brand violations. Keep hard constraints and a distribution/diversity audit outside the objective.

### 7.5 Optional adaptive campaign weighting—not a new named distillation method

A conservative experiment can combine coverage-preserving training with bounded quality weights. For an offline frozen critic and example `i`, define:

`qi = a + (b−a) sigmoid((rhuman,i−m)/τ)`, with `0 < a ≤ b`;

`q̃i = sg(qi / mean_training_batch(q))`;

`Ladapt = E[q̃i w(t) ℓbase,i] + λanchor Lreference + λconstraint Lconstraint`.

Use stratified sampling and nonzero minimum weight so rare conditions are not erased. Freeze `m`, `τ`, clipping limits and the critic version per candidate run. `Lreference` may be replay denoising/flow or a justified compatible-teacher KD term; do not invent it from an unavailable score oracle. `Lconstraint` requires actual differentiable constraint measurements, otherwise enforce the constraint outside the loss.

This bounded weighting is **an Angelos proposal**, inspired by reward-weighted adaptation, not a theorem or exact DMD/Min-SNR objective. It intentionally changes the learned data distribution. Compare against unweighted training and report diversity/conditioning regressions. Never turn held-out scores into training weights.

Do not directly weight diffusion examples by raw observed CTR from tiny samples. Mature outcomes first; use a separate calibrated, uncertainty-aware CTR estimator and causal/off-policy checks. Strong CTR weighting can amplify selection bias and reward hacking. Direct generator RL should be a later controlled experiment once the candidate policy and independent quality gates work.

For deterministic endpoint regression, preserve a deliberate shared coupling `(continuous input noise, c, final output)` and consider a latent/perceptual loss as in [Diffusion2GAN](https://mingukkang.github.io/Diffusion2GAN/). Integer seed equality across different pipelines is not that coupling. For sensitivity preservation, [Geometry-Aware Distillation](https://arxiv.org/abs/2606.01651) suggests measuring teacher/student responses to the same continuous noise perturbations; this is optional, costly, and inapplicable to unrelated hashed-seed changes.

## 8. Retain the human critic + CTR optimization loop

### First implementation: contextual candidate selection, not denoiser PPO

Define the action as a member of a finite, explicitly recorded candidate set: an admitted artifact/adapter, creative seed, declared sampler setting and optional approved creative controls. Each candidate is generated by a real, versioned model and checked for brand/safety/quality before selection. This makes policy probabilities tractable without differentiating through an arbitrary teacher graph or assuming the deterministic browser sampler exposes transition densities.

Train two separate predictive targets:

- **Human critic:** predict quality and brand compliance from delivered pixels plus campaign context and legitimate conditioning. Existing ordinal reviews support ordinal/regression/classification objectives. Use pairwise `−log sigmoid(rwinner−rloser)` only after genuine matched winner/loser labels exist; keep reviewer/source identity, uncertainty and held-out reviewers/items.
- **CTR head:** binary cross-entropy or another calibrated probability objective on consent-eligible, exposure-qualified, matured outcomes. Bind labels to the actual delivered creative, exposure-time UPL/model/encoder, placement and policy—not the latest profile. Synthetic clicks test plumbing only.

A proposed policy objective is:

`maxψ Ec Ea~πψ(·|c,A) [ R(c,a) ] − β Ec KL(πψ(·|c,A) || πref(·|c,A))`,

subject to explicit brand/quality constraints and candidate support. `R` can combine calibrated predicted click value and human-quality preferences with separately versioned scales; an uncalibrated weighted sum of raw logits and star ratings is not meaningful. Prefer a constrained CTR objective with a quality floor to “maximize clicks at any aesthetic cost.”

For initial stochastic selection, a softmax policy with a declared exploration mixture is a reasonable engineering choice. Log the **actual chosen-action probability conditional on the realized candidate set**, along with how that set was generated and filtered. Ranking scores are not propensities. If the candidate set itself changes, old logged selection probabilities alone do not identify a new set-generation policy.

Required records: action-set contents/digest, allowed candidate features, generation/filtering versions, selected index, policy and propensity versions, nonzero selected-action probability, artifact/adapter/component hashes, crop/compositor, creative seed, exposure snapshot, consent versions and label-maturity policy. Caching a generated image must not erase its condition/seed/model binding.

### Evaluation before promotion

For supported actions and known logging probability `μi`, a doubly robust estimator has the familiar form:

`VDR = mean_i [ Σa π(a|ci,Ai) r̂(ci,a) + π(ai|ci,Ai)/μi · (yi−r̂(ci,ai)) ]`.

Use held-out/cross-fitted reward predictions, inspect weight tails/effective sample size and uncertainty, and clearly disclose clipping bias if weights are clipped. The prerequisites include overlap/support and correctly attributed observations. The original [Doubly Robust Policy Evaluation and Learning, 2011-03-23; ICML 2011](https://arxiv.org/abs/1103.4601) explains why combining reward and behavior-policy models can improve evaluation; it does not rescue absent action support.

The inspected `propensity=1` single-candidate path supplies no evidence about unexposed alternatives. Do not infer uplift by scoring a new policy with the same CTR model that selected it. Before real traffic is authorized, report synthetic/offline capability only; actual lift requires an approved prospective experiment and meaningful confidence intervals.

### Later: generator updates, still retaining both critics

Keep replay/teacher fidelity and independent quality checks while optimizing the generator. Options include preference-pair training, reward-weighted LoRA, differentiable surrogate reward optimization or correctly formulated stochastic diffusion/flow RL. [DDPO](https://arxiv.org/abs/2305.13301), [Diffusion-DPO](https://arxiv.org/abs/2311.12908), [DMDR](https://arxiv.org/abs/2511.13649), [TDM-R1](https://arxiv.org/abs/2603.07700) and [RTDMD](https://arxiv.org/abs/2605.26108) provide different concrete precedents—not interchangeable losses.

Do not optimize a discontinuous real click as if it were a differentiable image reward. A learned CTR surrogate permits gradients but can be exploited. Keep frozen audit critics, periodic fresh human evaluation, rollback and caps on distribution shift. The human reviewer remains valuable after distillation; compression does not replace product judgment.

## 9. Acceptance gates: what would justify calling this successful?

All gates below are **proposed**, not executed or upstream benchmark claims. Freeze the evaluation protocol before selecting a winning candidate.

| Gate | Required evidence |
| --- | --- |
| Genuine 1080p | Save latent shape, pre-crop decoded raster and final crop for every test. Confirm 1920×1088 → 1920×1080 or genuinely supported direct 1920×1080, with no hidden lower-resolution binning/resizing. Inspect details at 1:1 pixels, not only thumbnail previews. |
| Impressive visual quality | Blinded review against the complete teacher and strongest admitted pretrained comparator at the same delivered resolution. Use a pilot of at least 50 held-out cases with four seeds, score all outputs, and expand until uncertainty supports a decision. A proposed gate is a clear preference improvement over the unadapted compact base, plus a predeclared acceptable gap to the quality reference. Do not invent a universal FID threshold for a tiny campaign set. |
| Semantic fidelity | Product/identity, counts, scene attributes, factual copy and layout checked independently. Evaluate rare graph branches and difficult multi-concept cases, not just average style. |
| Useful personalization | Correct versus shuffled UPL comparisons with matched image/noise/time conditions, plus independently annotated desired behavioral changes. Pixel sensitivity alone is insufficient. Include held-out UPL regions and asset/template leakage checks. |
| Diversity and optimization | Multiple seeds must produce meaningfully different valid candidates. Measure duplicate/mode collapse rates and human-perceived variety; check whether candidate selection improves a held-out criterion without shrinking support to one template. |
| Brand/typography | Measure raw student and delivered/composited outputs separately. Exact required marks, legal copy and masks must pass deterministic/OCR/geometry checks; a compositor pass must not be called learned brand fidelity. |
| Compression tradeoff | Compare unadapted base, campaign LoRA, step-distilled candidate, and quantized candidate under matched conditions. Report NFE, parameter count, total component bytes and quality drop per transformation. |
| Browser execution | Actual hardware WebGPU evidence at native target resolution; component/end-to-end parity against the pinned GPU reference with declared tolerances; cold download/init, warm p50/p95, decoder time, completion/readback and peak memory/allocation behavior. No CPU fallback or cloud response relabeled as browser success. |
| RL correctness | Synthetic tests for logging, consent, support, propensity normalization, delayed labels, critic source separation, policy training and rollback. Real CTR improvement remains unclaimed until authorized real evaluation exists. |
| Operational limits | Cancellation, budget exhaustion, missing components, unsupported device/operators, stale graph/encoder/adapter hashes, malformed tensors and nonfinite outputs fail loudly. No auto-download from a graph and no automatic promotion/nightly cloud job. |

The existing brand fallback and attribution design can survive this upgrade. The existing 32×32 architecture cannot be the final success criterion. A genuine native-1080p CUDA candidate is a quality milestone; a genuine native-1080p WebGPU candidate is a separate deployment milestone. Neither follows from a paper's H100/4090 timing at 1024×1024.

## 10. Final implementation recommendation and open decisions

**Immediate implementation work:** finish the generic candidate-policy/human-critic/CTR interfaces with explicit synthetic tests, artifact compatibility and propensity support. Do not block that work on a large distillation experiment, but keep the final product gate at native 1080p.

**First funded model experiment:** qualify pinned SANA-Sprint 0.6B/1.6B at native 1920×1088 with resolution binning disabled, and the already registered pinned Klein 4B at a verified model-aligned native shape. Compare against their appropriate base/quality references. If SANA passes, pursue base SANA campaign LoRA with a matched codec, then qualified 2–4-step distillation. If it fails detailed campaign requirements, escalate to 1.6B/Klein rather than hiding the deficit with display enlargement. Keep quantization behind quality gates.

**Black-box graph path:** pretrained latent denoising/flow adaptation to final outputs is the practical default. Conditional RDM-style feature matching is the most directly relevant 2026 sample-only research extension found here. DMD2/TDM becomes appropriate only after a compatible campaign score/velocity teacher exists. Whole-graph CUDA compilation is a separate systems optimization and never a substitute for learning.

**Still unresolved, deliberately:** any future cloud authorization/device (paid jobs are currently prohibited), actual campaign corpus size and rights, which native-resolution prior meets the visual bar, adequate conditioning capacity, component-specific license admission, exact browser runtime/quantization support, measured memory/latency, and actual human/CTR gains. The immutable candidate revisions above resolve model identity—not those empirical questions.

This review used original papers, conference proceedings, author/publisher repositories, publisher model cards and public configuration metadata. Search summaries, community forks and third-party benchmark claims were not used as the authority for admission. It is a decision-oriented review through the cutoff, not a guarantee that every 2026 preprint has been enumerated. No source establishes a universal loss that faithfully compresses every heterogeneous graph into an arbitrarily small model at arbitrary resolution.
