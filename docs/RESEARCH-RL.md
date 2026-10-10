# Reinforcement learning for visual generators

Research cutoff and source access: **2026-10-08**. This is a literature review and proposed experiment, not an implementation, benchmark result, deployment authorization, or claim of advertising effectiveness.

**Operating boundary: no paid jobs.** No paid inference, cloud GPU provisioning, advertising spend, model download, or training job was performed for this review. The native-32 RL regression and the requested high-resolution GPU-quality target are separate deliverables; the former cannot satisfy the latter.

## Recommendation

Use **DDPO-style score-function policy optimization with a clipped PPO surrogate** for the **native32 algorithm-regression track only**. Generate stochastic training trajectories, score their completed images with a frozen learned click-probability model, and update the **generator's convolution weights**. Keep human-quality and brand costs separately supervised, separately reported, and independently gated. Anchor the policy to the distilled reference and evaluate every candidate using the actual deployed **eta=0 DDIM16** sampler. This combines ideas from [DDPO, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/14f75513f0f1ca01de1e826b52e6b840-Abstract-Conference.html), [PPO](https://arxiv.org/abs/1707.06347), and [DPOK, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc65fab891d83433bd3c8d966edde311-Abstract-Conference.html); it is an Angelos-specific choice, not a reproduction or a globally best algorithm.

The selection reflects the requirement for genuine score-function RL, a small epsilon-predicting DDPM, and potentially nondifferentiable quality checks. If differentiable surrogate optimization alone were sufficient, **AlignProp/DRaFT-style reward backpropagation** would be a strong, probably cheaper baseline to measure. Do not rename it PPO. **Diffusion-DPO** is another useful weight-updating baseline when valid comparison pairs exist. **Flow-GRPO is not an interchangeable DDPM implementation.**

**For the separately selected HD foundation, SanaSprint 0.6B, the recommendation changes:** use a **DRaFT/AlignProp-inspired direct reward optimization** experiment through its actual few-step **SCM consistency sampler**, training compatible LoRA adapters while freezing foundation weights. Its flow-model ancestry does not make the deployed consistency sampler a Flow-GRPO or DDIM policy. This is a generator-weight adaptation, but **not likelihood-ratio RL** and not a claimed published reproduction on Sprint. See sections 1 and 6.5 for the architecture and gradient contract.

Critical preconditions for the native32 PPO experiment:

- Resolve the stochastic sampler's terminal zero-variance transition before computing likelihood ratios; merely setting eta=1 does not resolve it.
- Preserve separate frozen rollout-policy, immutable reference-policy, click-reward, and human-quality model versions.
- Treat predicted click improvement as model-based candidate generation, not measured CTR lift. Synthetic clicks establish neither human preference nor conversion lift.
- Allow the experiment to produce **no acceptable candidate**. Better reward must not compensate for a failed quality, brand, privacy, attribution, or deployment-parity gate.

## 1. Scope, verification, and repository fit

This review covers the requested algorithm families, visual reward learning, constrained optimization, and sparse/delayed contextual-bandit feedback. It prioritizes author manuscripts, conference proceedings, author project pages, and official implementations. Search summaries and third-party paper aggregators were discovery aids only. Dates below distinguish first arXiv submission from conference year; preprints are not promoted to peer-reviewed work based on popularity. For recent manuscripts, explicit versions avoid silently importing revisions after the cutoff. Coverage is broad but is not an exhaustive systematic review of every visual post-training paper.

**Repository observations at review time:** [DIFFUSION.md](DIFFUSION.md) and [diffusion.py](../ml/angelos_ml/diffusion.py) describe a native 32×32 RGB, four-dimensional-UPL-conditioned epsilon network, 64 cosine training timesteps, and 16 explicitly listed inference timesteps from 63 to 0. Inference clamps predicted clean images and adds no inter-step noise. `sample_ddim` is decorated with `torch.no_grad()`. The existing format exports five convolution layers, not a flow-transformer, VAE, pretrained text encoder, LoRA stack, reward model, or value critic. Reward-based training proposed here would be a separate stage after teacher execution and supervised student training.

The [event store](../src/core/store.js) binds clicks to impressions and their exposure-time snapshot, artifact, encoder, consent, and attribution policy. `maturedRows` excludes immature rows and checks training eligibility. The inspected [server](../src/server.js) exposure route records `single-candidate-v1`, `operator-exposure-v1`, and propensity 1. These are useful attribution foundations, **not randomized multi-candidate traffic or evidence of counterfactual coverage**. This document does not change those contracts.

The literature's large pretrained models provide evidence about methods, not expected gains, compute costs, or visual capability of this tiny student. Improving its reward does not create high-resolution generation, establish GPU success, or repair insufficient teacher data/capacity.

### Two architecture tracks, not one inflated claim

| Track | Explicit policy choice | Evidence required |
| --- | --- | --- |
| **Current algorithm regression** | Existing native-32 epsilon DDPM; DDPO-style stochastic denoising PPO; independent eta=0 DDIM16 evaluation. | Correct likelihoods, real generator-weight updates, honest reward provenance, quality/brand gates, and numerical parity. No 1080-quality claim. |
| **Selected high-resolution research target** | The user's pinned official **SanaSprint 0.6B** foundation, **SCM consistency sampling**, and **direct reward backpropagation into LoRA**. Freeze the foundation; retain a frozen parent output for matched-noise anchoring. | Faithful sampler and decoder gradients, unchanged-base/changed-adapter evidence, actual high-resolution GPU quality, reward-fit gates, and independent held-out quality/brand acceptance. Nothing is provisioned or downloaded here. |

Distinguish **SANA** from **SANA-Sprint**. The original SANA is a latent linear-DiT model with flow-based training/sampling. Sprint adds continuous-time consistency and latent adversarial distillation for one/few-step generation. [SANA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/34345e243156da67605d4b63d71c8d98-Abstract-Conference.html), [SANA-Sprint, ICCV 2025, peer reviewed](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_SANA-Sprint_One-Step_Diffusion_with_Continuous-Time_Consistency_Distillation_ICCV_2025_paper.pdf), [Sprint author manuscript](https://arxiv.org/abs/2503.09641)

The official `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers` [model index](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/blob/main/model_index.json) specifies `SanaSprintPipeline`, `SCMScheduler`, `SanaTransformer2DModel`, `AutoencoderDC`, and Gemma text conditioning. This corroborates the architecture, not a newly selected immutable revision. Carry the exact already-pinned revision and file hashes from the integration manifest; do not load moving `main` or silently substitute standard SANA. Version the permitted UPL-to-conditioning interface; native32's four spatially broadcast UPL channels are not an input contract for unmodified Sprint.

**Flow-GRPO remains a future alternative only if a distinct, verified base flow checkpoint and corresponding stochastic path are deliberately selected.** It is not the recommendation for pinned Sprint. Swapping `SCMScheduler` for a flow scheduler, labeling a consistency update “DDPO,” or treating raw transformer outputs as Gaussian log probabilities would not establish a valid RL policy. Supporting score-function RL on a consistency model would require its own density/transition derivation and validation.

If “1080” means a 1920×1080 deliverable, specify orientation and the pinned model's supported resolution/aspect buckets, then disclose any necessary padding/crop or resizing. A 1024-square result, native32 nearest-neighbor enlargement, or GPU-executed resize is not evidence of meeting that quality target. Check full-resolution details, text/product identity, artifacts, diversity, and exact composition on actual GPU-rendered outputs. This architecture needs a new artifact/runtime contract, not a mislabeled `angelos.diffusion.v1` export. Its resource feasibility remains unmeasured; the no-paid-jobs restriction is not relaxed by this recommendation.

## 2. Foundational and established methods

“Applicable” below means a plausible adaptation, not implemented support or demonstrated benefit.

| Work and verified status | Mechanism and evidence boundary | Fit to this student |
| --- | --- | --- |
| **DDPM** — Ho et al.; first submitted 2020-06-19; NeurIPS 2020. [Paper](https://arxiv.org/abs/2006.11239) | Epsilon/score-related denoising training and stochastic reverse generation. An epsilon-MSE training loss is not an exact terminal-image log likelihood. | Existing model family. Keep the exact schedule and prediction convention explicit. |
| **DDIM** — Song et al.; 2020-10-06; ICLR 2021. [Paper](https://arxiv.org/abs/2010.02502) | A family of samplers compatible with diffusion training, including deterministic sampling and stochastic variants. | Existing eta=0 deployment. A stochastic training variant requires a new, separately specified rollout contract. |
| **PPO** — Schulman et al.; 2017-07-20; original technical report/preprint. [Paper](https://arxiv.org/abs/1707.06347) | Alternates rollout collection with optimization of a clipped likelihood-ratio surrogate. Clipping is not a hard guarantee of KL, quality, or monotonic improvement. | Use the surrogate on valid denoising transition densities; not on arbitrary scores or deterministic transitions. |
| **DDPO** — Black et al.; 2023-05-22; **ICLR 2024, peer reviewed**. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/14f75513f0f1ca01de1e826b52e6b840-Abstract-Conference.html), [manuscript](https://arxiv.org/abs/2305.13301), [author JAX code](https://github.com/jannerm/ddpo) | Treats denoising as a finite-horizon decision process. DDPO's score-function and importance-sampled variants update generator weights using terminal rewards without differentiating the reward. Tests include compressibility, aesthetics, and prompt alignment, not CTR. | **Selected family.** Adapt trajectory likelihoods and PPO updates to this small network; no Stable Diffusion pipeline dependency is inherently necessary. |
| **DPOK** — Fan et al.; 2023-05-25; **NeurIPS 2023, peer reviewed**. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fc65fab891d83433bd3c8d966edde311-Abstract-Conference.html), [manuscript](https://arxiv.org/abs/2305.16381), [official code](https://github.com/google-research/google-research/tree/master/dpok) | Online diffusion policy-gradient fine-tuning with KL regularization toward a pretrained reference. Relevant support for preserving the starting generator while optimizing a learned reward. Despite its name, this is not the pairwise DPO algorithm. | Use as the reference-regularization rationale, not as proof that a chosen penalty preserves human quality. |
| **DPO** — Rafailov et al.; 2023-05-29; **NeurIPS 2023, peer reviewed**. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract.html), [manuscript](https://arxiv.org/abs/2305.18290) | Reparameterizes a KL-regularized preference problem into a pairwise policy objective under its preference/model assumptions. No separately fitted reward model or actor-critic rollout loop is required in the original offline procedure. | Not directly usable with intractable image likelihoods; use a diffusion-specific derivation instead. |
| **Diffusion-DPO** — Wallace et al.; 2023-11-21; **CVPR 2024, peer reviewed**. [Proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Wallace_Diffusion_Model_Alignment_Using_Direct_Preference_Optimization_CVPR_2024_paper.html), [manuscript](https://arxiv.org/abs/2311.12908), [official code](https://github.com/SalesforceAIResearch/DiffusionDPO) | Uses diffusion variational/noise-prediction terms to make preference optimization tractable. It is a surrogate, not direct evaluation of exact image likelihood. Reported human-preference gains on SDXL do not establish click gains. | Good offline weight-update comparator. Sample winner/loser images under the same conditioning and use paired noise/timesteps as prescribed. Clicked versus unclicked impressions from different contexts are not automatically preference pairs. |
| **AlignProp** — Prabhudesai et al.; 2023-10-05; **historical preprint, withdrawn as superseded**, not treated here as a verified standalone conference publication. [Status/history](https://arxiv.org/abs/2310.03739), [project](https://align-prop.github.io/), [official code](https://github.com/mihirp1998/AlignProp/) | Backpropagates differentiable rewards through sampling; checkpointing, adapters, and truncated/randomized gradient horizons manage cost. The arXiv record explicitly says it is subsumed by the later video paper; v5, 2024-11-07, is withdrawn. This is a bibliographic distinction, not evidence the method failed. | Strong deterministic-DDIM alternative, but would need a gradient-enabled training sampler rather than calling the existing no-grad function. |
| **DRaFT** — Clark et al.; 2023-09-29; **ICLR 2024, peer reviewed**. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html), [manuscript](https://arxiv.org/abs/2309.17400) | Direct reward gradients through sampling; DRaFT-K truncates to the last K steps and DRaFT-LV reduces variance for K=1. Truncation changes the gradient estimator. | Useful cheaper baseline with the same learned click reward and constraints; no guarantee of robustness to reward-model exploitation. |
| **D3PO** — Yang et al.; **CVPR 2024, peer reviewed**. [Proceedings](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Using_Human_Feedback_to_Fine-tune_Diffusion_Models_without_Any_Reward_CVPR_2024_paper.html) | Direct preference optimization of denoising policies without an explicit reward model. | Relevant alternative when actual comparison labels and the required trajectory construction are available; not a substitute for click-outcome attribution. |
| **Dense-reward preference alignment** — Yang, Chen, Zhou; **ICML 2024, peer reviewed**. [Paper/code link](https://proceedings.mlr.press/v235/yang24e.html) | Introduces temporal asymmetry into preference alignment rather than treating all denoising stages identically. | An ablation to consider after a working terminal-reward baseline. It does not mean users supplied a click label for each noisy state. |
| **Adjoint Matching** — Domingo-Enrich et al.; **ICLR 2025, peer reviewed**. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/852f50969a9e523ec41d26f2f68bd456-Abstract-Conference.html) | Stochastic-optimal-control formulation and regression-based fine-tuning; its memoryless noise schedule is a substantive assumption. | Interesting later comparator. Do not transfer its guarantees to arbitrary clipped DDIM transitions. |
| **Flow-GRPO** — Liu et al.; 2025-05-08; **NeurIPS 2025, peer reviewed**. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3a10c46572628d58cb44fb705f25cbbf-Abstract-Conference.html), [paper](https://arxiv.org/abs/2505.05470), [official code](https://github.com/yifan123/flow_grpo) | Uses group-relative rewards, a flow ODE-to-SDE construction, and reduced training denoising steps. SD3.5-M GenEval improves from 63% to 95% in the authors' setting; this is composition accuracy, not advertising performance. Reported limited reward hacking is an experimental finding, not a guarantee. | Borrow the idea of same-context baselines, not the flow transition formulas. Requires model/path-specific adaptation, not renaming DDPO. |

Reward-weighted regression and rejection-sampling-plus-supervised-fine-tuning also update generator weights if followed by training, and belong in cost-matched baselines. Ranking seeds, choosing prompts, or selecting among unchanged creatives **without a weight update** does not satisfy this task. DDPO compares against reward-weighted alternatives, but does not establish universal superiority across architectures, data budgets, or rewards. [Author explanation](https://bair.berkeley.edu/blog/2023/07/14/ddpo/)

### Why flow and diffusion “scores” are not interchangeable

Flow Matching learns a velocity field along a specified probability path; the student's network predicts epsilon for a discrete variance-preserving noise schedule. A diffusion score estimates a gradient of log noisy-data density, not click utility, and is neither an RL value critic nor an exact endpoint log probability. Relationships between velocity, score, and epsilon require the correct path coefficients and time convention. Arbitrary eta=1 noise injection does not reproduce Flow-GRPO's marginal-preserving continuous-time construction; numerical discretization, imperfect learned fields, and clean-image clipping also matter. [Flow Matching, ICLR 2023](https://arxiv.org/abs/2210.02747), [Flow-GRPO derivation](https://arxiv.org/html/2505.05470v5)

## 3. Credible developments available by the cutoff

These are relevant alternatives and warnings, not a leaderboard. **Preprint** means that this review verified an author manuscript but did not establish a peer-reviewed publication by the cutoff. Claimed speedups use different models, rewards, starting checkpoints, and accounting; they are not performance predictions for Angelos.

| Development and date/status | Verified contribution | Practical implication |
| --- | --- | --- |
| **DiffusionNFT** — first submitted 2025-09-19; **ICLR 2026, peer reviewed**. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d8e68ddfe22520b45b8fb8d5cbde5a21-Abstract-Conference.html), [manuscript](https://arxiv.org/abs/2509.16117) | Forward-process, negative-aware training using clean generated samples and reward signals, avoiding reverse-trajectory likelihood estimation in its formulation. Uses a flow-matching training construction. | A substantial alternative to reverse-process PPO, not just another clipping rule. Arbitrary rollout solvers do not imply arbitrary epsilon-model training targets are plug-compatible. |
| **TempFlow-GRPO** — **ICLR 2026, peer reviewed**. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d75f561eaaf2cb754bc8d7e36d8af362-Abstract-Conference.html) | Trajectory branching, noise-aware temporal weighting, and seed groups improve credit assignment in flow models. | Supports testing timestep sensitivity and controlling seed effects. Does not supply missing click evidence or a DDPM terminal likelihood. |
| **PaCo-RL** — **CVPR 2026, peer reviewed**, June proceedings. [Proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Ping_PaCo-RL_Advancing_Reinforcement_Learning_for_Consistent_Image_Generation_with_Pairwise_CVPR_2026_paper.html), [author manuscript](https://arxiv.org/abs/2512.04784) | Pairwise reward modeling for consistent image generation; PaCo-GRPO includes resolution-decoupled optimization and multi-reward aggregation. | Reference-conditioned consistency is relevant to products/brands, but neither learned consistency nor reward aggregation guarantees exact logo fidelity. |
| **Understanding Reward Hacking in Text-to-Image RL** — 2026-01-06; v2 2026-08-31; **preprint**. [Versioned paper](https://arxiv.org/abs/2601.03468v2) | Documents artifacts under aesthetic/preference and prompt-consistency optimization; reward ensembles only partly mitigate them. Proposes a separately learned artifact reward. | Strong reason to retain independent artifact/quality checks rather than trust click reward or a single aggregate score. |
| **Rethinking the Design Space of RL for Diffusion Models** — 2026-02-04; v3 2026-09-17; **preprint**. [Versioned paper](https://arxiv.org/abs/2602.04663v3) | Separates policy-gradient loss, likelihood estimator, and rollout sampler; studies endpoint ELBO-based likelihood estimation on SD3.5-M. | Audit density estimation before copying a fashionable loss. An ELBO-based ratio is not automatically an exact likelihood ratio for this student's deployed image distribution. |
| **DeDPO** — 2026-02-05; v1; **preprint**. [Paper](https://arxiv.org/abs/2602.06195v1) | Combines limited human labels with synthetic preference labels and a debiasing estimator. | Potential label-efficiency research. Synthetic annotator correction does not debias advertising exposure selection, prove causal CTR effects, or turn synthetic labels into human observations. |
| **Finite Difference Flow Optimization (FDFO)** — 2026-03-13; v2 2026-06-30; **preprint with official code/checkpoints**. [Paper](https://arxiv.org/abs/2603.12893v2), [NVIDIA code](https://github.com/NVlabs/finite-difference-flow-optimization), [official model card](https://huggingface.co/nvidia/finite-difference-flow-optimization) | Pairs trajectories and uses their reward difference to move flow velocities; treats generation as a whole action rather than separate denoising actions. | Credible black-box-reward alternative, not PPO and not a drop-in epsilon update. The published asset card identifies non-commercial licensing; do not assume unrestricted reuse. |
| **Reinforce Adjoint Matching (RAM)** — 2026-05-11; v2 2026-05-17; **preprint**. [Paper](https://arxiv.org/abs/2605.10759v2) | Generates clean endpoints, scores and forward-noises them, then uses a reward-corrected regression target. Avoids reward gradients and backward adjoint sweeps in the proposed method. | Worth investigating if trajectory PPO is unstable or expensive. Its KL/control derivation needs a faithful translation to this finite clipped sampler before guarantees can be claimed. |
| **RL-Native Distillation / REST** — 2026-08-10; v2 2026-09-28; **preprint**. [Paper](https://arxiv.org/abs/2608.09226v2) | A decoupled student learns from a teacher's reward-scored trajectories, with advantage-modulated distillation. | Useful future bridge between reward alignment and few-step deployment. Existing teacher final rasters are not the intermediate trajectories this method requires. DDIM16 is not evidence that this distillation has been performed. |
| **DOHF** — 2026-09-25; v2 2026-10-01; **preprint**. [Paper](https://arxiv.org/abs/2609.31882v2) | Estimates an online Doob h-transform correction from reward-weighted samples and distills it into the generator, accommodating black-box rewards. | Very recent alternative for rare desired outcomes. Keep it on a research watchlist; sparse user clicks and finite-sample reward-model bias are still separate problems. |

**Beyond still images:** [InstructVideo, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yuan_InstructVideo_Instructing_Video_Diffusion_Models_with_Human_Feedback_CVPR_2024_paper.html) uses partial denoising and image reward models for video tuning. [Video Diffusion Alignment via Reward Gradients / VADER](https://arxiv.org/abs/2407.08737) is an author preprint submitted 2024-07-11 and is the successor named by AlignProp's withdrawal notice. It uses reward gradients for video adaptation. Neither makes frame-level quality sufficient for motion, temporal consistency, audio synchronization, or session-level outcomes. Video/audio/3D extensions need their own models, rewards, and contracts; the present proposal targets image weights only.

## 4. Reward models are not value critics

The separation between learning a reward from comparisons and optimizing a policy is established in [Deep RL from Human Preferences, NeurIPS 2017](https://arxiv.org/abs/1706.03741). For this project, keep the following roles explicit:

| Component | Input, target, and role | What it does not establish |
| --- | --- | --- |
| Learned click reward `p_click` | Exposure-time permitted context plus the exact displayed creative; estimates probability of at least one attributed click within a fixed window. Fit from matured outcomes. | Not a human-quality score, conversion probability, causal uplift model, or measured outcome on a new creative. |
| Human-quality model `q_human` | Image and relevant campaign requirements; supervised by identifiable human ratings/comparisons for appeal, coherence, artifacts, and suitability. | An operator/synthetic heuristic cannot be relabeled human feedback. Preference scores need not be calibrated probabilities. |
| Brand checks/costs | Exact identity/layout/text/plate constraints, with learned semantic checks only where appropriate. | Learned similarity is not exact typography, legal-copy, QR, or logo integrity. |
| Optional value critic `V` | Noisy state, conditioning, timestep, and a policy version; predicts expected remaining training return for variance reduction/advantages. | It is not independent evidence that a completed image is good and does not replace the click or quality reward model. |
| Diffusion epsilon/score network | Denoises the image; its weights are the optimized policy parameters. | A “score network” is not a reward scorer or value estimator. |

PPO is often implemented with a critic, but **the clipped policy objective does not require a separately learned value network**. A terminal-return baseline, preferably estimated from independent earlier batches or leave-one-out same-context samples, is sufficient for an initial DDPO-style experiment. Group-relative baselines also avoid a value network; that fact alone does not make an epsilon-DDPM experiment Flow-GRPO. Avoid silently biasing a nominally unbiased baseline by including its own sample; PPO clipping and sample normalization already introduce approximation choices that should be reported.

### Human preference evidence versus click evidence

- [ImageReward/ReFL, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/33646ef0ed554145eab65f6250fab0c9-Abstract-Conference.html) learns a preference scorer from expert comparisons and uses reward feedback for generator tuning. [Official code](https://github.com/zai-org/ImageReward) is an integration reference, not a click model.
- [Pick-a-Pic/PickScore, NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/73aacd8b3b05b4b503d58310b523553c-Abstract-Conference.html) learns from explicit image choices. A user's comparative selection in that app is not an ad click after an exposure.
- [HPS v2, author preprint](https://arxiv.org/abs/2306.09341) also models visual preferences. These scorers can inform quality evaluation after domain validation, but upscaling native-32 images to their input size does not establish score validity.

**Proposed click-model protocol:** use a small image-conditioned predictor, not only profile or artifact-ID lookup. Otherwise changing generator pixels may have no meaningful effect on predicted reward, or the model may memorize identities. Use binary likelihood training on mature impression-level labels, regularization, a prevalence baseline, and calibration checks on held-out time/profile/creative groups. Report log loss, Brier score, reliability by score band, PR-AUC, positive counts, and uncertainty; ROC-AUC alone is insufficient for rare clicks. Calibration is a separate property from ranking. [Calibration research, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)

Do not treat unexposed candidates as negatives, count repeated click events as independent positive impressions, label pending exposures as zero, or silently mix observation windows. If negatives are subsampled or class weights alter prevalence, correct/recalibrate before calling the output a click probability. Keep synthetic and observed outcome provenance separate. Consent must be valid at exposure and remain valid for the proposed training use; never rebuild historical context from the latest profile.

Freeze reward models during each policy experiment. Version and hash their training data, calibration, preprocessing, and label policy. Refit only in a separately evaluated stage. An uncertainty penalty or ensemble disagreement may discourage extrapolation, but neither is a certified lower confidence bound without a justified statistical construction. Too few positives or no creative overlap should yield **insufficient evidence**, not a default “improved” result.

**Two mandatory model-readiness gates:** (1) validate the training reward/quality scorers on independent labels, including human-quality labels when that claim is made; (2) validate the click predictor's fit and calibration on exposure-bound, matured **observed** clicks if an observed-CTR predictor is claimed. Synthetic labels qualify only for the synthetic regression track. Passing either fit gate does not pass the other, and passing both still does not establish true online CTR lift for new generator weights. Keep quality/brand rejection active in both architecture tracks regardless of reward-model fit. If eligibility filters exclude creatives from training/evaluation, report the resulting restricted population rather than silently changing the estimand.

## 5. Constrained candidate generation, not unconstrained engagement

Let `context` denote the allowed conditioning/evaluation distribution, `image` a final rendered sample, and `policy_theta` the stochastic training generator. A useful *design objective* is:

```text
maximize_theta  E[p_click(context, image)]
subject to      E[quality_cost(context, image)] <= quality_budget
                E[brand_cost(context, image)] <= brand_budget
                reference_divergence(theta, reference) <= divergence_budget
```

Costs, budgets, and eligible contexts must be specified before candidate selection. A Lagrangian training reward can be `p_click - lambda_quality * quality_cost - lambda_brand * brand_cost`, with reference KL added separately to the optimization objective. Fixed penalties are **a scalarized heuristic**, not proof of constraint satisfaction. Dual-variable updates are a possible constrained extension; expected-cost feasibility still does not guarantee every creative passes. [Constrained Policy Optimization, ICML 2017](https://proceedings.mlr.press/v70/achiam17a.html) motivates explicit constraints, but its guarantees do not transfer automatically to this proposed PPO-plus-gates implementation.

**Proposed safeguards:**

1. Keep component scores visible and independently versioned; do not let aggregate reward hide a quality deficit. Predeclare quality noninferiority margins and brand rejection criteria.
2. Preserve exact deterministic brand fallback where required. Score click reward on the actual displayed composition, but measure and report raw-student and composited fidelity separately. A compositor must not conceal a claim about raw generator learning.
3. Preserve a feasible supervised reference and an operator-controlled rejection path. Check quality, diversity, saturation, artifacts, and support distance on held-out contexts and seeds. Average quality cannot replace checks for critical individual violations.
4. Include blinded human review independent of reward training where human-quality claims are made. Keep calibration/selection data separate from the final audit; do not repeatedly tune on the final test set.
5. Stop or reject on reward/evaluation disagreement, excessive divergence, degenerate diversity, or deployment-quality regression. No automatic promotion follows a higher learned reward.

Overoptimization is not hypothetical: [Gao et al., ICML 2023](https://proceedings.mlr.press/v202/gao23h.html) measures proxy-versus-gold reward degradation in a synthetic reward-model setup, not a real-user conversion experiment. [Confidence-aware Reward Optimization, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/275cabf15970186f9a4910702a77d357-Abstract-Conference.html) specifically studies confidence-calibrated rewards for image tuning. Alongside the 2026 artifact study, these support independent checks, uncertainty awareness, and limited optimization. They do not supply a universal safe KL threshold.

## 6. Algorithm choices: native32 PPO and HD consistency adaptation

Sections 6.1–6.4 specify the native32 design and mathematical audit. The coordinating implementation now reports a native32 DDPO regression; this document does not certify that parallel implementation's numerical behavior. Its intended artifact is an immutable `angelos.diffusion.v1` candidate with independently evaluated deterministic inference. Section 6.5 specifies the separate HD proposal, which does not use that artifact or algorithm contract.

### 6.1 State, action, and valid stochastic transitions

Use `(noisy_image, timestep, UPL)` as state; the next noisy image is the continuous action. Sample initial noise independently of trainable weights. A complete trajectory receives the frozen click reward and separately computed constraint costs. Rollout is performed without reward backpropagation; for score-function updates, states, sampled actions, reward values, and old log probabilities are detached.

For an actual adjacent pair `time -> next_time` in the stored schedule, let `alpha` and `alpha_next` be their cumulative alphas. For nonterminal transitions, the stochastic DDIM construction is:

```text
clean_prediction = clamp((noisy - sqrt(1-alpha) * epsilon_theta) / sqrt(alpha), -1, 1)
variance = eta^2 * (1-alpha_next)/(1-alpha) * (1-alpha/alpha_next)
mean_theta = sqrt(alpha_next) * clean_prediction
             + sqrt(1-alpha_next-variance) * epsilon_theta
next_noisy ~ Normal(mean_theta, variance * identity)
```

This retains the student's clipped-clean prediction convention without silently recomputing epsilon from the clipped prediction. Other conventions are possible, but sampling, likelihood recomputation, old/reference comparisons, and metadata must agree. Use the **stored pairs**, not an assumed constant subtraction: rounding 16 indices across 64 timesteps produces nonuniform gaps. The underlying sampling family is described by [DDIM](https://arxiv.org/abs/2010.02502); the [DDPO author sampler implementation](https://raw.githubusercontent.com/kvablack/ddpo-pytorch/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py) illustrates why scheduler and log-probability details need inspection rather than wholesale copying.

**Terminal issue:** Angelos uses `alpha_next=1` after timestep 0. The formula therefore yields zero variance even at eta=1. A Gaussian log density, Gaussian KL denominator, or PPO ratio at that deterministic transition is undefined. Randomness in initial noise does not fix its conditional density.

For a clean score-function experiment, a defensible **research sampler modification** is to replace the last transition with an explicitly declared positive-variance Gaussian around the final clean prediction. Store the *pre-clamp* sampled terminal action and its actual density; map it to display pixels through fixed clipping/composition for reward evaluation. This is **stochastic DDIM-inspired rollout with a terminal-noise modification**, not exact standard eta=1 DDIM. Choose and record its variance in a pilot; evaluate the mismatch to deployed eta=0 independently. Do not insert a variance floor into the terminal DDIM square-root formula, which would make its residual coefficient invalid.

An alternative is a policy-independent frozen terminal decoder and likelihood optimization only on preceding stochastic actions, also explicitly changing the training process. Simply dropping the terminal likelihood while its theta-dependent decoder keeps changing yields a partial-gradient heuristic, not the full score-function gradient of the stated generator objective. Adding epsilon only to a log-probability denominator without sampling that variance is not a valid fix. This terminal treatment is a design decision required before implementation, not an assumption to hide.

**Implementation handoff update:** the coordinating native32 implementation is being changed to **15 stochastic transitions plus a frozen-reference terminal decoder**, rather than a parameter-dependent final candidate step. This is a coherent density contract for that training process, conditional on correct implementation. Its candidate still runs all 16 candidate-model steps at deployment, so the training/deployment distribution gap must be explicitly reported and independently audited. This research note records the intended fix, not a claim that its tests have passed.

### 6.2 Likelihood ratios, baselines, and clipping

For nondegenerate fixed-covariance Gaussian actions, compute a scalar **joint** log density by summing over all image coordinates. Define `ratio = exp(log_prob_theta - log_prob_old)` on the *same stored state and action*. Optimize the mean clipped surrogate `min(ratio * advantage, clip(ratio, 1-clip_width, 1+clip_width) * advantage)`. A completed-image return supplies an advantage to earlier denoising decisions; it is not a claim of observed intermediate user rewards. [PPO objective](https://arxiv.org/abs/1707.06347), [DDPO training implementation](https://raw.githubusercontent.com/kvablack/ddpo-pytorch/main/scripts/train.py)

**Implementation distinction:** the inspected author `ddim_with_logprob.py` averages elementwise log densities. Exponentiating a difference of these means gives a dimension-normalized/tempered ratio, not the joint Gaussian density ratio. Copying that choice may be a useful heuristic, but must be labeled and ablated. For this experiment's literal likelihood-ratio contract, start with summed log densities; report per-coordinate-normalized diagnostics separately. At 3×32×32 action dimensions, sums can be numerically sensitive, so use small updates, stable arithmetic, and divergence/clip-fraction diagnostics rather than silently changing the probability definition.

Maintain an immutable `old` snapshot for each collected batch, limit update epochs, and collect fresh trajectories before excessive drift. Use a baseline fitted before the current batch or leave-one-out rewards for repeated identical contexts. Do not normalize across unrelated campaigns as though all return scales were interchangeable. If reward variance is negligible, report a no-signal condition instead of amplifying numerical noise. A value network is optional and should be added only if its measured variance reduction justifies complexity.

The accompanying implementation handoff also selects **leave-one-out group baselines**, replacing a group mean that includes the sample's own return. Correct indexing, group size, replay behavior, and fixed terminal-decoder identity require regression tests before this implementation is endorsed.

### 6.3 Reference KL and supervised anchoring

The frozen rollout policy and the frozen distilled reference are **different roles**: old-policy ratios control reuse of the latest batch; reference regularization limits departure from the original useful model. With identical positive variance in both policies, conditional Gaussian KL is `squared_norm(mean_theta - mean_reference) / (2 * variance)`. Compute both means at the same state, schedule, and conditioning.

A sum of conditional KLs averaged under current-policy trajectories gives the corresponding path KL when initial distributions and transition conventions match. Estimating it on old-policy states during PPO updates is a local surrogate, not an exact current-path or deployed-image KL. Clipping, endpoint mapping, and changing eta must not be ignored when interpreting it. DPOK provides the diffusion-specific motivation for reference regularization. [DPOK analysis and code](https://github.com/google-research/google-research/tree/master/dpok)

An optional supervised epsilon-MSE replay loss on retained authorized teacher data can additionally reduce forgetting. Neither teacher replay nor parameter-distance regularization should be called an exact KL. Log each objective separately, including its scale and reduction over coordinates/timesteps. Very small variances can dominate KL and gradients; monitoring only the batch-average reward misses this failure.

### 6.4 What this optimizes, and why not another method first

The policy learns to obtain a **predicted** click reward under a stochastic training distribution. It does not differentiate through actual binary clicks, and it does not estimate causal lift merely by using PPO. The reward is a surrogate environment learned from outcomes. It is still genuine score-function generator RL when updates use sampled-action log probabilities and detached returns.

The method is attractive here because it updates existing weights, tolerates nondifferentiable checks, and leaves the inference format unchanged. Its liabilities are high-dimensional likelihood variance, rollout cost, reward-model exploitation, clipping bias, and stochastic-to-deterministic distribution mismatch. These are reasons to compare alternatives, not to advertise PPO as optimal.

For the direct-gradient baseline, reward-model parameters remain frozen but gradients flow through its image input and a gradient-enabled DDIM16 chain into generator weights. The existing `torch.no_grad()` sampler cannot perform that task. Full-chain gradients require more activation memory; truncated gradients trade compute for bias, and repeated x0 clipping can saturate either method's parameter gradients. [DRaFT](https://proceedings.iclr.cc/paper_files/paper/2024/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html) supports this comparator, not an unmeasured speedup claim on this repository.

### 6.5 HD SanaSprint: direct reward optimization through SCM

**Decision:** adapt the pinned consistency generator itself, using low-rank weight updates, not a fictitious Gaussian policy layered onto its outputs. DRaFT supplies the peer-reviewed precedent for differentiable-reward fine-tuning; AlignProp supplies a closely related historical design. Neither paper establishes this specific Sprint implementation. As noted in section 2, **AlignProp arXiv:2310.03739 is withdrawn as subsumed by arXiv:2407.08737**, not a separate verified current publication. [DRaFT](https://arxiv.org/abs/2309.17400), [AlignProp status](https://arxiv.org/abs/2310.03739), [VADER successor](https://arxiv.org/abs/2407.08737)

The inspected [Diffusers v0.40.0 Sprint pipeline](https://raw.githubusercontent.com/huggingface/diffusers/v0.40.0/src/diffusers/pipelines/sana/pipeline_sana_sprint.py) has a no-grad inference entry point, specific timestep/output transformations, and autoencoder scaling before decoding. Its [SCM scheduler](https://raw.githubusercontent.com/huggingface/diffusers/v0.40.0/src/diffusers/schedulers/scheduling_scm.py) uses trigflow prediction and an explicit `sigma_data` convention. A gradient-enabled training path must preserve the actual pinned implementation's arithmetic, schedule, guidance, resolution handling, and random draws; directly calling the public no-grad pipeline cannot train the adapters.

**Proposed objective and parameter ownership:** let `adapter` be the only trainable generator parameters, `image_adapter` the output of the real SCM sampler plus differentiable decode, and `image_parent` the adapter-disabled parent at identical conditioning and noise. Minimize:

```text
E[-p_click(context, displayed_image_adapter)
  + lambda_quality * differentiable_quality_cost(image_adapter)
  + lambda_brand * differentiable_brand_cost(image_adapter)
  + lambda_anchor * distance(image_adapter, stop_gradient(image_parent))]
```

This is a pathwise/reparameterized gradient objective. It needs **no PPO ratio, action log density, GAE, or value critic**. The matched-noise parent distance is an output/feature/latent consistency regularizer, depending on its declared definition, **not a KL divergence or a constraint guarantee**. The reward model may be trained from exposure-bound observed clicks, but gradients optimize its prediction, not the unobserved user response to a new image.

Practical design requirements, not claimed tested behavior:

1. Freeze base transformer, text encoder, autoencoder, click predictor, and quality scorer parameters. Enable gradients only on a documented, compatible set of LoRA targets. Frozen decoder/reward parameters must still allow gradients **with respect to their inputs**; putting their entire forward passes under no-grad would break the required chain. Cache frozen text embeddings where appropriate.
2. Hold initial noise and any scheduler re-noising draws fixed within each parent/candidate comparison; repeat with fresh draws across training batches. Reset scheduler state between executions. Preserve the exact one/few-step deployment configuration and guidance conditioning; do not infer DDIM semantics from a similarly named `eta` argument.
3. Use differentiable tensor preprocessing for the learned image-click and quality surrogates, preserving image gradients through resizing and decoding. Do not introduce PIL conversion, byte quantization, detached CPU copies, or detached scorer features into the training path. Keep nondifferentiable exact brand/text/QR checks as **hard evaluation/rejection gates**, with explicitly labeled smooth training penalties if desired.
4. Anchor at matched noise/conditioning, inspect both latent and decoded consequences, and declare any mask that excludes a deterministic brand plate. A strong pixel anchor can suppress useful variation; a weak one permits drift. Select coefficients on a validation split, report a tradeoff curve, and retain independent held-out acceptance rather than claiming an optimal penalty.
5. Validate gradient flow, a small finite-difference/gradient-direction check, base-weight immutability, adapter-weight changes, and adapter-disabled parent replay. Profile actual device, memory, wall time, decoder backpropagation, and tensor resolution; a few sampling steps do not make high-resolution backward passes free. No paid job is authorized.
6. Evaluate the saved candidate with the **unchanged actual Sprint SCM inference path**, not native32 DDIM16. Use disjoint prompts/contexts/seeds, raw and composited outputs, full-resolution artifact and brand checks, human review when claimed, and separate click-predictor fit/calibration gates. Compare against its unadapted parent; reject regressions. Record base revision, adapter hashes/targets, sampler settings, and reward-data provenance in a distinct HD artifact.

This makes the adaptation an actual generator-weight candidate despite frozen foundation weights: the trained adapter changes the generator's effective weights. It is not merely reward-model fitting, prompt tuning, or reranking. It also does **not** satisfy a demand that the HD method itself be likelihood-ratio RL; that distinction must remain visible. Native32 PPO demonstrates the score-function algorithm separately, and neither track demonstrates online CTR or conversion lift without appropriate fresh real-outcome evidence.

## 7. Sparse delayed clicks, contextual bandits, and OPE

### 7.1 There are two different decision problems

The **internal denoising MDP** is an algorithmic representation used to update a generator. The **external exposure decision** chooses a creative for a context and receives, at most, a delayed observed outcome for that chosen action. For a single-exposure click objective this is naturally a contextual bandit; sequential user fatigue, carryover, or long-term value would require a richer causal/sequential model. [Li et al., WWW 2010](https://arxiv.org/abs/1003.0146) is foundational evidence for contextual-bandit recommendation, not proof of applicability of its numerical results to this campaign.

Do not confuse Gaussian denoising transition probabilities with the **logging propensity of selecting/displaying a creative**. A propensity is neither the predicted CTR nor the probability a user clicks. If a safety filter or reranker changes selection, its effect belongs in the actual exposure-policy probability.

### 7.2 Delayed and sparse outcomes

[Chapelle, KDD 2014, official publication listing](https://ailab.criteo.com/publications/) introduced delayed-feedback modeling in display advertising. [Vernade, Cappé, Perchet, UAI 2017](https://arxiv.org/abs/1706.09186) explicitly distinguishes delayed and censored conversions and assumes a known delay distribution in its model. These are reasons not to convert “not yet observed” into “negative”; they do not imply clicks and conversions have the same delay process.

For the small demo, prefer the existing mature-window labels over fitting a new survival model from tiny data. Fix a window and late-arrival policy, freeze an `asOf` snapshot, deduplicate outcomes per impression, and retain missing/pending/censored status. A mature zero is zero for that window, not a proof of no eventual conversion. Positive clicks are not conversion labels. A more ambitious delay/survival model needs identifiable censoring assumptions and independent validation.

Split by time and relevant profile/creative/campaign clusters to prevent leakage from repeated exposures or duplicate images. Report effective positive sample counts and uncertainty, not only a point estimate or rounded percentage. Unexposed images provide no directly observed outcome, however confidently a simulator or reward model scores them.

### 7.3 Logging prerequisites

For a future authorized multi-candidate experiment, preserve:

- Immutable exposure-time permitted context, consent and encoder versions, campaign, placement, and decision time.
- Exact artifact/creative identity and replayable seed or content-addressed displayed pixels, including composition and preprocessing versions.
- Available candidate set, eligibility/filter decisions, behavior-policy version, and actual conditional probability of the selected action after all selection steps.
- Impression and outcome IDs, event and receipt times, maturity/censoring rules, outcome source, and experiment/randomization unit.
- Dataset cutoff, exclusions, consent revocations, and enough provenance to reconstruct the evaluation estimand without fetching latest profiles.

This is a proposed evidence contract, not a claim that every field is currently stored. In particular, an action-set identifier alone does not reconstruct its membership, and a hashed creative identity alone does not necessarily reconstruct its pixels.

### 7.4 Estimators and their limits

For logged `(context, action, outcome, behavior_probability)` tuples, a target selection policy has weight `target_probability / behavior_probability`. With adequate support and correct logging/identification assumptions:

- **Direct method:** average a learned outcome predictor under the target. This is model-dependent and is essentially what the generator's click surrogate supplies for novel images; extrapolation error is not removed by optimizing it.
- **IPS:** average `weight * outcome`. It corrects action-selection imbalance but can have enormous variance with rare clicks or tiny propensities.
- **Self-normalized IPS:** divide weighted outcomes by the sum of weights. It can stabilize estimates but is generally finite-sample biased; normalization does not create support.
- **Doubly robust:** combine the target-policy expectation of the outcome model with a propensity-weighted residual on observed actions. Under overlap and standard identification conditions, correct propensities or a correct outcome model can protect consistency; neither protects unsupported new actions.

[Dudík, Langford, Li, ICML 2011](https://www.microsoft.com/en-us/research/?p=580345) establishes the direct/IPS/DR distinction. [Swaminathan and Joachims, ICML 2015](https://proceedings.mlr.press/v37/swaminathan15.html) motivates variance-aware counterfactual policy learning; [Thomas et al., AAAI 2015](https://ojs.aaai.org/index.php/AAAI/article/view/9541) motivates lower-confidence-bound evaluation rather than point-estimate promotion. [Offline Contextual Bandits with Overparameterized Models, ICML 2021](https://proceedings.mlr.press/v139/brandfonbrener21a.html) also warns that supervised generalization behavior does not automatically transfer to policy-based logged-feedback objectives.

**Proposed OPE audit:** report support violations, maximum/tail weights, effective sample size `(sum(weights))^2 / sum(weights^2)`, events by context/creative, clipping sensitivity, and uncertainty for each estimator. Weight clipping trades variance for bias and must be disclosed. Cross-fit nuisance models; reserve a final evaluation set separate from training and checkpoint selection. Account for clustered/repeated users, time drift, and adaptive logging in the uncertainty procedure rather than assuming all impressions are independent. A small effective sample size or near-zero event count should block a performance claim.

**Current hard limit:** propensity 1 for an operator-selected single candidate does not create overlap for alternative creatives. For an unseen generator-weight candidate, old impressions generally cannot identify its click rate nonparametrically. A new seed or continuous pixel array is not supported merely because a similar image appeared before. A model can extrapolate to it, but that is not validated OPE. Nor do internal denoising ratios retrospectively supply missing exposure propensities.

A practical route is to freeze a finite set of eligible generator-weight candidates, then, only after explicit authorization, randomize their exposure under known probabilities and evaluate on fresh matured outcomes. The action can be a versioned generator-plus-seed policy rather than a pixel array if its generation distribution and assignment are consistently defined. An OPE estimate from real logs still needs stated assumptions; a controlled online comparison is stronger evidence for lift. No external tracking, live advertising, or real personal data is authorized by this document.

## 8. Small-demo experiment and acceptance evidence

### Minimum comparison set

This matrix and the likelihood-specific checks below concern **native32**. For HD Sprint, use section 6.5's SCM-gradient, adapter, parent-replay, and full-resolution checks instead; do not apply DDIM likelihood tests or detach reward-input gradients in that direct-gradient training path.

| Arm | Does it change generator weights? | Purpose |
| --- | --- | --- |
| Frozen supervised student | No | Quality, reward, diversity, and parity reference. |
| Best-of-N / reranking | No | Detect whether expensive weight training improves on selection alone; not the requested deliverable. |
| Reward-weighted supervised fine-tuning | Yes | Simple reward-driven training baseline. |
| DDPO-style PPO plus reference/constraint controls | Yes | Selected genuine score-function experiment. |
| DRaFT/AlignProp-style direct gradients | Yes | Cost/variance comparator when all training surrogates are differentiable. |
| Diffusion-DPO | Yes | Optional when defensible comparison pairs exist; reward-generated pairs must be labeled synthetic, not human clicks. |

Compare with the same starting student, authorized training split, frozen reward models, candidate-selection protocol, and declared reward-query/compute budgets. Show total sampling/training wall time, actual hardware, peak memory, number of generated images and reward calls, optimizer steps, and rejected candidates. Do not import published “times faster” figures as measurements. Full convolution-weight updates are feasible to investigate without adding LoRA machinery; feasibility and resource use must be measured, not assumed.

### Required evidence before calling a candidate successful

1. **Actual learning:** weight/artifact hashes change, parameters stay finite, and multiple fixed-context/fixed-seed outputs change under deterministic inference. Reward-model-only, prompt-only, and seed-only changes do not qualify. Preserve the original reference artifact.
2. **RL numerics:** old/new replay ratios equal one before an update; recomputed likelihoods match stored actions; empirical nonterminal variance matches the sampler; terminal handling is explicit; all optimized transitions have valid densities. Verify gradient flow into generator weights while reward parameters and rollout actions stay detached.
3. **Signal controls:** compare no-update and shuffled-reward controls, plus ablations for reference KL and each constraint. Log reward components, gradient norms, conditional KL, clipping fraction, saturation, and reward variance. A nonzero gradient alone is not effectiveness.
4. **Deployment evaluation:** generate held-out paired contexts/seeds with **eta=0 DDIM16**, separately from stochastic training rollouts. Test raw/composited quality and brand fidelity, diversity, and deterministic renderer parity. Reject quality regressions even if stochastic-training reward rises. Do not use a stochastic evaluator to imply deterministic deployment success.
5. **Honest labels:** separate training-surrogate reward, held-out predicted click probability, synthetic realized click outcomes, human-review quality, and actual observed matured CTR. Report uncertainty and failed gates. There is no conversion-lift metric without actual attributable conversion data and a valid comparison design.
6. **No automatic deployment:** export an unpromoted candidate plus reference/reward/data/sampler provenance and explicit limitations. Keep graph compilation, teacher execution, supervised training, RL adaptation, evaluation, and deployment as separate stages.

### Claims this experiment can and cannot support

With synthetic outcomes only, it can demonstrate an exposure-aware training pipeline, a learned synthetic click predictor, genuine generator-weight policy updates, and measured acceptance/rejection under specified synthetic and quality checks. A simulator held out from the learned reward can test internal generalization to that simulator, but it remains synthetic evidence.

With authorized observational clicks, it can train and evaluate a predictor on the logged population and propose new weight candidates. It cannot infer new-candidate lift merely because predicted reward increases. Human ratings can support a quality finding, not a click or conversion finding. Only an appropriately identified and uncertainty-qualified real-outcome evaluation can support a corresponding real-world effectiveness claim.

**Bottom line:** DDPO-style PPO is the selected **native32 RL regression**, not the HD solution. **Pinned SanaSprint/SCM plus LoRA and direct reward backpropagation** is the coherent proposed HD adaptation; it is not likelihood-ratio RL or Flow-GRPO. Both require validated reward models and independent quality/brand gates, and neither implies online CTR or conversion lift. The deliverable is an auditable generator-weight candidate and evidence trail, not a universal algorithm ranking or a paid-job authorization.
