# HD consistency-model reward alignment

`ml/angelos_ml/hd_alignment.py` adapts the **provided, already-loaded** pinned
SanaSprint 0.6B transformer with small q/v LoRA weights. It uses the actual
**SCMScheduler consistency sampler**, not DDIM, Flow-GRPO, Gaussian likelihood
ratios, PPO, or an RL value function. This is **direct reward optimization**.
The 0.6B label describes the transformer, not the entire Gemma/decoder pipeline.

No downloads, paid jobs, cloud provisioning, external trackers, or actual
foundation-weight execution are performed by this documentation/test work.
Native32 DDPO remains a separate algorithm regression, not HD evidence.

## Caller contract

```python
from angelos_ml.hd_alignment import AlignmentConfig, align_pipeline

result = align_pipeline(
    pipeline,
    train_cases,
    audit_cases,
    rewards,
    output="ml/artifacts/my-new-hd-alignment",
    config=AlignmentConfig(steps=4, sampling_steps=2, rank=2, blocks=2),
    seed=2026,
)
report = result["report"]
adapter_state = result["adapter_state"]
```

- `pipeline`: exact Diffusers **0.40.0** `SanaSprintPipeline`, with
  `SanaTransformer2DModel`, `SCMScheduler(prediction_type="trigflow")`,
  `AutoencoderDC`, and its text encoder. Use the existing `hd_models.load_pipeline`
  result, already on MPS or CUDA. No offload hooks or existing PEFT adapters.
  The loader snapshot must be the pinned official revision
  `aa76e7f4f4928f378716b6716a2130fba3caf5b1`; this check is a provenance contract,
  not independent attestation of every weight byte.
- Cases are nonempty lists of dictionaries containing `id`, `profileId`,
  `prompt`, normalized `upl` of length four, and uint32 `seed`. At most **eight
  total** train plus audit cases. IDs are unique; profile groups, IDs, and seeds
  are disjoint between train and audit. Prompts may repeat across independent
  profiles. UPL enters the reward model; generator conditioning is the caller's
  supplied prompt, not a newly invented native UPL input to Sprint.
- Generator-training profiles cannot be reward-validation profiles. Audit
  profiles cannot be reward-training profiles; fresh audit groups are preferable.
  Membership in the corresponding reward split is **not** required: unseen
  training and unseen audit profiles are allowed. This is extrapolation by the
  frozen reward predictor, not evidence of calibration on those profiles.
- `rewards`: the learned CPU `RewardBundle`. Export/identity/metadata validation
  must pass. Held-out click BCE and Brier, quality MSE, and brand BCE must each
  beat their corresponding training-intercept baseline **before** any sampling.
  Real alignment requires nonsynthetic evidence, observed ledger clicks, and
  **at least 100 operator quality/brand reviews in total**, with operator reviews
  in both reward-data partitions. This count is an evidence minimum, not a power
  calculation or HD-fidelity guarantee. Missing or bad
  evidence blocks training rather than being replaced by favorable constants.
- `config.steps`: **1–20 optimizer updates**, four by default; each update uses
  one train case in deterministic round-robin order. Sampling uses 1–4 SCM
  steps, two by default. Rank 1–8, first 1–4 transformer blocks, and at most
  one million adapter parameters. No convergence loop or audit-based early stop.
- `seed` initializes adapters only. Case seeds initialize CPU RNGs for the
  sampler on every call, including matched parent/candidate re-noising. CPU RNG
  and explicitly copying images to the CPU reward model are not GPU model
  fallback. Do not call inside `torch.inference_mode()`.

The result contains CPU `adapter_state` tensors plus a JSON-compatible `report`.
The supplied pipeline's original modules, parameter trainability, and module
training flags are restored even on errors. Calls are not concurrency-safe on
the same pipeline; its scheduler is reset by each sampler call. The caller must
not mutate the pipeline/rewards during the experiment.

## Faithful gradients and bounded updates

The implementation unwraps the installed official `SanaSprintPipeline.__call__`
no-grad decorator, requests `output_type="latent"`, and leaves SCM timestep,
guidance, sigma scaling, output conversion, and re-noising arithmetic intact.
It then explicitly differentiates through
`vae.decode(latents / vae.config.scaling_factor)` and tensor RGB normalization.
There is no PIL conversion, byte quantization, or detached image feature path.
Only frozen text conditioning and the adapter-disabled parent pass use no-grad.
Frozen decoder/reward **weights** do not disable their input gradients.
The supplied decoder's official tiling settings are retained and recorded;
offline regression also checks gradients through its tiled decode/blending path.

Production sampling/decoding is **1920×1088**, with resolution binning disabled.
The displayed/reward-scored tensor is a crop only: columns 0–1919, rows 4–1083,
giving **1920×1080**. No native32 or 1024 image is upscaled and relabeled HD.
The native latent/RGB shapes are validated. OOM or unsupported device operations
propagate explicitly: no CPU model fallback, automatic resizing, or retry.

The objective minimizes negative learned click mean-minus-disagreement,
separate quality/brand hinge costs, matched-seed parent latent/image MSE, and
adapter L2. “LCB” in metrics is a **heuristic conservative score**, not a
calibrated lower confidence interval. The same conservative scoring convention
is used for audit. Anchoring is not KL and is not a constraint guarantee.

`BoundedLoRALinear` wraps only self-attention `attn1.to_q` and `attn1.to_v` in
the configured initial blocks. It adds a float32 `up(down(input))/rank` residual
to the frozen original Linear; its result is cast to the original output dtype.
Down weights use seeded initialization, up weights start at zero. Gradients are
clipped to norm one; adapter elements are clamped to the configured bound.
Existing base weights never enter the optimizer. A parameter-version check
detects unexpected base mutation without copying the whole foundation model.
Tests additionally compare tiny base state dictionaries byte-for-byte.

## Audit and artifact contract

Audit cases are sampled **once after all updates**, parent and candidate at the
same case noise. They never receive gradients, choose a checkpoint, tune a
penalty, or restart training. Each case must meet quality and brand floors and
must not regress versus its parent (1e-6 numerical tolerance). Average predicted
click conservative-score gain must be positive and meet the configured minimum.
Adapter change, base preservation, and reward-validation gates are also required.

Passing yields `status="surrogate-gates-passed"`; failure yields `"rejected"`.
Both remain `autoPromoted=false`, `observedCtrLift=null`,
`conversionLift=null`, and `hdQualityEstablished=false`. Full-resolution human
and exact brand/text/product checks remain necessary: the existing 2×2 pooled
reward model cannot certify those details. The numerical brand gate is a learned
surrogate gate, **not an implemented exact-brand certification**. Any real
deployment gate must reject failures of those additional checks. A fitted
observational predictor and its gradient do not establish new-candidate lift.

The bundle's metadata does not establish source-image resolution coverage.
`rewardResolutionCoverage="unverified"` explicitly records this mismatch: a
64-feature pooled critic fitted on low-resolution images is not validated for
fine HD detail merely because it accepts a 1920×1080 tensor. Separate
`deploymentGates` for `hdRewardValidation`, `humanFullResolutionReview`, and
`exactBrandReview` remain **false**, and `deploymentEligible=false` regardless
of the surrogate result. Clearing them requires new, independently recorded
HD-domain reward evidence and full-resolution human/exact-brand review outside
this experiment; this API does not invent or automatically clear those gates.

`output` must be a new directory. After successful execution (including a
rejected candidate), it receives:

- `adapter.safetensors`: `angelos.sana-qv-lora.v1` tensors, keyed by exact target
  path plus `.down`/`.up`; no base weights, optimizer, or pickle.
- `alignment.json`: `angelos.hd-alignment.v1`, parent revision, config, sampler,
  dimensions, device, reward identity/evidence, case split/hash, adapter targets,
  file checksum, update trace, independent audit, and explicit limitations.

This custom adapter format is **not PEFT/Diffusers LoRA interchange**. For a
trusted saved artifact, reconstruct its `AlignmentConfig` and use
`adapter_scope(pipeline.transformer, config, state=adapter_state)` as a context
manager, then invoke the ordinary inference pipeline. Validate artifact format,
parent revision, and checksum first; the context validates exact tensor keys,
shapes, finiteness, and parameter bounds. It restores original q/v modules on
exit. `output=None` returns the same candidate in memory without writing files.
Files are never overwritten; disk failures can leave an incomplete new directory,
which must not be treated as a completed candidate.

## Offline tests, not a pretrained HD run

```sh
PYTHONPATH=ml .venv/bin/python -m unittest discover -s ml/tests -p test_hd_alignment.py -v
```

Tests use tiny randomly initialized **real** SanaTransformer, AutoencoderDC,
and SCMScheduler modules, no downloads, and explicit `fixture_only=True`.
Fixture-only precomputed prompt embeddings avoid loading a text model;
production forbids them. Fixture output is 32×32 (optionally 64×64), with
`fixtureOnly=true` and no manual-review eligibility. It cannot qualify the HD
milestone. Tests cover official sampler/decoder parity at 1/2/4 steps, finite
difference reward-path gradients, actual adapter changes, unchanged base weights,
held-out isolation, critic rejection, fail-closed OOM, and safetensors replay.

## Primary sources

- [SANA-Sprint, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Chen_SANA-Sprint_One-Step_Diffusion_with_Continuous-Time_Consistency_Distillation_ICCV_2025_paper.pdf): consistency-distilled generator, not an unmodified base-flow policy.
- [Official pinned model metadata](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers/blob/aa76e7f4f4928f378716b6716a2130fba3caf5b1/model_index.json).
- [Official Diffusers 0.40.0 Sprint pipeline](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/pipelines/sana/pipeline_sana_sprint.py) and [SCM scheduler](https://github.com/huggingface/diffusers/blob/v0.40.0/src/diffusers/schedulers/scheduling_scm.py): the reused sampler and decoder scaling contract.
- [LoRA](https://arxiv.org/abs/2106.09685): frozen-base low-rank weight adaptation; this bounded custom wrapper is not a reproduction of that paper's benchmarks.
- [DRaFT, ICLR 2024](https://arxiv.org/abs/2309.17400): direct differentiable-reward fine-tuning precedent, not proof of efficacy on Sprint.
- [AlignProp status](https://arxiv.org/abs/2310.03739): withdrawn as subsumed by [VADER](https://arxiv.org/abs/2407.08737); no standalone current-publication claim.

The wider algorithm comparison and real-outcome/OPE boundaries are in
[RESEARCH-RL.md](RESEARCH-RL.md). No globally best method, measured HD quality,
observed CTR improvement, or conversion lift is asserted here.
