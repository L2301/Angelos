# Native32 RL: operator guide

Code and existing execution report reviewed **2026-10-08**. This guide describes
[rl.py](../ml/angelos_ml/rl.py), [rl_policy.py](../ml/angelos_ml/rl_policy.py),
[rewards.py](../ml/angelos_ml/rewards.py), and
[feedback.py](../ml/angelos_ml/feedback.py), not a proposed future trainer.

**Scope:** real parameter updates to the native **32×32** diffusion student,
using DDPO-style PPO as a numerical algorithm regression. This is **never the
target HD campaign model**, even if its output is enlarged. Aurora is a fictional
benchmark. A learned click score is not measured CTR, conversions, or evidence of
successful personalization. Candidates are never automatically promoted.

**No paid jobs are authorized.** This documentation work starts no training,
downloads, cloud jobs, advertising calls, or recurring jobs. The commands below
are explicit local operator workflows, not records of commands executed here.

## Existing corrected execution

The user supplied the completed
[corrected smoke report](../artifacts/rl-smoke-corrected-20261008/experiment.json).
This review read that file; it did **not** execute or independently reproduce the
experiment. Its `createdAt` is `2026-10-08T20:08:37.882543+00:00`, campaign `aurora`,
with two rounds and two epochs per round for each candidate. Both histories have
four entries and no rollback. Both candidates have nonzero weight changes, but
**both are rejected**.

| Report field | Candidate 1 | Candidate 2 |
| --- | --- | --- |
| File | `candidate-1.student.json` | `candidate-2.student.json` |
| `weightDeltaL2` | `0.0003980996261816472` | `0.00040028090006671846` |
| `conservativePredictedClickDelta` | `-0.08449498051777482` | `-0.0855987707618624` |
| Final `referenceKlPerCoordinate` | `4.896037561744985e-06` | `1.4914196844036572e-06` |
| Audit samples | `4` | `4` |
| `eligibleForFurtherTesting` | `false` | `false` |
| `status` | `rejected` | `rejected` |

The exact gate outcomes are identical for both candidates:

- Failed: `qualityFloor`, `brandFloor`, `conservativePredictedClickGain`,
  `notSyntheticEvidence`.
- Passed: `qualityNonRegression`, `brandNonRegression`, `rewardGeneralization`,
  `nonzeroWeightUpdate`, `teacherNoiseNonRegression`, `deliveryBrandValid`.

The report explicitly records `fixtureOnly: true`, `feedbackSha256: null`,
`nativeResolution: 32`, `meets1080pRequirement: false`, `actualConversionsUsed:
false`, `autoPromoted: false`, and `observedCtrLift: null` for each candidate.
Synthetic reward validation beating a constant baseline does not override the
failed gates or create human-feedback evidence. Positive weight deltas establish
an update, not an improvement in real campaign quality.

## What is actually optimized

The paper-level basis is denoising as a sequence of policy decisions in
[Black et al., *Training Diffusion Models with Reinforcement Learning*](https://arxiv.org/abs/2305.13301)
(submitted 2023-05-22; revised 2024-01-04), with a clipped probability-ratio
surrogate from [Schulman et al., *Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347)
(submitted 2017-07-20; revised 2017-08-28). The implementation identifies itself
as `ddpo-style-ppo-leave-one-out-v1`; this is not a reproduction of either paper's
image-quality results.

1. Each candidate starts from a fresh copy of the same parent. A separate parent
   copy stays frozen as the reference throughout optimization; candidates are
   not chained together.
2. Each round samples four trajectories per training profile by default, using
   stochastic DDIM with `eta=1`. There are **15 Gaussian-density transitions** in
   the 16-step schedule. Gaussian log probabilities sum over image coordinates;
   detached rollout actions and old log probabilities are retained for PPO.
3. The final deterministic denoising step has zero variance, so it is **not**
   assigned a Gaussian action density. It uses the **frozen reference model**,
   making the terminal reward map independent of candidate parameters for the
   likelihood-ratio update. The report calls this a terminal decoder; here it
   means the final native32 denoiser, not an HD VAE.
4. Delivered images, including configured brand compositing, receive frozen
   reward-model scores. The terminal reward is detached. Within each profile's
   sample group, each advantage is its reward minus the mean reward of the
   **other** samples. This leave-one-out baseline is not a learned value network,
   GAE, or group-standard-deviation normalization.
5. PPO uses the clipped surrogate, a fixed-parent KL penalty, and a teacher
   noise-prediction anchor. The same rollout is reused for the configured epochs;
   the next round collects a new rollout. Gradients are clipped to norm 1.
6. Post-update KL above the limit, or a nonfinite measured KL, restores the
   candidate's **start-of-round weights** and ends optimization. Reported KL is
   the mean conditional Gaussian KL **per image coordinate on visited rollout
   states**, averaged across transitions, not total trajectory KL or a general
   safety guarantee.

Deployment-style auditing is deliberately different: deterministic DDIM16,
`eta=0`, with the candidate's own weights at **every** step, compared with the
parent on matched profiles and seeds. Training reward therefore cannot substitute
for the candidate audit. History `surrogateReward` is the sampled pre-update
reward, not a fresh post-update quality measurement. `ppoLoss` includes the KL
penalty but excludes the separately backpropagated teacher anchor.

## Click predictor and quality/brand critic

These are separate frozen CPU float32 predictors, **not an RL state-value
critic**. Each uses three profile-bootstrap heads. Features consist of 12
centered RGB values from adaptive 2×2 pooling, four centered UPL values, and 48
image/UPL interactions: 64 features total. Click fitting uses binary
cross-entropy; review fitting uses normalized quality MSE plus brand BCE.

Operator `qualityRating` values 1–5 become `(rating - 1) / 4`. The brand target
is boolean `brandPass`; an optional stored `brandRating` is not used here.
Prediction returns means and population standard deviations across heads for
click, quality, and brand. **Bootstrap-head spread is a heuristic, not a
calibrated confidence interval or a lower confidence bound.**

With defaults, optimization uses:

```text
conservative_score = clamp(predicted_mean - predicted_std, 0, 1)
reward = conservative_click
         - 5 * (max(0, 0.6 - conservative_quality)
                + max(0, 0.8 - conservative_brand))
```

These are soft optimization penalties; separate audit floors are hard candidate
gates. A high predicted click score cannot override a failed audit floor.
Although the reward API accepts image dimensions from 16 to 4096, pooling erases
fine detail. It cannot certify product identity, logo geometry, typography,
legibility, or HD realism. The reward API preserves image gradients for other
callers, but this PPO path uses detached sampled rewards, not backpropagation
through the reward model and denoising trajectory.

No actual candidate click experiment, causal CTR estimator, propensity-weighted
off-policy evaluation, or conversion objective is implemented by this runner.

## Inputs, provenance, and holdouts

- **Parent:** an existing local `angelos.diffusion.v1` native32 student JSON,
  compatible with `DiffusionStudent.from_artifact`. This CLI does not accept
  Sana, FLUX, Qwen, or another pretrained HD checkpoint.
- **Teacher anchor:** an existing compatible NPZ and adjacent
  `NAME.manifest.json` for `NAME.npz`. Campaign ID and recomputed graph hash must
  match the parent. The dataset loader validates native32 images, normalized
  four-dimensional UPLs, train/held-out splits, teacher seeds, and provenance.
  Procedural fixture provenance is allowed only in explicit synthetic mode.
- **Feedback:** exactly one of `--synthetic-demo` or `--feedback FILE` is
  required. Normal feedback must have format `angelos.feedback.v1`, outcome
  `click`, and population `local-synthetic-adults`. It is not authorization to
  ingest real personal data or evidence from real customers.
- **Freshness:** normal feedback's `asOf` must be no more than one hour old when
  training starts and cannot be in the future. Export again rather than editing
  the timestamp. Freshness is checked at startup, not continuously during a run.
- **Evidence:** normal mode requires **at least 100 operator review rows**.
  This is not 100 independent raters or a statistical power calculation. The
  reward metadata's 100–1000 review target is not proof those reviews exist.
  Synthetic review rows still fail normal reward fitting even if 100 operator
  rows are also present; never relabel fabricated reviews to bypass the gate.
- **Reconstruction:** retain every exposure's exact model artifact. `--parent`
  is included automatically; repeat `--artifact` for other historical models,
  without duplicating the parent's ID. Raw-file SHA256, artifact ID, campaign,
  graph hash, and encoder are checked. Creative identity binds artifact ID,
  exposure snapshot ID, and seed. Reconstruction uses the saved exposure UPL
  and schedule, **never the latest profile or latest model**.
- **Labels and consent:** export uses eligible mature click labels and eligible
  reviews with current consent filtering. The materializer rejects duplicate
  impression/review IDs and rows newer than the cutoff. Consent and UPL encoder
  versions are independent. Hashes establish artifact bindings, not authentic
  human judgments; Python trusts the local export and does not re-query consent.
  Later revocations are not detected by a saved export.

Reward fitting makes one seeded, label-blind split of the **union of profile
IDs** across click and review records, approximately 25% validation (at least two
validation groups). For **each task**, defaults require at least eight training
records from two profiles and four validation records from two profiles. Both
click partitions need positive and negative labels. Training reviews need
variation in both quality and brand targets. An insufficient split fails; do
not search seeds for a favorable holdout.

Policy optimization uses up to the first eight unique reward-training profiles,
four samples each, with at most 32 rollout samples. Audit uses up to the first
16 reward-validation profiles, with at least two groups. A profile contributes
its first encountered UPL in this selection. This is profile-separated, not
necessarily creative-disjoint or representative of all profile snapshots.

The teacher anchor samples four **training-split** examples per round for
epsilon MSE. A fixed teacher held-out noise bank compares parent and candidate;
repeating held-out rows to fill a bank does not create independent evidence.
Teacher holdout separation and reward-profile separation are different checks:
the runner does not certify a globally disjoint cross-dataset population.
All candidates reuse the same reward audit profiles/seeds and teacher noise
bank, so selection across candidates consumes that holdout. Reserve fresh
confirmatory evidence before making further claims.

## CLI workflows

Run from the repository root with the existing local `.venv` and dependencies.
[scripts/rl.js](../scripts/rl.js) sets `PYTHONPATH=ml`, `HF_HUB_OFFLINE=1`, and
`TRANSFORMERS_OFFLINE=1`, and invokes `.venv/bin/python -m angelos_ml.rl` without
a shell. It does not schedule anything. The following input paths are
**placeholders** for already-existing compatible files; output directories must
not exist. No installation or model download is implied.

### Synthetic algorithm smoke test

```sh
npm run rl -- \
  --parent path/to/native32-parent.student.json \
  --dataset path/to/native32-teacher.npz \
  --output artifacts/rl-my-new-smoke \
  --synthetic-demo --rounds 2 --candidates 2 --device cpu
```

This mode fabricates 128 click rows and 128 review rows over 16 synthetic
profiles using uniform 32×32 brightness images. Labels are brightness rules,
not photographs, people, or observed clicks. It can exercise actual parameter
updates, but `notSyntheticEvidence` is always false, so it cannot produce an
eligible candidate. A rejected smoke result can still be a useful algorithm
regression result; it is not a quality milestone.

### Fresh local feedback experiment

First collect eligible local-ledger exposures and genuine operator ratings for
the fictional benchmark. The export command needs existing
`data/angelos.sqlite` and `data/campaign.ang`; it uses the active campaign and
creates a **new file** in an existing directory. Exporting does not train or
promote anything.

```sh
npm run feedback:export -- artifacts/feedback-my-new-export.json
npm run rl -- \
  --parent path/to/native32-parent.student.json \
  --dataset path/to/native32-teacher.npz \
  --feedback artifacts/feedback-my-new-export.json \
  --artifact path/to/other-exposure-model.student.json \
  --output artifacts/rl-my-new-feedback-experiment \
  --rounds 4 --candidates 2 --reward-steps 200 --seed 2026 --device cpu
```

Omit `--artifact` if all exposures use the parent; add one per additional model
otherwise. Do not add `--synthetic-demo` to a feedback run. Missing models,
insufficient reviews or class support, stale consent exports, fixture teacher
data, or identity mismatches are reasons to stop and correct the evidence, not
to relax the gates.

| CLI control | Default / boundary |
| --- | --- |
| `--rounds` | `4`; integer 1–100 |
| `--candidates` | `2`; integer 1–4, independent copies of parent |
| `--reward-steps` | `200`; positive integer for reward fitting, not PPO rounds |
| `--seed` | `2026`; uint32 |
| `--device` | `cpu`; also local `mps` or `cuda` when available |
| `--quality-floor` | `0.6`; normalized range [0,1] |
| `--brand-floor` | `0.8`; normalized range [0,1] |

Other `PolicyConfig` defaults are not CLI flags: group size 4, epochs 2,
learning rate `1e-6`, PPO clip `0.2`, KL weight `0.1`, maximum per-coordinate KL
`0.01`, critic penalty `5`, uncertainty penalty `1`, teacher-anchor weight `0.1`.
CPU is the default; selecting MPS is not a claim that this guide benchmarked it.
Reward fitting remains on CPU. Native32 memory behavior says nothing about HD
training fitting in 24 GB of shared Apple Silicon memory.

## Gates and output interpretation

All ten gates must pass for `eligibleForFurtherTesting: true` and status
`eligible-for-manual-test`. Neither field means deployable or promoted.

| Gate | Implemented condition |
| --- | --- |
| `qualityFloor` | Minimum candidate quality mean minus weighted std across audit profiles is at least the configured floor. |
| `brandFloor` | Same minimum test for brand. |
| `qualityNonRegression` | Mean candidate quality prediction is at least parent's. |
| `brandNonRegression` | Mean candidate brand prediction is at least parent's. |
| `conservativePredictedClickGain` | Mean candidate click minus mean parent click minus uncertainty penalty times the sum of their mean stds is strictly positive. |
| `rewardGeneralization` | Held-out click **Brier**, quality **MSE**, and brand **BCE** each strictly beat the corresponding constant training-label baseline. Click BCE is reported but is not this runner's gate. |
| `nonzeroWeightUpdate` | Final weight L2 delta from parent is positive. |
| `teacherNoiseNonRegression` | Candidate held-out epsilon MSE is at most `1.05 × parent MSE`; permits up to 5% worsening. |
| `deliveryBrandValid` | Configured raster/composite brand error is within tolerance; automatically true if there is no configured brand. |
| `notSyntheticEvidence` | Experiment is not `--synthetic-demo`. |

Audit floors use the raw mean-minus-std expression, unlike the clamped
optimization reward. Neither learned brand prediction nor an exact compositing
check establishes overall product fidelity or readable generated text.

The new output directory contains:

- `rewards.json`: `angelos.rewards.v1` frozen predictors, split IDs, source/count
  metadata, held-out metrics, and content-derived reward identity.
- `candidate-N.student.json`: candidate layers and provenance, **even when
  rejected**. Training metadata includes parent ID/hash, reward and feedback
  IDs/hashes, seed, device, native resolution, fixture flag, and `candidate-only`
  promotion status. `training.steps` counts history entries, not images.
- `experiment.json`: written last, with parent/dataset/feedback SHA256 values,
  graph/campaign IDs, candidate hashes, configuration, histories, matched audit
  scores, gates, limitations, and explicit `autoPromoted: false`.

Common input/runtime failures exit with code 2. Gate rejection is instead a
completed experiment with rejected candidate statuses; a zero process exit code
does **not** mean a candidate passed. A failure after directory creation can
leave partial files. Require a complete report and inspect it rather than
treating the existence of a candidate file as success. Use a new output path for
a deliberate rerun; do not overwrite exposure-bound parent bytes or publish a
candidate merely because its JSON exists.

## Manual nighttime operation, not automatic promotion

1. Before each individually authorized local run, freeze parent, graph, dataset,
   feedback provenance, seed, evidence mode, and bounded candidate/round counts.
   Export fresh consent-filtered feedback immediately before a normal run.
   Preserve the teacher manifest alongside the NPZ and keep a record of the code
   revision/environment; the experiment report alone is not a complete replay
   package.
2. Run once, inspect completion/errors and every gate, and retain rejected
   results. There is no nightly scheduler, cron loop, automatic retraining,
   automatic candidate selection, or deployment action in this CLI. Calling a
   manually launched experiment a "night run" does not authorize recurring work.
3. Review matched parent/candidate images and score failures. Do not lower floors,
   relabel synthetic records, or repeatedly tune against the same audit set to
   manufacture a pass. Check diversity, personalization, quality, and brand
   constraints independently of the pooled critic.
4. Even if all gates pass, recheck consent and provenance, reserve fresh
   profile/creative evidence for confirmatory evaluation, and require explicit
   human approval for any separate test or deployment. Keep the frozen parent
   available for rollback. No paid jobs or real-user advertising experiment is
   authorized by this guide.
5. Report learned-score changes as **surrogate prediction changes**, never CTR
   lift. Actual uplift needs a separately designed, authorized observed-outcome
   experiment; neither synthetic smoke evidence nor the current audit supplies it.

## Research, HD boundary, and checks

- [RL research](RESEARCH-RL.md) covers broader algorithm choices; its research
  discussion is not a list of capabilities implemented by this CLI.
- [Diffusion research](RESEARCH-DIFFUSION.md) covers pretrained HD model selection.
  A successful native-HD inference image does not establish an RL-trained campaign
  student, successful personalization, or any reward/CTR improvement.
- [HD alignment](HD-ALIGNMENT.md) documents a **separate direct-gradient reward
  optimization** path using the provided SanaSprint pipeline's actual
  `SCMScheduler` consistency sampler and small transformer adapters. It is not
  this stochastic DDIM/PPO algorithm, not Flow-GRPO, and not a value-function
  method. Shared reward code does not transfer native32 success into evidence of
  HD adaptation or HD brand fidelity.

For a code regression check, use the existing targeted suites; these commands
are not new experiment evidence:

```sh
npm run rl -- --help
PYTHONPATH=ml .venv/bin/python -m unittest discover -s ml/tests -p 'test_rl_policy.py'
PYTHONPATH=ml .venv/bin/python -m unittest discover -s ml/tests -p 'test_rewards.py'
PYTHONPATH=ml .venv/bin/python -m unittest discover -s ml/tests -p 'test_feedback.py'
```

This documentation review verified the CLI help and inspected the existing
corrected report. It did not rerun training or these test suites. Numerical
regression tests are not perceptual-quality, brand-legibility, or CTR benchmarks.
