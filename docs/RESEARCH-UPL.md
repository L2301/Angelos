# Advertising identity, user representations and UPL v2

Research checked 2026-10-08. This review motivated the [UPL v2 implementation](UPL.md):
richer synthetic ingredients, expiring intent events, offline semantic encoding,
versioned snapshots and ComfyUI handoffs. The broader evaluation plan below is
not a claim that a behavioral encoder or HD conditioning adapter has been trained.
No tracking integration, real personal-data collection or paid job is authorized.
The existing four-coordinate demo remains compatible as an explicitly legacy path.

## 1. The cookie is not the representation

Google documents multiple advertising cookies, not one universal profile cookie.
For example, IDE supports advertising on non-Google sites; NID can support ads
for signed-out users on Google services; DSID relates to signed-in users and their
personalization setting. `_gcl_` cookies primarily support conversion measurement.
Their purposes differ. [Google cookie documentation](https://policies.google.com/technologies/cookies?hl=en_gl).

The most useful technical account is Google's Authorized Buyers documentation:
cookie matching associates a bidder's identifier with a bidder-specific Google
User ID using match tables. Browser origin restrictions prevent simply reading
another domain's cookie. This is an identity-association mechanism, not an API
that returns Google's private learned user embedding. Access also requires an
appropriately configured buyer account.
[Cookie Matching](https://developers.google.com/authorized-buyers/rtb/cookie-guide).

Thus the useful design analogy is **identifier → authorized state lookup →
features/representations → prediction**, not a cookie containing a complete
personality vector. This is an architectural inference from the documented
interfaces, not a claim to reconstruct Google's proprietary production system.
The reviewed public sources do not establish a single current Google Ads UPL
schema, embedding width, training objective or universal user model.

Independent measurement research also distinguishes identifiers from the data
that can be linked through them. Papadopoulos, Kourtellis and Markatos studied
cookie synchronization and its cross-domain linkability in a historical mobile
browsing dataset. Its findings are not a measurement of 2026 browser behavior.
[Cookie Synchronization, WWW 2019](https://arxiv.org/abs/1805.10505).

### Current browser-platform caveat

Old FLoC/Topics/Protected Audience tutorials are not a sound new dependency plan.
Google announced retirement of several advertising-related Privacy Sandbox
technologies in October 2025. Its status page, updated August 2026, lists Topics,
Protected Audience and Attribution Reporting for deprecation/removal; CHIPS and
FedCM remain supported categories. This does not mean that all cookies or all
Google advertising infrastructure have disappeared, nor that third-party
identity is universally available.
[Announcement](https://privacysandbox.google.com/blog/update-on-plans-for-privacy-sandbox-technologies),
[current status](https://privacysandbox.google.com/overview/status).

## 2. Research that matters to Angelos

Google work below is distinguished from other companies' work. Recommendation,
ranking and language-generation results are not proof of personalized image
generation quality or causal advertising lift.

| Work | Public finding | Implication for our design |
| --- | --- | --- |
| [Google: Ad Click Prediction, KDD 2013](https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/) | Deployed sponsored-search CTR modeling with sparse features, online FTRL updates and attention to calibration/engineering. | Separate observed events, prediction, model updating and evaluation. A cookie identifier is not a prediction model. |
| [Google: Wide & Deep, 2016](https://research.google/pubs/wide-deep-learning-for-recommender-systems/) | Combines sparse feature interactions with learned dense representations in Google Play recommendations. | Keep explicit facts alongside embeddings rather than compressing everything irreversibly into one opaque vector. |
| [Google: DCN V2, WWW 2021](https://arxiv.org/abs/2008.13535) | More expressive feature-cross modeling with reported deployments in Google ranking systems. | User state must interact with the particular creative, campaign and context. Width alone is not the modeling objective. |
| [Google: USER-LLM, 2024](https://arxiv.org/abs/2402.13598) | Encodes interaction histories and conditions an LLM through cross-attention; evaluates personalization on public recommendation/review datasets. | Strong architectural precedent for a separately trained user encoder plus a model-specific conditioning adapter. It is not a disclosed Google Ads cookie implementation. |
| [Google researchers: User Embedding Model, PERSONALIZE 2024](https://aclanthology.org/2024.personalize-1.12/) | Compresses free-form user histories into embeddings used as language-model soft prompts. | Relevant to the proposed JSON/text bootstrap, while distinguishing learned conditioning from concatenating raw JSON into a prompt. |
| [Meta: Scaling User Modeling / SUM, 2023–2024](https://arxiv.org/html/2311.09544v2) | Shares upstream user representations across ads models. Section 5.3 reports two 96-dimensional embeddings per user model, stored in FP16. Its serving system asynchronously refreshes representations; the paper discusses staleness and encoder distribution shift. | Particularly close to a reusable, live UPL service. The reported width is an existence proof, not Google's width or the optimum for Angelos. |
| [Alibaba: DIN, 2017–2018](https://arxiv.org/abs/1706.06978), [DIEN, AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/download/4545/4423) | DIN makes interest representations candidate-dependent; DIEN models their temporal evolution from behavior sequences. | Preserve multiple interests and recent intent; avoid treating all clicks as permanent traits or forcing every campaign to see one identical summary. |
| [Tencent: LFM4Ads, 2025 preprint](https://arxiv.org/abs/2508.14948) | Transfers user, item and user-item interaction representations rather than only a user vector. | A reminder that the creative/campaign side of the interaction also needs representation. Not an implementation to import wholesale into this demo. |
| [Matryoshka Representation Learning, 2022](https://arxiv.org/abs/2205.13147) | Trains representations with useful nested dimensionalities. | Evaluate bandwidth/quality trade-offs using an encoder trained for them; arbitrary truncation of an unrelated vector is not equivalent. |

## 3. Why the present 4D value is insufficient

The legacy repository path does not learn a rich four-dimensional representation.
`src/core/store.js` directly maps adult age band, coarse region, openness and
energy to four normalized numbers. ComfyUI validates that same deterministic
mapping. The image student and reward model assume that exact contract.

This is useful for regression tests but omits behavioral history, uncertainty,
multiple interests, changing intent and richer declared preferences. Moreover,
normalizing region enum indices introduces arbitrary numeric distances; categorical
features should not acquire semantic ordering merely from their position in a list.

Expanding these same four facts into 512 numbers would not add observations about
the person. A larger encoder may express useful priors or nonlinear interactions,
but it cannot establish unobserved personal facts. The replacement needs **richer
authorized inputs, an appropriate learning objective and adequate output capacity**.
Dimension is a parameter to evaluate, not an information-quality certificate.

## 4. Proposed UPL v2: a versioned bundle, not a universal fixed vector

Everything in this section is an Angelos engineering recommendation.

### Identity and ingredients remain independent

Keep identity resolution separate from the ML payload. An eventual first-party
cookie/session should hold an opaque, revocable reference—not raw facts or a
global vector copied to every advertiser. A shared internal user-owned state can
support campaign-scoped authorized views without exposing a cross-advertiser
global identifier. Browser restrictions mean cross-site identity needs an explicit
supported flow; it must not depend on silent universal third-party cookies.

Store typed, provenance-bearing ingredients: declared facts and preferences,
consented interaction events, observation time, source, confidence/uncertainty
where supported, expiry, and applicable permissions. Distinguish missing,
explicitly declined and observed values. Begin with synthetic adults as before;
do not infer sensitive traits or ingest real browsing histories for this task.

Facts-schema version, event-schema version, consent version and encoder revision
must be independent. A new encoder should not rewrite the underlying ingredients.

### Bootstrap with real semantic embeddings

For the first richer baseline, serialize the permitted ingredients using a stable,
auditable template and encode them with a pinned local text-embedding model.
Keep exact categorical/numeric facts as a parallel branch. Exclude opaque user
IDs from semantic text so similarity does not become accidental identity lookup.
Do not let a generative summary invent missing preferences.

A practical candidate is **Qwen3-Embedding-0.6B**, whose official card supports
32–1024 output dimensions and Matryoshka representations. This is a distinct
checkpoint from the existing Qwen3 text-generation node.
[Official model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

My provisional serving default is **512 semantic dimensions**, with controlled
comparisons at 128, 256, 512 and 1024. That is a starting hypothesis, not a result
established by Google or a guarantee that 512 beats 192. Retain independent facts
and recent-event information instead of declaring this one vector sufficient.
Pin model/tokenizer revision, text template, pooling, normalization, output width
and numeric format. Follow the checkpoint's supported dimension-reduction method.
Never pad, hash or randomly project four numbers and label that a trained UPL.

### Add temporal and multiple-interest representations

Once suitable event data exists, compare a sequence encoder against the semantic
bootstrap. Maintain a long-term representation and a separate short-term/session
representation or a small set of interest tokens. Their widths/token counts are
separate measured decisions, not mandatory global constants.

Learn using time-correct behavioral prediction and relevant preference outcomes;
compare against simple recency-weighted baselines. Keep observed negative feedback,
qualified unclicked exposures and absent observations distinct. Log the shown
creative/model and selection policy so the model does not treat its own serving
choices as an unbiased view of user preference.

An event can update stored state and trigger embedding inference without retraining
encoder weights. Weight training can remain separately scheduled or operator-run.
This distinction makes live UPL updates compatible with nightly model experiments.

### Adapt to the campaign and generator

Use a campaign-conditioned adapter over the available user representations,
campaign brief and current placement context. A declared hiking interest should
help an outdoor campaign without erasing the same person's other preferences.
Task-specific similarity should be measured; a generic text embedding is not
automatically an ad-response or visual-taste space.

For ComfyUI, keep human-readable fact/prompt selectors, and introduce a separate
typed embedding payload. A raw 512D vector is **not** a valid CLIP/SANA/FLUX
conditioning tensor. A model-specific learned projection/token adapter must be
trained and evaluated before exposing such conditioning; otherwise use the
explicit text-conditioning path and label it honestly. Opaque coordinate access
remains a numerical operation, not a way to decode named personality traits.

### Freshness, compatibility and privacy

Pin exposure-time fact/event snapshots and encoder artifacts. Record event cutoff,
embedding creation time, model revision, dtype, dimensions and permitted scope.
Use the last valid compatible representation only within an explicit freshness
budget; otherwise use a contextual/unpersonalized fallback. No hidden encoder
swap or zero-vector substitution for missing data.

Retraining can change the coordinate system even at the same width. Update or
validate downstream adapters with the encoder; do not blindly average vectors
from incompatible revisions. Train/audit splits must group stable internal
synthetic-person identity across changing facts, not just an audience-content hash.

Embeddings are not automatically anonymous. Google's research explicitly studies
re-identification risk from compact user representations. Access control,
purpose-limited views, retention limits, deletion propagation and evaluations
against linkage/inference attacks are needed independently of dimensionality.
[Measuring Re-identification Risk, SIGMOD 2023](https://research.google/pubs/measuring-re-identification-risk/).
Removing source data does not prove its influence has been removed from weights.

## 5. Migration and acceptance gates

Do not change a single `dimension: 4` constant and call this an upgrade. Current
contracts are spread across the store/graph, ComfyUI exporter/importer, dataset
manifests, Python diffusion sampler, reward features, HD alignment and WebGPU
artifact layout. Keep `structured-demo-v1` as a legacy regression fixture.

1. Define a new, bounded UPL bundle schema and trusted encoder registry. Support
   declared variable dimensions/token shapes, signed finite values, explicit
   missingness and immutable encoder identities. Reject mismatched payloads.
2. Extend ComfyUI import/export and local handoff without breaking pack v1. Preserve
   provenance; require explicit supported adapters rather than accepting arbitrary
   model IDs/code URLs or silently truncating vectors.
3. Implement and measure a real pinned semantic encoder on a richer synthetic
   ingredient corpus. Document that semantic clustering alone is not ad effectiveness.
4. Introduce a new conditioning/student artifact version and reward feature
   contract. Retrain compatible weights rather than reusing the old 4D model.
5. Compare no-UPL, existing 4D, explicit facts, semantic embeddings at several
   widths, and facts plus temporal/multi-interest representations. Split by person
   and time before fitting; use independent campaign/creative holdouts where feasible.
6. Measure held-out click log loss/Brier score, calibration, preference prediction,
   conditioning sensitivity, counterfactual stability, full-resolution brand/quality
   constraints, freshness/latency, and payload/memory cost. Use shuffled-UPL controls
   and cold-start/missing-data tests. Offline surrogate gains are not causal CTR lift.

The 100–1,000 creative ratings discussed for a campaign critic are not, by
themselves, sufficient evidence to train a general behavioral user encoder from
scratch. Start with pretrained semantics and honest baselines; validate a learned
behavioral extension only with adequate authorized data. No paid jobs are needed
to define the schema or conduct initial local experiments, and none are authorized.
