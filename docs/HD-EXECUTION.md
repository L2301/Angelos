# Local high-resolution evidence

## Executed, 2026-10-08

A local Apple Silicon MPS run produced a **1920×1088** SANA-Sprint image,
cropped by four pixels at the top and bottom to **1920×1080**. No image
enlargement, remote inference or paid job was used. The recorded sample is
`web/evidence/hd-baseline.png`; its machine-readable provenance is
`web/evidence/hd-manifest.json`.

- Foundation: `Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers`.
- Revision: `aa76e7f4f4928f378716b6716a2130fba3caf5b1`.
- Diffusers 0.40.0; local cached weights; Apple MPS.
- Transformer/text encoder BF16, VAE FP32, tiled decode.
- Two sampling steps, seed 2026, guidance 4.5, resolution binning disabled.
- Original probe retained the pipeline's default complex human instruction.
- Elapsed time including loading: 135.11 seconds. This is one observed run,
  not a throughput benchmark or placement-time latency claim.
- Delivered PNG SHA-256:
  `e582b57da60d37c296842c98dff5976fa6b51233c27463e6931722ddd5a96e39`.

The image depicts a fictional Aurora outdoor concept: a moss-green backpack
beside an alpine lake. Visual inspection confirms recognizable photographic
content and substantially more detail than the earlier 32×32 student. It does
not certify exact product identity, typography, logo fidelity or brand approval.

## What this does not prove

The packaged CLI was also executed successfully using
`node scripts/hd.js generate examples/aurora-hd.json --device mps --output-dir artifacts/hd/aurora-cli-001`.
Its quiet Aurora bottle variant completed in **98.31 seconds** including loading,
with native `[1,1088,1920,3]` finite float pixels and no resize. The recorded output
hash is `8303c018d97f2240e49d88752ce0f9cb76f5b599831d30a22e1cd068de4432db`.
The local run manifest records both sampling latents on MPS and the precise recipe.
This validates the offline CLI on actual weights, separately from injected unit tests.

This is a **pretrained foundation-model baseline**, not an Angelos campaign-trained
student, not a model distilled from a ComfyUI workflow, and not browser WebGPU
execution. The prompt was supplied directly; learned UPL conditioning and
click-rate improvement are unproven. The checkpoint was trained at 1024;
successful native HD inference is not evidence of 2K training. “0.6B” describes
the denoiser, not the complete pipeline including text encoder and VAE.

The separate CLI in [HD.md](HD.md) disables complex human instruction and records
its own prompts, file hashes, sizes and runtime. It therefore should not be
expected to reproduce this earlier probe byte-for-byte. Reward adaptation has
separate tests and gates in [HD-ALIGNMENT.md](HD-ALIGNMENT.md); none of its
unit tests establish high-resolution trained-model quality.

## Reproduction and research

The public showcase now presents this existing 1920 × 1080 photograph with
separately authored HTML/CSS typography, an Aurora mark and responsive layout.
The mobile layout crops the displayed photo; the downloadable source stays
1920 × 1080. This presentation change ran no new inference or training and does
not change the manifest’s quality-gate result. The earlier 32px WebGPU image
comparison is no longer displayed on the homepage. Its execution report and
provenance remain available; removing the display does not change its failed
conditioning audit.

Use the pinned offline loader and explicit generation commands in [HD.md](HD.md).
No cloud provisioning is configured or authorized. See the
[diffusion review](RESEARCH-DIFFUSION.md) and
[distillation review](RESEARCH-DISTILLATION.md) for the model choice and alternatives.

Primary sources: [SANA-Sprint paper](https://arxiv.org/abs/2503.09641),
[official model](https://huggingface.co/Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers),
[Diffusers SANA-Sprint documentation](https://huggingface.co/docs/diffusers/api/pipelines/sana_sprint).
