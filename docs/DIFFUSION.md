# Diffusion student contract

The campaign student is an image-space conditional DDPM denoiser, sampled with deterministic DDIM (eta=0) from explicitly seeded Gaussian noise. It is not the old coordinate MLP, a pretrained foundation model, or a claim of production image quality. Open-weight graph teachers supply complete rendered campaign examples; compilation and learning remain separate.

## Version 1 interchange

`format: angelos.diffusion.v1` retains `id`, `campaignId`, `graphHash`, `encoder`, `brand`, `metrics`, and `training`. `architecture` is exactly `{name: "conditional-unet-v1", resolution: 32, channels: [10,16,32,32,16,3]}`. `layers` contains five objects with `input`, `output`, `kernel: 3`, `stride`, `activation`, flattened OIHW `weights`, and `bias`.

The network consumes NCHW noisy RGB, with four spatially broadcast UPL dimensions, timestep `t/(T-1)`, and normalized pixel-center x,y appended in that order. Conv0 is 10→16, stride 1, ReLU. Conv1 is 16→32, stride 2, ReLU. Conv2 is 32→32, stride 1, ReLU. Nearest-neighbor 2× upsample Conv2, then concatenate Conv0 after it. Conv3 is 48→16, stride 1, ReLU. Conv4 is 16→3, stride 1, linear epsilon prediction. Every convolution has one zero-padding pixel on each side. Arithmetic is float32.

`schedule` contains `predictionType: "epsilon"`, `alphaCumprod` (64 strictly decreasing values in (0,1)), and `timesteps` (16 strictly descending integer indices, starting 63 and ending 0). Training uses a cosine noise schedule and random timesteps, with epsilon MSE. Inference predicts x0 = clamp((xt - sqrt(1-alpha)*epsilon)/sqrt(alpha), -1,1); next sample = sqrt(nextAlpha)*x0 + sqrt(1-nextAlpha)*epsilon. After timestep 0, nextAlpha is 1. Output RGB is (x0+1)/2. No extra noise is added between steps.

Noise generation uses uint32 LCG state = (1664525*state+1013904223) modulo 2^32, uniform=(state+0.5)/2^32, Box–Muller pairs (cos then sin), cast individually to float32. The resulting array is channel-first RGB. Seeds are uint32, default 2026. Seed identifies a creative alongside artifact and exposure-time UPL. GPU output may vary slightly by device; seed replay is not a cross-device bitwise guarantee.

Browser render API stays `render(upl,width,height,{composite,seed})`; the native 32×32 raster is nearest-neighbor scaled for display, not represented as genuine high-resolution generation. Entire iterative denoising and exact brand fallback execute in WebGPU. CPU implementation exists only as an explicit numerical test reference. Raw and composited previews must use the same seed.

Diffusion brand-cell classification follows the tensor teacher's float32 arithmetic:
round the normalized pixel center and brand origin/cell to float32, round subtraction,
round division, then floor. This defines boundary ownership consistently with PyTorch,
including non-power-of-two display dimensions; mathematical double-precision boundaries
can differ by a pixel and must not be substituted. The legacy MLP renderer remains a
separate historical format.

Weights from this small trainable U-Net are a capacity-limited research student. Scaling image quality requires more data, capacity, pretrained initialization or latent-space deployment and evaluation; reducing denoising steps is a separate distillation problem.
