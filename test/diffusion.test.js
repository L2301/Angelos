import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { validateDiffusion, renderDiffusionCpu, createDiffusionGpuRenderer } from '../src/browser/diffusion.js';
import { validateStudent, renderCpu, createGpuRenderer } from '../src/browser/student.js';

const UPL = [0.1, 0.7, 0.3, 0.9];
const round = Math.fround;
const NOISE_PREFIX = [2.6681275367736816, 0.7641785740852356, -1.326427936553955, -2.180623769760132, -0.9322425127029419, 0.09389302879571915, 0.09695857018232346, -0.9650884866714478, 1.8710790872573853, -0.32721611857414246, 1.2408912181854248, 2.1603963375091553, -0.08675333112478256, 0.7804286479949951, 0.45386824011802673, -0.9094033241271973];
const PYTORCH_BRAND_MASKS = [
  [10, 10, '766c00f1d107d62324e476efb35778018a33f920e91beafce13c2099d70fbe35'],
  [20, 20, 'b3f0c852401131a2a48a78ebdf2f8a04f745a228038cb37ec5cded910b3eee9b'],
  [32, 32, '81240134d7047b0eb5b7877bebcd0290ba75c23747f8a7714ae546641c8de64e'],
  [67, 49, '84a43f33f9dc19eb3144a4a3078c366f0cd150b39ac9ee7397ab6a831c3ab2e2'],
  [192, 192, '7ebb5372bb5507e4df0dd3af5efce46ac7abf920600abb844dbc08cc9f3cf9df'],
];

function fixture() {
  return {
    format: 'angelos.diffusion.v1', id: 'untrained-unit-fixture', campaignId: 'fixture', graphHash: 'a'.repeat(64),
    encoder: { id: 'structured-demo-v1', dimension: 4 },
    architecture: { name: 'conditional-unet-v1', resolution: 32, channels: [10, 16, 32, 32, 16, 3] },
    layers: [[10, 16, 1], [16, 32, 2], [32, 32, 1], [48, 16, 1], [16, 3, 1]].map(([input, output, stride], index) => ({ input, output, stride, kernel: 3, activation: index === 4 ? 'linear' : 'relu', weights: Array(input * output * 9).fill(0), bias: Array(output).fill(0) })),
    schedule: { predictionType: 'epsilon', alphaCumprod: Array.from({ length: 64 }, (_, index) => round(0.99 - index * 0.01)), timesteps: Array.from({ length: 16 }, (_, index) => Math.round(63 * (15 - index) / 15)) },
    brand: null,
    metrics: { baselineMse: 1, heldoutMse: 1, brandMaxError: null, brandTolerance: 0.06 },
    training: { seed: 2026, steps: 0, device: 'cpu' },
  };
}

function oracleNoise(seed) {
  let state = BigInt(seed);
  function uniform() {
    state = (1664525n * state + 1013904223n) % 4294967296n;
    return (Number(state) + 0.5) / 4294967296;
  }
  return Float32Array.from(Array.from({ length: 1536 }, () => {
    const radius = Math.sqrt(-2 * Math.log(uniform()));
    const angle = uniform() * 2 * Math.PI;
    return [radius * Math.cos(angle), radius * Math.sin(angle)];
  }).flat());
}

function oracleSample(artifact, seed, epsilonAt = () => 0) {
  const sample = oracleNoise(seed);
  for (let step = 0; step < 16; step += 1) {
    const timestep = artifact.schedule.timesteps[step];
    const alpha = artifact.schedule.alphaCumprod[timestep];
    const next = step === 15 ? 1 : artifact.schedule.alphaCumprod[artifact.schedule.timesteps[step + 1]];
    for (let index = 0; index < sample.length; index += 1) {
      const epsilon = epsilonAt(Math.floor(index / 1024), Math.floor(index % 1024 / 32), index % 32, timestep);
      const predicted = round(round(sample[index] - round(round(Math.sqrt(round(1 - alpha))) * epsilon)) / round(Math.sqrt(alpha)));
      const clean = Math.min(1, Math.max(-1, predicted));
      sample[index] = round(round(round(Math.sqrt(next)) * clean) + round(round(Math.sqrt(round(1 - next))) * epsilon));
    }
  }
  return Float32Array.from(Array.from({ length: 1024 }, (_, pixel) => [0, 1, 2].map(channel => round(round(sample[channel * 1024 + pixel] + 1) * 0.5)).concat(1)).flat());
}

test('diffusion fixed schema is immutable and the student validator dispatches to it', () => {
  const artifact = fixture();
  const before = structuredClone(artifact);
  assert.equal(validateDiffusion(artifact), artifact);
  assert.equal(validateStudent(artifact), artifact);
  assert.deepEqual(artifact, before);
  for (const mutation of [
    value => { value.format = 'angelos.student.v1'; },
    value => { value.id = ''; },
    value => { value.id = 'a'.repeat(129); },
    value => { value.campaignId = 'bad id'; },
    value => { value.campaignId = 'a'.repeat(65); },
    value => { value.graphHash = 'bad'; },
    value => { value.encoder.dimension = 3; },
    value => { value.encoder.shader = 'arbitrary'; },
    value => { value.architecture.name = 'custom'; },
    value => { value.architecture.resolution = 64; },
    value => { value.architecture.channels[0] = 9; },
    value => { value.layers.pop(); },
    value => { value.layers[0].kernel = 1; },
    value => { value.layers[1].stride = 1; },
    value => { value.layers[2].output = 64; },
    value => { value.layers[3].input = 32; },
    value => { value.layers[4].activation = 'sigmoid'; },
    value => { value.layers[0].weights.pop(); },
    value => { value.layers[1].weights[0] = Infinity; },
    value => { delete value.layers[2].weights[4]; },
    value => { value.layers[3].bias[0] = NaN; },
    value => { value.layers[4].weights[0] = 1001; },
    value => { value.layers[0].shader = 'arbitrary'; },
    value => { value.schedule.predictionType = 'sample'; },
    value => { value.schedule.alphaCumprod.pop(); },
    value => { value.schedule.alphaCumprod[0] = 1; },
    value => { value.schedule.alphaCumprod[63] = 0; },
    value => { value.schedule.alphaCumprod[63] = 1e-100; },
    value => { value.schedule.alphaCumprod[1] = value.schedule.alphaCumprod[0] - 1e-12; },
    value => { value.schedule.alphaCumprod[2] = 0.999; },
    value => { value.schedule.timesteps.pop(); },
    value => { value.schedule.timesteps[0] = 62; },
    value => { value.schedule.timesteps[15] = 1; },
    value => { value.schedule.timesteps[1] = 63; },
    value => { value.schedule.timesteps[1] = 59.5; },
    value => { value.metrics.heldoutMse = -1; },
    value => { value.metrics.callback = () => 0; },
    value => { value.metrics.recursive = value.metrics; },
    value => { value.metrics.extra = Array(5000).fill(0); },
    value => { value.training.steps = -1; },
    value => { value.training.seed = NaN; },
    value => { value.training.seed = 4294967296; },
    value => { value.training.device = ''; },
    value => { delete value.brand; },
    value => { value.code = 'arbitrary'; },
  ]) {
    const invalid = fixture();
    mutation(invalid);
    assert.throws(() => validateDiffusion(invalid), TypeError);
  }
  for (const invalid of [null, [], 'model', undefined]) assert.throws(() => validateDiffusion(invalid), TypeError);
  artifact.id = 'a'.repeat(128);
  artifact.campaignId = 'a'.repeat(64);
  assert.equal(validateDiffusion(artifact), artifact);
});

test('brand geometry, mode, bounded colors, and metrics reject malformed values', () => {
  for (const mutation of [
    brand => { brand.mode = 'anything'; }, brand => { brand.x = -1; }, brand => { brand.y = 0.99; },
    brand => { brand.cell = 0.001; }, brand => { brand.cell = 0.051; }, brand => { brand.color[0] = NaN; },
    brand => { brand.background = [0, 1]; }, brand => { brand.color[1] = 1.1; }, brand => { brand.extra = 1; },
  ]) {
    const artifact = fixture();
    artifact.brand = { mode: 'composite', x: 0.125, y: 0.125, cell: 0.03125, color: [0.2, 0.4, 0.9], background: [0.9, 0.1, 0.3] };
    artifact.metrics.brandMaxError = 0.5;
    mutation(artifact.brand);
    assert.throws(() => validateDiffusion(artifact), /brand/);
  }
});

test('render input and creative seed validation happens before any inference', () => {
  const artifact = fixture();
  for (const seed of [-1, 4294967296, 0.5, NaN, Infinity, '2026', null, true]) assert.throws(() => renderDiffusionCpu(artifact, UPL, 1, 1, { seed }), /seed/);
  for (const upl of [null, [0, 0, 0], [0, 0, 0, 2], [NaN, 0, 0, 0], ['0', 0, 0, 0]]) assert.throws(() => renderDiffusionCpu(artifact, upl, 1, 1), /upl/);
  for (const size of [0, 513, 1.5, Infinity, '32']) {
    assert.throws(() => renderDiffusionCpu(artifact, UPL, size, 1), /width/);
    assert.throws(() => renderDiffusionCpu(artifact, UPL, 1, size), /height/);
  }
  for (const options of [null, { composite: 1 }, { arbitrary: 1 }]) assert.throws(() => renderDiffusionCpu(artifact, UPL, 1, 1, options));
});

test('LCG Box-Muller seeds, all 16 DDIM steps, and native nearest display match an independent reference', () => {
  const artifact = fixture();
  assert.deepEqual([...oracleNoise(2026).subarray(0, 16)], NOISE_PREFIX);
  for (const seed of [0, 2026, 0xffffffff]) assert.deepEqual(renderCpu(artifact, UPL, 32, 32, { seed }), oracleSample(artifact, seed));
  const native = renderDiffusionCpu(artifact, UPL, 32, 32);
  assert.deepEqual(native, renderDiffusionCpu(artifact, UPL, 32, 32, { seed: 2026 }));
  assert.notDeepEqual(native, renderDiffusionCpu(artifact, UPL, 32, 32, { seed: 2027 }));
  const scaled = renderDiffusionCpu(artifact, UPL, 67, 49);
  for (let row = 0; row < 49; row += 1) {
    for (let column = 0; column < 67; column += 1) {
      const source = (Math.floor(row * 32 / 49) * 32 + Math.floor(column * 32 / 67)) * 4;
      assert.deepEqual(scaled.subarray((row * 67 + column) * 4, (row * 67 + column + 1) * 4), native.subarray(source, source + 4));
    }
  }
});

test('OIHW convolutions use all five layers, stride-two padding, upsample-before-skip order, UPL, xy, time, and ReLU', () => {
  const artifact = fixture();
  function tap(layerIndex, output, input, row, column, value) {
    const layer = artifact.layers[layerIndex];
    layer.weights[((output * layer.input + input) * 3 + row) * 3 + column] = value;
  }
  tap(0, 0, 8, 1, 1, 1);
  tap(0, 1, 4, 1, 1, 1);
  artifact.layers[0].bias[1] = -0.25;
  tap(0, 2, 9, 0, 2, 1);
  tap(0, 3, 7, 1, 1, 1);
  tap(1, 0, 0, 1, 1, 1);
  tap(2, 0, 0, 2, 1, 1);
  tap(3, 0, 0, 1, 1, 1);
  tap(3, 0, 33, 1, 1, 2);
  tap(3, 1, 34, 1, 1, 1);
  tap(3, 2, 35, 1, 1, 1);
  tap(4, 0, 0, 1, 1, 0.1);
  tap(4, 1, 1, 1, 1, 0.2);
  tap(4, 2, 2, 1, 1, -0.3);
  artifact.layers[4].bias[0] = -0.1;
  for (const upl of [UPL, [1, 0, 1, 0]]) {
    const expected = oracleSample(artifact, 2026, (channel, row, column, timestep) => {
      if (channel === 0) {
        const coarse = row < 30 ? (2 * Math.floor(column / 2) + 0.5) / 32 : 0;
        const skip = Math.max(0, round(round(upl[1]) - 0.25));
        return round(round(-0.1) + round(round(0.1) * round(coarse + round(2 * skip))));
      }
      if (channel === 1) return round(round(0.2) * (row > 0 && column < 31 ? (row - 0.5) / 32 : 0));
      return round(round(-0.3) * round(timestep / 63));
    });
    assert.deepEqual(renderDiffusionCpu(artifact, upl, 32, 32), expected);
  }
});

test('CPU brand masks exactly match PyTorch float32 decimal-boundary fixtures at five display sizes', () => {
  const artifact = fixture();
  artifact.brand = { mode: 'composite', x: 0.1, y: 0.1, cell: 0.05, color: [0.2, 0.4, 0.9], background: [0.9, 0.1, 0.3] };
  artifact.metrics.brandMaxError = 0.5;
  const foreground = new Float32Array([...artifact.brand.color, 1]);
  const background = new Float32Array([...artifact.brand.background, 1]);
  for (const [width, height, expectedHash] of PYTORCH_BRAND_MASKS) {
    const raw = renderDiffusionCpu(artifact, UPL, width, height, { seed: 7, composite: false });
    const delivered = renderDiffusionCpu(artifact, UPL, width, height, { seed: 7 });
    let mask = '';
    for (let offset = 0; offset < delivered.length; offset += 4) {
      const pixel = delivered.subarray(offset, offset + 4);
      if (pixel.every((value, channel) => value === foreground[channel])) mask += 'i';
      else if (pixel.every((value, channel) => value === background[channel])) mask += 'b';
      else {
        mask += '.';
        assert.deepEqual(pixel, raw.subarray(offset, offset + 4));
      }
    }
    assert.equal(createHash('sha256').update(mask).digest('hex'), expectedHash, `${width}x${height}`);
    if (width === 10) assert.equal(mask, ['..........', '.bib......', '.ibi......', '.ibi......', '.ibi......', '..........', '..........', '..........', '..........', '..........'].join(''));
  }
});

test('exact display-resolution brand plate preserves seeded raw pixels outside half-open bounds', () => {
  const artifact = fixture();
  artifact.brand = { mode: 'composite', x: 0.5 / 32, y: 0.5 / 32, cell: 1 / 32, color: [0.2, 0.4, 0.9], background: [0.9, 0.1, 0.3] };
  artifact.metrics.brandMaxError = 0.5;
  const raw = renderDiffusionCpu(artifact, UPL, 32, 32, { seed: 7, composite: false });
  const delivered = renderDiffusionCpu(artifact, UPL, 32, 32, { seed: 7 });
  const glyph = ['01110', '10001', '10001', '11111', '10001', '10001', '10001'];
  for (let row = 0; row < 32; row += 1) {
    for (let column = 0; column < 32; column += 1) {
      const offset = (row * 32 + column) * 4;
      const expected = column < 7 && row < 9 ? new Float32Array([...(glyph[row - 1]?.[column - 1] === '1' ? artifact.brand.color : artifact.brand.background), 1]) : raw.subarray(offset, offset + 4);
      assert.deepEqual(delivered.subarray(offset, offset + 4), expected);
    }
  }
  artifact.brand.mode = 'neural';
  assert.deepEqual(renderDiffusionCpu(artifact, UPL, 32, 32, { seed: 7 }), raw);
});

function gpuDouble(context, options = {}) {
  const state = { buffers: [], scopes: [], shaders: [], dispatches: [], destroyed: false, writes: [], submissions: 0 };
  const listeners = new Map();
  let lose;
  const device = {
    limits: { maxBufferSize: 16 * 1024 * 1024, maxStorageBufferBindingSize: 16 * 1024 * 1024, maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256, maxStorageBuffersPerShaderStage: 8, maxBindGroups: 4, maxBindingsPerBindGroup: 1000, maxComputeWorkgroupsPerDimension: 65535, ...options.limits },
    lost: new Promise(resolve => { lose = resolve; }),
    addEventListener(name, callback) { listeners.set(name, callback); },
    removeEventListener(name) { listeners.delete(name); },
    destroy() { state.destroyed = true; lose({ reason: 'destroyed' }); },
    pushErrorScope(scope) { state.scopes.push(scope); },
    async popErrorScope() { state.scopes.pop(); return options.scopeError ? { message: 'test scoped error' } : null; },
    createBindGroupLayout(descriptor) { return descriptor; },
    createPipelineLayout(descriptor) { return descriptor; },
    createShaderModule(descriptor) {
      state.shaders.push(descriptor);
      return { async getCompilationInfo() { return { messages: options.shaderError ? [{ type: 'error', message: 'test WGSL failure', lineNum: 1, linePos: 1 }] : [] }; } };
    },
    async createComputePipelineAsync(descriptor) {
      if (options.pipelineError) throw new Error('test pipeline failure');
      return descriptor;
    },
    createBuffer(descriptor) {
      const data = new ArrayBuffer(descriptor.size);
      let rejectMap;
      const buffer = {
        ...descriptor, data, destroyed: false, mapped: false,
        destroy() { this.destroyed = true; this.mapped = false; rejectMap?.(new Error('mapping destroyed')); },
        async mapAsync() {
          if (options.mapError) throw new Error('test mapping failure');
          await new Promise((resolve, reject) => { rejectMap = reject; setTimeout(resolve, options.mapDelay ?? 0); });
          this.mapped = true;
        },
        getMappedRange(offset = 0, size = descriptor.size) { assert.equal(this.mapped, true); return data.slice(offset, offset + size); },
        unmap() { this.mapped = false; },
      };
      state.buffers.push(buffer);
      return buffer;
    },
    createBindGroup(descriptor) { return descriptor; },
    createCommandEncoder() {
      let pipeline;
      return {
        beginComputePass() { return { setPipeline(value) { pipeline = value; }, setBindGroup() {}, dispatchWorkgroups(count) { state.dispatches.push([pipeline.label, count]); }, end() {} }; },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) { new Float32Array(destination.data, destinationOffset, size / 4).fill(options.nonfinite ? NaN : 0.25); },
        finish() { return {}; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, data) {
        new Uint8Array(buffer.data, offset, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
        state.writes.push({ label: buffer.label, bytes: data.byteLength });
      },
      submit() { state.submissions += 1; },
    },
  };
  state.device = device;
  state.lose = () => lose({ message: 'test loss' });
  state.error = () => listeners.get('uncapturederror')?.({ preventDefault() {}, error: { message: 'test uncaptured' } });
  const globals = {
    navigator: { gpu: { async requestAdapter() { return options.noAdapter ? null : { info: { isFallbackAdapter: options.fallback ?? false }, isFallbackAdapter: options.legacyFallback ?? false, async requestDevice() { return device; } }; } } },
    GPUBufferUsage: { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 }, GPUMapMode: { READ: 1 }, GPUShaderStage: { COMPUTE: 4 },
  };
  for (const [name, value] of Object.entries(globals)) {
    const before = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, value });
    context.after(() => { if (before) Object.defineProperty(globalThis, name, before); else delete globalThis[name]; });
  }
  return state;
}

test('GPU refuses absent/fallback adapters without a CPU fallback', async context => {
  for (const options of [{ noAdapter: true }, { fallback: true }, { legacyFallback: true }]) {
    await context.test(JSON.stringify(options), async child => {
      const state = gpuDouble(child, options);
      await assert.rejects(createGpuRenderer(fixture()), /adapter/);
      assert.equal(state.buffers.length, 0);
    });
  }
  await context.test('no API', async child => {
    gpuDouble(child);
    delete globalThis.navigator.gpu;
    await assert.rejects(createDiffusionGpuRenderer(fixture()), /no CPU fallback/);
  });
});

test('GPU resource, shader, pipeline, and error-scope failures drain scopes and release the device', async context => {
  for (const options of [
    { limits: { maxComputeInvocationsPerWorkgroup: 32 } }, { limits: { maxComputeWorkgroupSizeX: 32 } },
    { limits: { maxComputeWorkgroupsPerDimension: 255 } }, { limits: { maxStorageBuffersPerShaderStage: 4 } },
    { limits: { maxBindGroups: 0 } }, { limits: { maxBindingsPerBindGroup: 4 } },
    { limits: { maxBufferSize: 4 } }, { limits: { maxStorageBufferBindingSize: 4 } },
    { shaderError: true }, { pipelineError: true }, { scopeError: true },
  ]) {
    await context.test(JSON.stringify(options), async child => {
      const state = gpuDouble(child, options);
      await assert.rejects(createDiffusionGpuRenderer(fixture()), /WebGPU|shader|pipeline/);
      assert.equal(state.destroyed, true);
      assert.ok(state.buffers.every(buffer => buffer.destroyed));
      assert.equal(state.scopes.length, 0);
    });
  }
});

test('GPU lifecycle double checks 97 real dispatch commands, immutable uploads, reuse, seeds, and single-flight, not shader numerics', async context => {
  const state = gpuDouble(context, { mapDelay: 15 });
  const artifact = fixture();
  const renderer = await createGpuRenderer(artifact);
  context.after(() => renderer.dispose());
  assert.equal(state.shaders.length, 7);
  artifact.layers[0].bias[0] = 100;
  const pending = renderer.render(UPL, 32, 32);
  await assert.rejects(renderer.render(UPL, 1, 1), /in flight/);
  const result = await pending;
  assert.equal(result.seed, 2026);
  assert.equal(result.steps, 16);
  assert.equal(result.nativeResolution, 32);
  assert.equal(result.pixels.length, 4096);
  assert.ok(result.milliseconds >= 10);
  assert.equal(state.dispatches.length, 97);
  assert.equal(state.dispatches.filter(([label]) => label === 'Diffusion DDIM').length, 16);
  for (let layer = 0; layer < 5; layer += 1) assert.equal(state.dispatches.filter(([label]) => label === `Diffusion conv${layer}`).length, 16);
  const initial = state.buffers.find(buffer => buffer.label === 'Diffusion sample 0');
  assert.deepEqual([...new Float32Array(initial.data).subarray(0, 16)], NOISE_PREFIX);
  const weights = state.buffers.find(buffer => buffer.label === 'Diffusion weights and schedule');
  assert.equal(new Float32Array(weights.data)[artifact.layers[0].weights.length], 0);
  const bufferCount = state.buffers.length;
  await renderer.render(UPL, 16, 12, { seed: 0, composite: false });
  assert.equal(state.buffers.length, bufferCount);
  assert.deepEqual(new Float32Array(initial.data), oracleNoise(0));
  await renderer.render(UPL, 64, 64, { seed: 0xffffffff });
  assert.equal(state.buffers.length, bufferCount + 2);
  assert.equal(state.buffers.filter(buffer => buffer.destroyed).length, 2);
  renderer.dispose();
  renderer.dispose();
  assert.ok(state.buffers.every(buffer => buffer.destroyed && !buffer.mapped));
  assert.equal(state.scopes.length, 0);
  await assert.rejects(renderer.render(UPL, 1, 1), /disposed/);
});

test('bad render arguments and limits allocate no extra GPU resources', async context => {
  const state = gpuDouble(context);
  const renderer = await createDiffusionGpuRenderer(fixture());
  context.after(() => renderer.dispose());
  const allocated = state.buffers.length;
  await assert.rejects(renderer.render(UPL, 1, 1, { seed: -1 }), /seed/);
  await assert.rejects(renderer.render(UPL, 513, 1), /width/);
  state.device.limits.maxComputeWorkgroupsPerDimension = 256;
  await assert.rejects(renderer.render(UPL, 512, 512), /maxComputeWorkgroups/);
  state.device.limits.maxBufferSize = 4;
  await assert.rejects(renderer.render(UPL, 1, 1), /maxBufferSize/);
  assert.equal(state.buffers.length, allocated);
});

test('GPU uploads the same PyTorch float32 brand classifications for every DDIM step', async context => {
  const state = gpuDouble(context);
  const artifact = fixture();
  artifact.brand = { mode: 'composite', x: 0.1, y: 0.1, cell: 0.05, color: [0.2, 0.4, 0.9], background: [0.9, 0.1, 0.3] };
  artifact.metrics.brandMaxError = 0.5;
  const renderer = await createDiffusionGpuRenderer(artifact);
  context.after(() => renderer.dispose());
  for (const [width, height, expectedHash] of PYTORCH_BRAND_MASKS) {
    await renderer.render(UPL, width, height);
    for (const buffer of state.buffers.filter(entry => entry.label.startsWith('Diffusion settings '))) {
      const settings = new Uint32Array(buffer.data);
      let mask = '';
      for (let row = 0; row < height; row += 1) {
        for (let column = 0; column < width; column += 1) {
          const plateColumn = settings[8 + column];
          const plateRow = settings[8 + width + row];
          const inside = plateColumn < 7 && plateRow < 9;
          const ink = inside && ['01110', '10001', '10001', '11111', '10001', '10001', '10001'][plateRow - 1]?.[plateColumn - 1] === '1';
          mask += ink ? 'i' : inside ? 'b' : '.';
        }
      }
      assert.equal(createHash('sha256').update(mask).digest('hex'), expectedHash, `${width}x${height} ${buffer.label}`);
    }
  }
});

test('failed map and nonfinite output release display resources and permit retry', async context => {
  for (const failure of ['mapError', 'nonfinite', 'scopeError']) {
    await context.test(failure, async child => {
      const options = {};
      const state = gpuDouble(child, options);
      const renderer = await createDiffusionGpuRenderer(fixture());
      child.after(() => renderer.dispose());
      options[failure] = true;
      await assert.rejects(renderer.render(UPL, 2, 2), /mapping|nonfinite|scoped/);
      assert.ok(state.buffers.slice(-2).every(buffer => buffer.destroyed));
      assert.equal(state.scopes.length, 0);
      options[failure] = false;
      await renderer.render(UPL, 2, 2);
    });
  }
});

test('device loss, uncaptured errors, and disposal during mapping destroy every resource', async context => {
  for (const failure of ['lose', 'error', 'dispose']) {
    await context.test(failure, async child => {
      const state = gpuDouble(child, { mapDelay: 20 });
      const renderer = await createDiffusionGpuRenderer(fixture());
      const pending = renderer.render(UPL, 2, 2);
      if (failure === 'dispose') renderer.dispose();
      else state[failure]();
      await assert.rejects(pending, /destroyed|disposed|loss|uncaptured/);
      await assert.rejects(renderer.render(UPL, 1, 1), /disposed|loss|uncaptured/);
      assert.ok(state.buffers.every(buffer => buffer.destroyed));
      assert.equal(state.scopes.length, 0);
      renderer.dispose();
    });
  }
});
