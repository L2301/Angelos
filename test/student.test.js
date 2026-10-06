import test from 'node:test';
import assert from 'node:assert/strict';
import { validateStudent, renderCpu, createGpuRenderer } from '../src/browser/student.js';

function layer(input, output, activation) {
  return { input, output, activation, weights: Array(input * output).fill(0), bias: Array(output).fill(0) };
}

function fixture(firstWidth = 2, secondWidth = firstWidth) {
  return {
    format: 'angelos.student.v1',
    id: 'untrained-test-fixture',
    campaignId: 'fixture',
    graphHash: '0123456789abcdef'.repeat(4),
    encoder: { id: 'structured-demo-v1', dimension: 4 },
    inputOrder: ['x', 'y', 'upl.0', 'upl.1', 'upl.2', 'upl.3'],
    layers: [layer(6, firstWidth, 'relu'), layer(firstWidth, secondWidth, 'relu'), layer(secondWidth, 3, 'sigmoid')],
    brand: null,
    metrics: { baselineMse: 0.1, heldoutMse: 0.1, brandMaxError: 0, brandTolerance: 0.06 },
    training: { seed: 0, steps: 0, device: 'cpu' },
  };
}

function brandFixture(mode = 'composite') {
  const artifact = fixture(1);
  artifact.brand = { mode, x: 0.125, y: 0.125, cell: 0.03125, color: [0.2, 0.4, 0.9], background: [0.9, 0.1, 0.3] };
  return artifact;
}

function closeEnough(actual, expected, tolerance = 1e-6) {
  assert.equal(actual.length, expected.length);
  for (let index = 0; index < actual.length; index += 1) {
    assert.ok(Math.abs(actual[index] - expected[index]) <= tolerance, `channel ${index}: ${actual[index]} versus ${expected[index]}`);
  }
}

test('validates the frozen schema without mutation, including unequal hidden widths', () => {
  for (const artifact of [fixture(1), fixture(64), fixture(3, 7), brandFixture('neural'), brandFixture()]) {
    const original = structuredClone(artifact);
    assert.equal(validateStudent(artifact), artifact);
    assert.deepEqual(artifact, original);
  }
});

test('brandMaxError permits null only for artifacts without a brand and preserves it', () => {
  const artifact = fixture();
  artifact.metrics.brandMaxError = null;
  assert.equal(validateStudent(artifact), artifact);
  assert.equal(artifact.metrics.brandMaxError, null);
  assert.deepEqual([...renderCpu(artifact, [0, 0, 0, 0], 1, 1)], [0.5, 0.5, 0.5, 1]);
  assert.equal(artifact.metrics.brandMaxError, null);
  for (const candidate of [fixture(), brandFixture('neural'), brandFixture('composite')]) {
    for (const invalid of [-0.01, NaN, Infinity, undefined, '0']) {
      candidate.metrics.brandMaxError = invalid;
      assert.throws(() => validateStudent(candidate), TypeError);
    }
    for (const valid of [0, 0.06, 1]) {
      candidate.metrics.brandMaxError = valid;
      assert.equal(validateStudent(candidate), candidate);
    }
    if (candidate.brand !== null) {
      candidate.metrics.brandMaxError = null;
      assert.throws(() => validateStudent(candidate), /brandMaxError/);
    }
  }
});

test('rejects malformed model identity, executable payloads, layouts, and weights', () => {
  const mutations = [
    artifact => { artifact.format = 'other'; },
    artifact => { artifact.id = ''; },
    artifact => { artifact.campaignId = 'not an ID'; },
    artifact => { artifact.graphHash = 'stale'; },
    artifact => { artifact.encoder.id = 'other'; },
    artifact => { artifact.encoder.dimension = 3; },
    artifact => { artifact.encoder.shader = 'code'; },
    artifact => { artifact.inputOrder.reverse(); },
    artifact => { artifact.layers.pop(); },
    artifact => { artifact.layers[0].input = 5; },
    artifact => { artifact.layers[0].output = 0; },
    artifact => { artifact.layers[0].output = 65; },
    artifact => { artifact.layers[0].output = 1.5; },
    artifact => { artifact.layers[1].input = 1; },
    artifact => { artifact.layers[1].activation = 'sigmoid'; },
    artifact => { artifact.layers[2].output = 2; },
    artifact => { artifact.layers[2].activation = 'relu'; },
    artifact => { artifact.layers[0].weights.pop(); },
    artifact => { artifact.layers[0].bias.push(0); },
    artifact => { delete artifact.layers[0].weights[0]; },
    artifact => { artifact.layers[0].weights[0] = NaN; },
    artifact => { artifact.layers[1].weights[0] = Infinity; },
    artifact => { artifact.layers[0].bias[0] = -1000.01; },
    artifact => { artifact.layers[2].weights[0] = 1000.01; },
    artifact => { artifact.layers[0].weights[0] = '0'; },
    artifact => { artifact.layers[0].shader = 'code'; },
    artifact => { artifact.shader = 'code'; },
    artifact => { artifact.metrics.heldoutMse = NaN; },
    artifact => { artifact.metrics.callback = () => 0; },
    artifact => { artifact.training.steps = -1; },
    artifact => { artifact.training.seed = 0.5; },
    artifact => { artifact.training.device = ''; },
    artifact => { delete artifact.brand; },
  ];
  for (const mutate of mutations) {
    const artifact = fixture();
    mutate(artifact);
    assert.throws(() => validateStudent(artifact), TypeError);
  }
  for (const artifact of [null, [], undefined, 'model']) assert.throws(() => validateStudent(artifact), TypeError);
  const artifact = fixture();
  artifact.metrics.cycle = artifact.metrics;
  assert.throws(() => validateStudent(artifact), /deeply nested/);
});

test('brand validation bounds both plate geometry and its colors', () => {
  for (const mutate of [
    artifact => { artifact.brand.mode = 'unknown'; },
    artifact => { artifact.brand.x = -0.1; },
    artifact => { artifact.brand.y = 1; },
    artifact => { artifact.brand.cell = 0.004; },
    artifact => { artifact.brand.cell = 0.051; },
    artifact => { artifact.brand.color[0] = 1.1; },
    artifact => { artifact.brand.background = [0, 0]; },
    artifact => { artifact.brand.mask = 'code'; },
  ]) {
    const artifact = brandFixture();
    mutate(artifact);
    assert.throws(() => validateStudent(artifact), TypeError);
  }
});

test('zero weights yield finite opaque half-gray without any trained artifact', () => {
  const pixels = renderCpu(fixture(), [0, 0.25, 0.75, 1], 3, 2);
  assert.ok(pixels instanceof Float32Array);
  assert.deepEqual([...pixels], Array.from({ length: 6 }, () => [0.5, 0.5, 0.5, 1]).flat());
});

test('the integration UI resolution produces independent complete 192 by 192 RGBA frames', () => {
  const artifact = fixture(1);
  const first = renderCpu(artifact, [0, 0.25, 0.75, 1], 192, 192);
  const second = renderCpu(artifact, [1, 0.75, 0.25, 0], 192, 192);
  assert.equal(first.length, 192 * 192 * 4);
  assert.notEqual(first.buffer, second.buffer);
  for (let index = 0; index < first.length; index += 1) assert.equal(first[index], index % 4 === 3 ? 1 : 0.5);
  first[0] = 0;
  assert.equal(second[0], 0.5);
});

test('row-major layers use biases, both ReLUs, and a stable sigmoid', () => {
  const artifact = fixture(2, 3);
  artifact.layers[0].weights = [1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6];
  artifact.layers[0].bias = [-1, 1];
  artifact.layers[1].weights = [2, 0, -2, 0, 1, 3];
  artifact.layers[1].bias = [0.25, -0.5, -0.25];
  artifact.layers[2].weights = [1, 0, 0, 0, 1, 0, -1, 0, 2];
  artifact.layers[2].bias = [0.1, -0.2, 0.3];
  closeEnough(renderCpu(artifact, [0, 0, 0, 0], 1, 1), [1 / (1 + Math.exp(-1.35)), 1 / (1 + Math.exp(0.2)), 1 / (1 + Math.exp(0.45)), 1]);
  artifact.layers[2].weights.fill(0);
  artifact.layers[2].bias = [-1000, 0, 1000];
  assert.deepEqual([...renderCpu(artifact, [1, 1, 1, 1], 1, 1)], [0, 0.5, 1, 1]);
});

test('all six features follow the declared input order and pixel-center orientation', () => {
  const artifact = fixture(6);
  for (let feature = 0; feature < 6; feature += 1) {
    artifact.layers[0].weights[feature * 6 + feature] = 1;
    artifact.layers[1].weights[feature * 6 + feature] = 1;
  }
  const upl = [0.125, 0.25, 0.5, 0.875];
  for (let feature = 0; feature < 6; feature += 1) {
    artifact.layers[2].weights.fill(0);
    artifact.layers[2].weights[feature] = 1;
    const pixels = renderCpu(artifact, upl, 2, 2);
    for (let pixel = 0; pixel < 4; pixel += 1) {
      const input = feature === 0 ? (pixel % 2 + 0.5) / 2 : feature === 1 ? (Math.floor(pixel / 2) + 0.5) / 2 : upl[feature - 2];
      closeEnough(pixels.subarray(pixel * 4, pixel * 4 + 4), [1 / (1 + Math.exp(-input)), 0.5, 0.5, 1]);
    }
  }
});

test('CPU diagnostics accumulate f32 rather than retaining double-precision intermediates', () => {
  const artifact = fixture(1);
  artifact.layers[0].weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
  artifact.layers[0].bias = [0.7];
  artifact.layers[1].weights = [1];
  artifact.layers[2].weights = [1, 0, 0];
  const inputs = [Math.fround(0.5), Math.fround(0.5), Math.fround(0.1), Math.fround(0.2), Math.fround(0.3), Math.fround(0.4)];
  let total = Math.fround(0.7);
  for (let index = 0; index < 6; index += 1) total = Math.fround(total + Math.fround(Math.fround(artifact.layers[0].weights[index]) * inputs[index]));
  const exponential = Math.fround(Math.exp(-total));
  assert.equal(renderCpu(artifact, [0.1, 0.2, 0.3, 0.4], 1, 1)[0], Math.fround(1 / Math.fround(1 + exponential)));
});

test('render inputs reject malformed, nonfinite, or out-of-domain values', () => {
  const artifact = fixture();
  for (const upl of [null, [0, 0, 0], [0, 0, 0, 0, 0], [NaN, 0, 0, 0], [Infinity, 0, 0, 0], [-0.1, 0, 0, 0], [0, 0, 0, 1.01], ['0', 0, 0, 0]]) {
    assert.throws(() => renderCpu(artifact, upl, 1, 1), /upl/);
  }
  for (const dimension of [0, -1, 513, 1.5, NaN, Infinity, '1']) {
    assert.throws(() => renderCpu(artifact, [0, 0, 0, 0], dimension, 1), /width/);
    assert.throws(() => renderCpu(artifact, [0, 0, 0, 0], 1, dimension), /height/);
  }
  for (const options of [null, { composite: 1 }, { extra: true }]) assert.throws(() => renderCpu(artifact, [0, 0, 0, 0], 1, 1, options));
  assert.equal(renderCpu(fixture(1), [0, 0, 0, 0], 512, 1).length, 2048);
});

test('composite mode replaces the full plate, preserves everything outside, and raw stays available', () => {
  const artifact = brandFixture();
  const raw = renderCpu(artifact, [0, 0, 0, 0], 64, 64, { composite: false });
  const delivered = renderCpu(artifact, [0, 0, 0, 0], 64, 64);
  const glyph = ['01110', '10001', '10001', '11111', '10001', '10001', '10001'];
  let inkPixels = 0;
  let backgroundPixels = 0;
  for (let row = 0; row < 64; row += 1) {
    for (let column = 0; column < 64; column += 1) {
      const offset = (row * 64 + column) * 4;
      const plateColumn = Math.floor((column - 8) / 2);
      const plateRow = Math.floor((row - 8) / 2);
      let expected = [0.5, 0.5, 0.5];
      if (plateColumn >= 0 && plateColumn < 7 && plateRow >= 0 && plateRow < 9) {
        const ink = glyph[plateRow - 1]?.[plateColumn - 1] === '1';
        expected = ink ? artifact.brand.color : artifact.brand.background;
        if (ink) inkPixels += 1;
        else backgroundPixels += 1;
      }
      assert.deepEqual(delivered.subarray(offset, offset + 4), new Float32Array([...expected, 1]));
      assert.deepEqual([...raw.subarray(offset, offset + 4)], [0.5, 0.5, 0.5, 1]);
    }
  }
  assert.ok(inkPixels > 0 && backgroundPixels > inkPixels);
  artifact.brand.mode = 'neural';
  assert.deepEqual(renderCpu(artifact, [0, 0, 0, 0], 64, 64), raw);
});

test('brand mask uses half-open bounds at pixel-center boundaries', () => {
  const artifact = brandFixture();
  artifact.brand.x = 0.5 / 32;
  artifact.brand.y = 0.5 / 32;
  const pixels = renderCpu(artifact, [0, 0, 0, 0], 32, 32);
  assert.deepEqual(pixels.subarray(0, 4), new Float32Array([...artifact.brand.background, 1]));
  assert.deepEqual([...pixels.subarray(7 * 4, 8 * 4)], [0.5, 0.5, 0.5, 1]);
  assert.deepEqual([...pixels.subarray(9 * 32 * 4, 9 * 32 * 4 + 4)], [0.5, 0.5, 0.5, 1]);
});

function gpuDouble(context, options = {}) {
  const buffers = [];
  const scopes = [];
  const state = { buffers, scopes, destroyed: false, submissions: 0, shader: '', commands: [] };
  let loseDevice;
  const listeners = new Map();
  const device = {
    limits: {
      maxBufferSize: 16 * 1024 * 1024,
      maxStorageBufferBindingSize: 16 * 1024 * 1024,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupSizeX: 256,
      maxStorageBuffersPerShaderStage: 8,
      maxBindGroups: 4,
      maxBindingsPerBindGroup: 1000,
      maxComputeWorkgroupsPerDimension: 65535,
      ...options.limits,
    },
    lost: new Promise(resolve => { loseDevice = resolve; }),
    addEventListener(name, callback) { listeners.set(name, callback); },
    removeEventListener(name) { listeners.delete(name); },
    destroy() { state.destroyed = true; loseDevice({ reason: 'destroyed', message: 'test device destroyed' }); },
    pushErrorScope(scope) { scopes.push(scope); },
    async popErrorScope() { scopes.pop(); return options.scopeError ? { message: 'test validation error' } : null; },
    createShaderModule(descriptor) {
      state.shader = descriptor.code;
      return { async getCompilationInfo() { return { messages: options.shaderErrors ?? [] }; } };
    },
    async createComputePipelineAsync() {
      if (options.pipelineError) throw new Error('test pipeline failure');
      return { getBindGroupLayout() { return {}; } };
    },
    createBuffer(descriptor) {
      const data = new ArrayBuffer(descriptor.size);
      let rejectMapping;
      const buffer = {
        ...descriptor,
        data,
        destroyed: false,
        mapped: false,
        destroy() { this.destroyed = true; rejectMapping?.(new Error('test mapping destroyed')); },
        async mapAsync() {
          if (options.mapError) throw new Error('test map failure');
          await new Promise((resolve, reject) => {
            rejectMapping = reject;
            setTimeout(resolve, options.mapDelay ?? 0);
          });
          this.mapped = true;
        },
        getMappedRange() { assert.ok(this.mapped); return data; },
        unmap() { this.mapped = false; },
      };
      buffers.push(buffer);
      return buffer;
    },
    createBindGroup(descriptor) { state.bindings = descriptor; return descriptor; },
    createCommandEncoder() {
      return {
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups(count) { state.workgroups = count; },
            end() {},
          };
        },
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          state.commands.push(() => new Uint8Array(destination.data, destinationOffset, size).set(new Uint8Array(source.data, sourceOffset, size)));
        },
        finish() { return {}; },
      };
    },
    queue: {
      writeBuffer(buffer, offset, data) { new Uint8Array(buffer.data, offset, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); },
      submit() {
        state.submissions += 1;
        const output = state.bindings.entries[2].resource.buffer;
        new Float32Array(output.data).fill(0.25);
        for (const command of state.commands.splice(0)) command();
      },
    },
  };
  state.device = device;
  state.lose = () => loseDevice({ reason: 'unknown', message: 'test device lost' });
  state.error = () => listeners.get('uncapturederror')?.({ preventDefault() {}, error: { message: 'test uncaptured error' } });
  const globals = {
    navigator: { gpu: { async requestAdapter() { return options.noAdapter ? null : { info: { isFallbackAdapter: options.fallback ?? false }, async requestDevice() { return device; } }; } } },
    GPUBufferUsage: { MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, STORAGE: 128 },
    GPUMapMode: { READ: 1 },
  };
  for (const [name, value] of Object.entries(globals)) {
    const previous = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, value });
    context.after(() => {
      if (previous) Object.defineProperty(globalThis, name, previous);
      else delete globalThis[name];
    });
  }
  return state;
}

test('GPU API rejects unavailable and fallback adapters without running a CPU renderer', async context => {
  for (const options of [{ noAdapter: true }, { fallback: true }]) {
    await context.test(JSON.stringify(options), async child => {
      const state = gpuDouble(child, options);
      await assert.rejects(createGpuRenderer(fixture()), /adapter/);
      assert.equal(state.buffers.length, 0);
    });
  }
  await context.test('no WebGPU API', async child => {
    gpuDouble(child);
    delete globalThis.navigator.gpu;
    await assert.rejects(createGpuRenderer(fixture()), /WebGPU is unavailable/);
  });
});

test('GPU setup rejects device limits, shader errors, pipeline errors, and scoped errors', async context => {
  for (const options of [
    { limits: { maxComputeInvocationsPerWorkgroup: 32 } },
    { limits: { maxComputeWorkgroupSizeX: 32 } },
    { limits: { maxStorageBuffersPerShaderStage: 2 } },
    { limits: { maxBufferSize: 4 } },
    { limits: { maxStorageBufferBindingSize: 4 } },
    { shaderErrors: [{ type: 'error', lineNum: 4, linePos: 2, message: 'test bad WGSL' }] },
    { pipelineError: true },
    { scopeError: true },
  ]) {
    await context.test(JSON.stringify(options), async child => {
      const state = gpuDouble(child, options);
      await assert.rejects(createGpuRenderer(fixture()), /WebGPU|shader|pipeline/);
      assert.equal(state.destroyed, true);
      assert.ok(state.buffers.every(buffer => buffer.destroyed));
      assert.equal(state.scopes.length, 0);
    });
  }
});

test('GPU lifecycle double verifies packed weights, completed readback timing, and cleanup, not shader numerics', async context => {
  const state = gpuDouble(context, { mapDelay: 30 });
  const artifact = brandFixture();
  artifact.layers[0].weights[0] = 0.1;
  const renderer = await createGpuRenderer(artifact);
  context.after(() => renderer.dispose());
  const expectedWeights = new Float32Array(artifact.layers.flatMap(entry => [...entry.weights, ...entry.bias]).concat(artifact.brand.color, artifact.brand.background));
  assert.deepEqual(new Float32Array(state.buffers[0].data), expectedWeights);
  artifact.brand.mode = 'neural';
  artifact.layers[0].weights[0] = 100;
  const pending = renderer.render([0.1, 0.2, 0.3, 0.4], 9, 9);
  await assert.rejects(renderer.render([0, 0, 0, 0], 1, 1), /in flight/);
  const result = await pending;
  assert.ok(result.pixels instanceof Float32Array);
  assert.equal(result.pixels.length, 9 * 9 * 4);
  assert.ok(result.pixels.every(value => value === 0.25));
  assert.ok(result.milliseconds >= 20, `readback wait was not included: ${result.milliseconds}`);
  assert.equal(state.workgroups, 2);
  const settingsBuffer = state.buffers.find(buffer => buffer.label === 'Student inputs');
  assert.deepEqual([...new Uint32Array(settingsBuffer.data, 0, 4)], [9, 9, 1, 0]);
  assert.deepEqual(new Float32Array(settingsBuffer.data, 16, 4), new Float32Array([0.1, 0.2, 0.3, 0.4]));
  assert.ok(state.buffers.slice(1).every(buffer => buffer.destroyed && !buffer.mapped));
  assert.equal(state.buffers[0].destroyed, false);
  await renderer.render([0, 0, 0, 0], 1, 1, { composite: false });
  const rawSettings = state.buffers.filter(buffer => buffer.label === 'Student inputs').at(-1);
  assert.equal(new Uint32Array(rawSettings.data)[2], 0);
  renderer.dispose();
  renderer.dispose();
  assert.ok(state.buffers.every(buffer => buffer.destroyed));
  assert.equal(state.destroyed, true);
  assert.equal(state.scopes.length, 0);
  await assert.rejects(renderer.render([0, 0, 0, 0], 1, 1), /disposed/);
});

test('GPU render rejects oversized dispatch/buffers and bad inputs before allocation', async context => {
  const state = gpuDouble(context);
  const renderer = await createGpuRenderer(fixture());
  context.after(() => renderer.dispose());
  await assert.rejects(renderer.render([0, 0, 0, NaN], 1, 1), /upl/);
  await assert.rejects(renderer.render([0, 0, 0, 0], 513, 1), /width/);
  state.device.limits.maxComputeWorkgroupsPerDimension = 1;
  await assert.rejects(renderer.render([0, 0, 0, 0], 9, 9), /maxComputeWorkgroupsPerDimension/);
  state.device.limits.maxBufferSize = 128;
  await assert.rejects(renderer.render([0, 0, 0, 0], 8, 8), /maxBufferSize/);
  assert.equal(state.buffers.length, 1);
});

test('failed mapping cleans temporary buffers and permits a later retry', async context => {
  const options = { mapError: true };
  const state = gpuDouble(context, options);
  const renderer = await createGpuRenderer(fixture());
  context.after(() => renderer.dispose());
  await assert.rejects(renderer.render([0, 0, 0, 0], 1, 1), /map failure/);
  assert.ok(state.buffers.slice(1).every(buffer => buffer.destroyed));
  assert.equal(state.scopes.length, 0);
  options.mapError = false;
  await renderer.render([0, 0, 0, 0], 1, 1);
});

test('device loss and uncaptured errors close the renderer and destroy owned buffers', async context => {
  for (const failure of ['lose', 'error']) {
    await context.test(failure, async child => {
      const state = gpuDouble(child);
      const renderer = await createGpuRenderer(fixture());
      state[failure]();
      await Promise.resolve();
      await assert.rejects(renderer.render([0, 0, 0, 0], 1, 1), /device lost|uncaptured error/);
      assert.ok(state.buffers.every(buffer => buffer.destroyed));
      assert.equal(state.destroyed, true);
      renderer.dispose();
    });
  }
});

test('disposing during readback rejects pending work and releases every owned buffer', async context => {
  const state = gpuDouble(context, { mapDelay: 30 });
  const renderer = await createGpuRenderer(fixture());
  const pending = renderer.render([0, 0, 0, 0], 2, 2);
  renderer.dispose();
  await assert.rejects(pending, /destroyed|disposed/);
  assert.ok(state.buffers.every(buffer => buffer.destroyed));
  assert.equal(state.scopes.length, 0);
});
