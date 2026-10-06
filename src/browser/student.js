import { validateDiffusion, renderDiffusionCpu, createDiffusionGpuRenderer } from './diffusion.js';

const INPUT_ORDER = ['x', 'y', 'upl.0', 'upl.1', 'upl.2', 'upl.3'];
const GLYPH_ROWS = [14, 17, 17, 31, 17, 17, 17];
const OUTSIDE_PLATE = 0xffffffff;
const WORKGROUP_SIZE = 64;

function record(value, path, fields) {
  if (value === null || typeof value !== 'object' || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) {
    throw new TypeError(`${path} must be an object`);
  }
  if (fields) {
    for (const key of Reflect.ownKeys(value)) {
      if (!fields.includes(key)) throw new TypeError(`${path} has unknown field ${String(key)}`);
    }
    for (const key of fields) {
      if (!Object.hasOwn(value, key)) throw new TypeError(`${path}.${key} is required`);
    }
  }
}

function bounded(value, minimum, maximum, path) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new TypeError(`${path} must be finite and within [${minimum}, ${maximum}]`);
  }
}

function integer(value, minimum, maximum, path) {
  bounded(value, minimum, maximum, path);
  if (!Number.isInteger(value)) throw new TypeError(`${path} must be an integer`);
}

function vector(value, length, minimum, maximum, path) {
  if (!Array.isArray(value) || value.length !== length) throw new TypeError(`${path} must contain ${length} numbers`);
  for (let index = 0; index < length; index += 1) bounded(value[index], minimum, maximum, `${path}[${index}]`);
}

function identifier(value, path) {
  if (typeof value !== 'string' || value.trim().length === 0) throw new TypeError(`${path} must be a nonempty string`);
}

function metadata(value, path, depth = 0, budget = { remaining: 4096 }) {
  budget.remaining -= 1;
  if (depth > 8 || budget.remaining < 0) throw new TypeError(`${path} metadata is too large or deeply nested`);
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number' && Number.isFinite(value)) return;
  if (Array.isArray(value)) {
    if (value.length > budget.remaining) throw new TypeError(`${path} metadata is too large`);
    for (const entry of value) metadata(entry, path, depth + 1, budget);
    return;
  }
  record(value, path);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string') throw new TypeError(`${path} metadata must be JSON data`);
    metadata(value[key], `${path}.${key}`, depth + 1, budget);
  }
}

export function validateStudent(artifact) {
  if (artifact?.format === 'angelos.diffusion.v1') return validateDiffusion(artifact);
  record(artifact, 'student', ['format', 'id', 'campaignId', 'graphHash', 'encoder', 'inputOrder', 'layers', 'brand', 'metrics', 'training']);
  if (artifact.format !== 'angelos.student.v1') throw new TypeError('Unsupported student format');
  identifier(artifact.id, 'student.id');
  if (typeof artifact.campaignId !== 'string' || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(artifact.campaignId)) {
    throw new TypeError('student.campaignId must be a graph ID');
  }
  if (typeof artifact.graphHash !== 'string' || !/^[a-f0-9]{64}$/.test(artifact.graphHash)) {
    throw new TypeError('student.graphHash must be a lowercase SHA-256 digest');
  }
  record(artifact.encoder, 'student.encoder', ['id', 'dimension']);
  if (artifact.encoder.id !== 'structured-demo-v1' || artifact.encoder.dimension !== 4) throw new TypeError('Unsupported student encoder');
  if (!Array.isArray(artifact.inputOrder) || artifact.inputOrder.length !== INPUT_ORDER.length || INPUT_ORDER.some((name, index) => artifact.inputOrder[index] !== name)) {
    throw new TypeError('student.inputOrder does not match the six-input contract');
  }
  if (!Array.isArray(artifact.layers) || artifact.layers.length !== 3) throw new TypeError('student.layers must contain exactly three layers');
  let previousWidth = 6;
  for (let index = 0; index < 3; index += 1) {
    const layer = artifact.layers[index];
    const path = `student.layers[${index}]`;
    record(layer, path, ['input', 'output', 'activation', 'weights', 'bias']);
    if (layer.input !== previousWidth) throw new TypeError(`${path}.input does not match the preceding width`);
    integer(layer.output, 1, index === 2 ? 3 : 64, `${path}.output`);
    if (index === 2 && layer.output !== 3) throw new TypeError('The final layer must have three outputs');
    if (layer.activation !== (index === 2 ? 'sigmoid' : 'relu')) throw new TypeError(`${path}.activation is unsupported`);
    vector(layer.weights, layer.input * layer.output, -1000, 1000, `${path}.weights`);
    vector(layer.bias, layer.output, -1000, 1000, `${path}.bias`);
    previousWidth = layer.output;
  }
  if (artifact.brand !== null) {
    const brand = artifact.brand;
    record(brand, 'student.brand', ['mode', 'x', 'y', 'cell', 'color', 'background']);
    if (!['neural', 'composite'].includes(brand.mode)) throw new TypeError('student.brand.mode is unsupported');
    bounded(brand.x, 0, 1, 'student.brand.x');
    bounded(brand.y, 0, 1, 'student.brand.y');
    bounded(brand.cell, 0.005, 0.05, 'student.brand.cell');
    if (brand.x + 7 * brand.cell > 1 || brand.y + 9 * brand.cell > 1) throw new TypeError('student.brand plate must fit inside the image');
    vector(brand.color, 3, 0, 1, 'student.brand.color');
    vector(brand.background, 3, 0, 1, 'student.brand.background');
  }
  record(artifact.metrics, 'student.metrics');
  metadata(artifact.metrics, 'student.metrics');
  for (const name of ['baselineMse', 'heldoutMse', 'brandTolerance']) {
    bounded(artifact.metrics[name], 0, Number.MAX_VALUE, `student.metrics.${name}`);
  }
  if (artifact.brand !== null || artifact.metrics.brandMaxError !== null) {
    bounded(artifact.metrics.brandMaxError, 0, Number.MAX_VALUE, 'student.metrics.brandMaxError');
  }
  record(artifact.training, 'student.training');
  metadata(artifact.training, 'student.training');
  integer(artifact.training.seed, 0, Number.MAX_SAFE_INTEGER, 'student.training.seed');
  integer(artifact.training.steps, 0, Number.MAX_SAFE_INTEGER, 'student.training.steps');
  identifier(artifact.training.device, 'student.training.device');
  return artifact;
}

function snapshotStudent(artifact) {
  validateStudent(artifact);
  const values = [];
  const layers = artifact.layers.map(layer => {
    const weightsOffset = values.length;
    values.push(...layer.weights);
    const biasOffset = values.length;
    values.push(...layer.bias);
    return { input: layer.input, output: layer.output, weightsOffset, biasOffset };
  });
  const brandOffset = values.length;
  const brand = artifact.brand === null ? null : {
    ...artifact.brand,
    color: [...artifact.brand.color],
    background: [...artifact.brand.background],
  };
  values.push(...(brand?.color ?? [0, 0, 0]), ...(brand?.background ?? [0, 0, 0]));
  return { layers, weights: new Float32Array(values), brand, brandOffset };
}

function renderInputs(upl, width, height, options) {
  vector(upl, 4, 0, 1, 'upl');
  integer(width, 1, 512, 'width');
  integer(height, 1, 512, 'height');
  record(options, 'render options');
  if (Reflect.ownKeys(options).some(key => key !== 'composite')) throw new TypeError('Unknown render option');
  const composite = options.composite === undefined ? true : options.composite;
  if (typeof composite !== 'boolean') throw new TypeError('composite must be a boolean');
  return composite;
}

function plateIndex(coordinate, origin, cell, count) {
  const index = Math.floor((coordinate - origin) / cell);
  return index >= 0 && index < count ? index : OUTSIDE_PLATE;
}

function renderSettings(model, upl, width, height, composite) {
  const settings = new Uint32Array(8 + width + height);
  settings.set([width, height, Number(composite && model.brand?.mode === 'composite'), 0]);
  new Float32Array(settings.buffer, 16, 4).set(upl);
  if (settings[2]) {
    for (let column = 0; column < width; column += 1) {
      settings[8 + column] = plateIndex((column + 0.5) / width, model.brand.x, model.brand.cell, 7);
    }
    for (let row = 0; row < height; row += 1) {
      settings[8 + width + row] = plateIndex((row + 0.5) / height, model.brand.y, model.brand.cell, 9);
    }
  }
  return settings;
}

function sigmoid(value) {
  const exponential = Math.fround(Math.exp(-Math.abs(value)));
  const denominator = Math.fround(1 + exponential);
  return Math.fround(value >= 0 ? 1 / denominator : exponential / denominator);
}

export function renderCpu(artifact, upl, width, height, options = {}) {
  if (artifact?.format === 'angelos.diffusion.v1') return renderDiffusionCpu(artifact, upl, width, height, options);
  const composite = renderInputs(upl, width, height, options);
  const model = snapshotStudent(artifact);
  const settings = renderSettings(model, upl, width, height, composite);
  const features = new Float32Array([0, 0, ...upl]);
  const firstHidden = new Float32Array(model.layers[0].output);
  const secondHidden = new Float32Array(model.layers[1].output);
  const channels = new Float32Array(3);
  const activations = [features, firstHidden, secondHidden, channels];
  const pixels = new Float32Array(width * height * 4);
  for (let row = 0; row < height; row += 1) {
    features[1] = (row + 0.5) / height;
    for (let column = 0; column < width; column += 1) {
      features[0] = (column + 0.5) / width;
      for (let layerIndex = 0; layerIndex < 3; layerIndex += 1) {
        const layer = model.layers[layerIndex];
        const input = activations[layerIndex];
        const output = activations[layerIndex + 1];
        for (let channel = 0; channel < layer.output; channel += 1) {
          let total = model.weights[layer.biasOffset + channel];
          for (let feature = 0; feature < layer.input; feature += 1) {
            total = Math.fround(total + Math.fround(model.weights[layer.weightsOffset + channel * layer.input + feature] * input[feature]));
          }
          output[channel] = layerIndex === 2 ? sigmoid(total) : Math.max(total, 0);
        }
      }
      const pixelOffset = (row * width + column) * 4;
      pixels.set(channels, pixelOffset);
      pixels[pixelOffset + 3] = 1;
      const plateColumn = settings[8 + column];
      const plateRow = settings[8 + width + row];
      if (settings[2] && plateColumn < 7 && plateRow < 9) {
        const ink = plateColumn >= 1 && plateColumn <= 5 && plateRow >= 1 && plateRow <= 7 && ((GLYPH_ROWS[plateRow - 1] >> (5 - plateColumn)) & 1) === 1;
        const colorOffset = model.brandOffset + (ink ? 0 : 3);
        pixels.set(model.weights.subarray(colorOffset, colorOffset + 3), pixelOffset);
      }
    }
  }
  return pixels;
}

function shaderSource(model) {
  const [firstLayer, secondLayer, finalLayer] = model.layers;
  return `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> settings: array<u32>;
@group(0) @binding(2) var<storage, read_write> pixels: array<vec4<f32>>;

fn sigmoid(value: f32) -> f32 {
  let exponential = exp(-abs(value));
  if (value >= 0.0) { return 1.0 / (1.0 + exponential); }
  return exponential / (1.0 + exponential);
}

@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let pixelIndex = invocation.x;
  let width = settings[0];
  let height = settings[1];
  if (pixelIndex >= width * height) { return; }
  let column = pixelIndex % width;
  let row = pixelIndex / width;
  var features: array<f32, 6>;
  features[0] = (f32(column) + 0.5) / f32(width);
  features[1] = (f32(row) + 0.5) / f32(height);
  for (var feature = 0u; feature < 4u; feature += 1u) {
    features[feature + 2u] = bitcast<f32>(settings[feature + 4u]);
  }
  var firstHidden: array<f32, 64>;
  var secondHidden: array<f32, 64>;
  var channels: array<f32, 3>;
  for (var channel = 0u; channel < ${firstLayer.output}u; channel += 1u) {
    var total = weights[${firstLayer.biasOffset}u + channel];
    for (var feature = 0u; feature < 6u; feature += 1u) {
      total += weights[${firstLayer.weightsOffset}u + channel * 6u + feature] * features[feature];
    }
    firstHidden[channel] = max(total, 0.0);
  }
  for (var channel = 0u; channel < ${secondLayer.output}u; channel += 1u) {
    var total = weights[${secondLayer.biasOffset}u + channel];
    for (var feature = 0u; feature < ${secondLayer.input}u; feature += 1u) {
      total += weights[${secondLayer.weightsOffset}u + channel * ${secondLayer.input}u + feature] * firstHidden[feature];
    }
    secondHidden[channel] = max(total, 0.0);
  }
  for (var channel = 0u; channel < 3u; channel += 1u) {
    var total = weights[${finalLayer.biasOffset}u + channel];
    for (var feature = 0u; feature < ${finalLayer.input}u; feature += 1u) {
      total += weights[${finalLayer.weightsOffset}u + channel * ${finalLayer.input}u + feature] * secondHidden[feature];
    }
    channels[channel] = sigmoid(total);
  }
  if (settings[2] != 0u) {
    let plateColumn = settings[8u + column];
    let plateRow = settings[8u + width + row];
    if (plateColumn < 7u && plateRow < 9u) {
      var colorOffset = ${model.brandOffset + 3}u;
      if (plateColumn >= 1u && plateColumn <= 5u && plateRow >= 1u && plateRow <= 7u) {
        let glyphRows = array<u32, 7>(14u, 17u, 17u, 31u, 17u, 17u, 17u);
        if (((glyphRows[plateRow - 1u] >> (5u - plateColumn)) & 1u) != 0u) {
          colorOffset = ${model.brandOffset}u;
        }
      }
      for (var channel = 0u; channel < 3u; channel += 1u) {
        channels[channel] = weights[colorOffset + channel];
      }
    }
  }
  pixels[pixelIndex] = vec4<f32>(channels[0], channels[1], channels[2], 1.0);
}`;
}

function requireLimit(device, name, required) {
  if (!Number.isFinite(device.limits[name]) || device.limits[name] < required) {
    throw new Error(`WebGPU ${name} must be at least ${required}`);
  }
}

function bufferLimits(device, size) {
  requireLimit(device, 'maxBufferSize', size);
  requireLimit(device, 'maxStorageBufferBindingSize', size);
}

async function checkedGpu(device, operation) {
  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  let result;
  let failure;
  try {
    result = await operation();
  } catch (error) {
    failure = error;
  }
  for (let scope = 0; scope < 2; scope += 1) {
    try {
      const error = await device.popErrorScope();
      if (error && !failure) failure = new Error(`WebGPU error: ${error.message}`);
    } catch (error) {
      failure ??= error;
    }
  }
  if (failure) throw failure;
  return result;
}

export async function createGpuRenderer(artifact) {
  if (artifact?.format === 'angelos.diffusion.v1') return createDiffusionGpuRenderer(artifact);
  const model = snapshotStudent(artifact);
  const gpu = globalThis.navigator?.gpu;
  if (!gpu) throw new Error('WebGPU is unavailable; no CPU fallback was used');
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter is available');
  if (adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true) throw new Error('A fallback WebGPU adapter is not supported');
  const device = await adapter.requestDevice();
  const buffers = new Set();
  let closedError = null;
  let rendering = false;
  let pipeline;
  let weightsBuffer;
  const usage = globalThis.GPUBufferUsage;
  const mapMode = globalThis.GPUMapMode;

  function close(error) {
    if (closedError) return;
    closedError = error;
    device.removeEventListener('uncapturederror', onError);
    for (const buffer of buffers) buffer.destroy();
    buffers.clear();
    device.destroy();
  }

  function onError(event) {
    event.preventDefault();
    close(new Error(`WebGPU uncaptured error: ${event.error.message}`));
  }

  function active() {
    if (closedError) throw closedError;
  }

  function allocate(label, size, bufferUsage, temporary) {
    active();
    const buffer = device.createBuffer({ label, size, usage: bufferUsage });
    buffers.add(buffer);
    temporary?.push(buffer);
    return buffer;
  }

  try {
    device.addEventListener('uncapturederror', onError);
    device.lost.then(info => close(new Error(`WebGPU device lost: ${info.message || info.reason}`)), error => close(error));
    if (!usage || !mapMode) throw new Error('WebGPU buffer APIs are unavailable');
    requireLimit(device, 'maxComputeInvocationsPerWorkgroup', WORKGROUP_SIZE);
    requireLimit(device, 'maxComputeWorkgroupSizeX', WORKGROUP_SIZE);
    requireLimit(device, 'maxStorageBuffersPerShaderStage', 3);
    requireLimit(device, 'maxBindGroups', 1);
    requireLimit(device, 'maxBindingsPerBindGroup', 3);
    bufferLimits(device, model.weights.byteLength);
    await checkedGpu(device, async () => {
      const module = device.createShaderModule({ label: 'Angelos student v1', code: shaderSource(model) });
      const compilation = await module.getCompilationInfo();
      const errors = compilation.messages.filter(message => message.type === 'error');
      if (errors.length) throw new Error(`Student shader compilation failed: ${errors.map(error => `${error.lineNum}:${error.linePos} ${error.message}`).join('; ')}`);
      active();
      pipeline = await device.createComputePipelineAsync({ label: 'Angelos student v1', layout: 'auto', compute: { module, entryPoint: 'main' } });
      weightsBuffer = allocate('Student weights', model.weights.byteLength, usage.STORAGE | usage.COPY_DST);
      device.queue.writeBuffer(weightsBuffer, 0, model.weights);
    });
    active();
  } catch (error) {
    close(error);
    throw error;
  }

  return {
    async render(upl, width, height, options = {}) {
      active();
      const composite = renderInputs(upl, width, height, options);
      if (rendering) throw new Error('A student render is already in flight');
      const settings = renderSettings(model, upl, width, height, composite);
      const outputBytes = width * height * 16;
      bufferLimits(device, settings.byteLength);
      bufferLimits(device, outputBytes);
      const workgroups = Math.ceil(width * height / WORKGROUP_SIZE);
      requireLimit(device, 'maxComputeWorkgroupsPerDimension', workgroups);
      rendering = true;
      const temporary = [];
      try {
        const result = await checkedGpu(device, async () => {
          const started = performance.now();
          const settingsBuffer = allocate('Student inputs', settings.byteLength, usage.STORAGE | usage.COPY_DST, temporary);
          const outputBuffer = allocate('Student RGBA', outputBytes, usage.STORAGE | usage.COPY_SRC, temporary);
          const readbackBuffer = allocate('Student readback', outputBytes, usage.COPY_DST | usage.MAP_READ, temporary);
          device.queue.writeBuffer(settingsBuffer, 0, settings);
          const bindings = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: weightsBuffer } },
              { binding: 1, resource: { buffer: settingsBuffer } },
              { binding: 2, resource: { buffer: outputBuffer } },
            ],
          });
          const encoder = device.createCommandEncoder({ label: 'Student render and readback' });
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindings);
          pass.dispatchWorkgroups(workgroups);
          pass.end();
          encoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, outputBytes);
          device.queue.submit([encoder.finish()]);
          await readbackBuffer.mapAsync(mapMode.READ);
          let pixels;
          try {
            active();
            pixels = new Float32Array(readbackBuffer.getMappedRange().slice(0));
          } finally {
            readbackBuffer.unmap();
          }
          return { pixels, milliseconds: performance.now() - started };
        });
        active();
        return result;
      } finally {
        for (const buffer of temporary) {
          if (buffers.delete(buffer)) buffer.destroy();
        }
        rendering = false;
      }
    },
    dispose() {
      close(new Error('Student GPU renderer is disposed'));
    },
  };
}
