const RESOLUTION = 32;
const AREA = RESOLUTION * RESOLUTION;
const STEPS = 16;
const WORKGROUP_SIZE = 64;
const MAX_DISPLAY = 512;
const GLYPH_ROWS = [14, 17, 17, 31, 17, 17, 17];
const CHANNELS = [10, 16, 32, 32, 16, 3];
const LAYERS = [[10, 16, 1], [16, 32, 2], [32, 32, 1], [48, 16, 1], [16, 3, 1]];
const round = Math.fround;

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

function identifier(value, path, maximum = 4096) {
  if (typeof value !== 'string' || value.trim().length === 0 || value.length > maximum) throw new TypeError(`${path} must be a nonempty string of at most ${maximum} characters`);
}

function metadata(value, path, depth = 0, budget = { remaining: 4096 }) {
  budget.remaining -= 1;
  if (depth > 8 || budget.remaining < 0) throw new TypeError(`${path} metadata is too large or deeply nested`);
  if (value === null || typeof value === 'boolean') return;
  if (typeof value === 'string' && value.length <= 16384) return;
  if (typeof value === 'number' && Number.isFinite(value)) return;
  if (Array.isArray(value)) {
    if (value.length > budget.remaining) throw new TypeError(`${path} metadata is too large`);
    for (const entry of value) metadata(entry, path, depth + 1, budget);
    return;
  }
  record(value, path);
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string' || key.length > 4096) throw new TypeError(`${path} metadata must be bounded JSON data`);
    metadata(value[key], `${path}.${key}`, depth + 1, budget);
  }
}

export function validateDiffusion(artifact) {
  record(artifact, 'diffusion', ['format', 'id', 'campaignId', 'graphHash', 'encoder', 'architecture', 'layers', 'schedule', 'brand', 'metrics', 'training']);
  if (artifact.format !== 'angelos.diffusion.v1') throw new TypeError('Unsupported diffusion format');
  identifier(artifact.id, 'diffusion.id', 128);
  if (typeof artifact.campaignId !== 'string' || artifact.campaignId.length > 64 || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(artifact.campaignId)) {
    throw new TypeError('diffusion.campaignId must be a graph ID');
  }
  if (typeof artifact.graphHash !== 'string' || !/^[a-f0-9]{64}$/.test(artifact.graphHash)) throw new TypeError('diffusion.graphHash must be a lowercase SHA-256 digest');
  record(artifact.encoder, 'diffusion.encoder', ['id', 'dimension']);
  if (artifact.encoder.id !== 'structured-demo-v1' || artifact.encoder.dimension !== 4) throw new TypeError('Unsupported diffusion encoder');
  record(artifact.architecture, 'diffusion.architecture', ['name', 'resolution', 'channels']);
  if (artifact.architecture.name !== 'conditional-unet-v1' || artifact.architecture.resolution !== RESOLUTION) throw new TypeError('Unsupported diffusion architecture');
  vector(artifact.architecture.channels, CHANNELS.length, 1, 48, 'diffusion.architecture.channels');
  if (CHANNELS.some((width, index) => artifact.architecture.channels[index] !== width)) throw new TypeError('Unsupported diffusion architecture channels');
  if (!Array.isArray(artifact.layers) || artifact.layers.length !== LAYERS.length) throw new TypeError('diffusion.layers must contain exactly five convolutions');
  for (let index = 0; index < LAYERS.length; index += 1) {
    const layer = artifact.layers[index];
    const path = `diffusion.layers[${index}]`;
    const [input, output, stride] = LAYERS[index];
    record(layer, path, ['input', 'output', 'kernel', 'stride', 'activation', 'weights', 'bias']);
    if (layer.input !== input || layer.output !== output || layer.kernel !== 3 || layer.stride !== stride) throw new TypeError(`${path} has an unsupported convolution shape`);
    if (layer.activation !== (index === 4 ? 'linear' : 'relu')) throw new TypeError(`${path}.activation is unsupported`);
    vector(layer.weights, input * output * 9, -1000, 1000, `${path}.weights`);
    vector(layer.bias, output, -1000, 1000, `${path}.bias`);
  }
  record(artifact.schedule, 'diffusion.schedule', ['predictionType', 'alphaCumprod', 'timesteps']);
  if (artifact.schedule.predictionType !== 'epsilon') throw new TypeError('Unsupported diffusion schedule predictionType');
  vector(artifact.schedule.alphaCumprod, 64, 0, 1, 'diffusion.schedule.alphaCumprod');
  let previousAlpha = 1;
  for (const alpha of artifact.schedule.alphaCumprod) {
    const rounded = round(alpha);
    if (rounded <= 0 || rounded >= previousAlpha) throw new TypeError('diffusion.schedule.alphaCumprod must be strictly decreasing within (0,1) in float32');
    previousAlpha = rounded;
  }
  vector(artifact.schedule.timesteps, STEPS, 0, 63, 'diffusion.schedule.timesteps');
  if (artifact.schedule.timesteps[0] !== 63 || artifact.schedule.timesteps.at(-1) !== 0) throw new TypeError('diffusion.schedule.timesteps must start at 63 and end at 0');
  let previousTime = 64;
  for (const timestep of artifact.schedule.timesteps) {
    if (!Number.isInteger(timestep) || timestep >= previousTime) throw new TypeError('diffusion.schedule.timesteps must be strictly descending integers');
    previousTime = timestep;
  }
  if (artifact.brand !== null) {
    const brand = artifact.brand;
    record(brand, 'diffusion.brand', ['mode', 'x', 'y', 'cell', 'color', 'background']);
    if (!['neural', 'composite'].includes(brand.mode)) throw new TypeError('diffusion.brand.mode is unsupported');
    bounded(brand.x, 0, 1, 'diffusion.brand.x');
    bounded(brand.y, 0, 1, 'diffusion.brand.y');
    bounded(brand.cell, 0.005, 0.05, 'diffusion.brand.cell');
    if (brand.x + 7 * brand.cell > 1 || brand.y + 9 * brand.cell > 1) throw new TypeError('diffusion.brand plate must fit inside the image');
    vector(brand.color, 3, 0, 1, 'diffusion.brand.color');
    vector(brand.background, 3, 0, 1, 'diffusion.brand.background');
  }
  record(artifact.metrics, 'diffusion.metrics');
  metadata(artifact.metrics, 'diffusion.metrics');
  for (const name of ['baselineMse', 'heldoutMse', 'brandTolerance']) bounded(artifact.metrics[name], 0, Number.MAX_VALUE, `diffusion.metrics.${name}`);
  if (artifact.brand !== null || artifact.metrics.brandMaxError !== null) bounded(artifact.metrics.brandMaxError, 0, Number.MAX_VALUE, 'diffusion.metrics.brandMaxError');
  record(artifact.training, 'diffusion.training');
  metadata(artifact.training, 'diffusion.training');
  integer(artifact.training.seed, 0, 0xffffffff, 'diffusion.training.seed');
  integer(artifact.training.steps, 0, Number.MAX_SAFE_INTEGER, 'diffusion.training.steps');
  identifier(artifact.training.device, 'diffusion.training.device');
  return artifact;
}

function snapshot(artifact) {
  validateDiffusion(artifact);
  const values = [];
  const layers = artifact.layers.map(layer => {
    const weightsOffset = values.length;
    values.push(...layer.weights);
    const biasOffset = values.length;
    values.push(...layer.bias);
    return { input: layer.input, output: layer.output, stride: layer.stride, weightsOffset, biasOffset };
  });
  const scheduleOffset = values.length;
  for (let step = 0; step < STEPS; step += 1) {
    const timestep = artifact.schedule.timesteps[step];
    const alpha = round(artifact.schedule.alphaCumprod[timestep]);
    const nextAlpha = step + 1 === STEPS ? 1 : round(artifact.schedule.alphaCumprod[artifact.schedule.timesteps[step + 1]]);
    values.push(Math.sqrt(alpha), Math.sqrt(round(1 - alpha)), Math.sqrt(nextAlpha), Math.sqrt(round(1 - nextAlpha)), timestep / 63);
  }
  const brand = artifact.brand === null ? null : { ...artifact.brand, color: [...artifact.brand.color], background: [...artifact.brand.background] };
  const brandOffset = values.length;
  values.push(...(brand?.color ?? [0, 0, 0]), ...(brand?.background ?? [0, 0, 0]));
  return { layers, weights: new Float32Array(values), scheduleOffset, brand, brandOffset };
}

function renderInputs(upl, width, height, options) {
  vector(upl, 4, 0, 1, 'upl');
  integer(width, 1, MAX_DISPLAY, 'width');
  integer(height, 1, MAX_DISPLAY, 'height');
  record(options, 'render options');
  if (Reflect.ownKeys(options).some(key => key !== 'composite' && key !== 'seed')) throw new TypeError('Unknown render option');
  const composite = options.composite === undefined ? true : options.composite;
  const seed = options.seed === undefined ? 2026 : options.seed;
  if (typeof composite !== 'boolean') throw new TypeError('composite must be a boolean');
  integer(seed, 0, 0xffffffff, 'seed');
  return { composite, seed };
}

function noise(seed) {
  let state = seed;
  const samples = new Float32Array(3 * AREA);
  for (let index = 0; index < samples.length; index += 2) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    const uniform = (state + 0.5) / 4294967296;
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    const angle = 2 * Math.PI * ((state + 0.5) / 4294967296);
    const radius = Math.sqrt(-2 * Math.log(uniform));
    samples[index] = radius * Math.cos(angle);
    samples[index + 1] = radius * Math.sin(angle);
  }
  return samples;
}

function settingsFor(model, upl, width, height, composite) {
  const settings = new Uint32Array(8 + width + height);
  settings.set([width, height, Number(composite && model.brand?.mode === 'composite'), 0]);
  new Float32Array(settings.buffer, 16, 4).set(upl);
  if (settings[2]) {
    for (let axis = 0; axis < 2; axis += 1) {
      const size = axis === 0 ? width : height;
      const origin = axis === 0 ? model.brand.x : model.brand.y;
      const count = axis === 0 ? 7 : 9;
      const lower = round(origin);
      const upper = round(origin + count * model.brand.cell);
      const cellSize = round(model.brand.cell);
      const offset = axis === 0 ? 8 : 8 + width;
      for (let pixel = 0; pixel < size; pixel += 1) {
        const coordinate = round((pixel + 0.5) / size);
        const cell = Math.floor(round(round(coordinate - lower) / cellSize));
        settings[offset + pixel] = coordinate >= lower && coordinate < upper ? Math.min(cell, count - 1) : 0xffffffff;
      }
    }
  }
  return settings;
}

function convolution(model, layer, input, inputSize, output, outputSize) {
  const inputArea = inputSize * inputSize;
  const outputArea = outputSize * outputSize;
  for (let channel = 0; channel < layer.output; channel += 1) {
    const taps = [];
    for (let feature = 0; feature < layer.input; feature += 1) {
      for (let kernelRow = 0; kernelRow < 3; kernelRow += 1) {
        for (let kernelColumn = 0; kernelColumn < 3; kernelColumn += 1) {
          const weight = model.weights[layer.weightsOffset + ((channel * layer.input + feature) * 3 + kernelRow) * 3 + kernelColumn];
          if (weight !== 0) taps.push({ featureOffset: feature * inputArea, row: kernelRow - 1, column: kernelColumn - 1, weight });
        }
      }
    }
    for (let row = 0; row < outputSize; row += 1) {
      for (let column = 0; column < outputSize; column += 1) {
        let total = model.weights[layer.biasOffset + channel];
        for (const tap of taps) {
          const sourceRow = row * layer.stride + tap.row;
          const sourceColumn = column * layer.stride + tap.column;
          if (sourceRow >= 0 && sourceRow < inputSize && sourceColumn >= 0 && sourceColumn < inputSize) {
            total = round(total + round(tap.weight * input[tap.featureOffset + sourceRow * inputSize + sourceColumn]));
          }
        }
        output[channel * outputArea + row * outputSize + column] = layer.output === 3 ? total : Math.max(0, total);
      }
    }
  }
}

export function renderDiffusionCpu(artifact, upl, width, height, options = {}) {
  const { composite, seed } = renderInputs(upl, width, height, options);
  const model = snapshot(artifact);
  const settings = settingsFor(model, upl, width, height, composite);
  const sample = noise(seed);
  const features = new Float32Array(10 * AREA);
  const hidden = [new Float32Array(16 * AREA), new Float32Array(32 * 256), new Float32Array(32 * 256), new Float32Array(16 * AREA)];
  const joined = new Float32Array(48 * AREA);
  const epsilon = new Float32Array(3 * AREA);
  for (let feature = 0; feature < 4; feature += 1) features.fill(round(upl[feature]), (3 + feature) * AREA, (4 + feature) * AREA);
  for (let pixel = 0; pixel < AREA; pixel += 1) {
    features[8 * AREA + pixel] = (pixel % RESOLUTION + 0.5) / RESOLUTION;
    features[9 * AREA + pixel] = (Math.floor(pixel / RESOLUTION) + 0.5) / RESOLUTION;
  }
  for (let step = 0; step < STEPS; step += 1) {
    const schedule = model.scheduleOffset + step * 5;
    features.set(sample);
    features.fill(model.weights[schedule + 4], 7 * AREA, 8 * AREA);
    convolution(model, model.layers[0], features, 32, hidden[0], 32);
    convolution(model, model.layers[1], hidden[0], 32, hidden[1], 16);
    convolution(model, model.layers[2], hidden[1], 16, hidden[2], 16);
    for (let channel = 0; channel < 32; channel += 1) {
      for (let pixel = 0; pixel < AREA; pixel += 1) joined[channel * AREA + pixel] = hidden[2][channel * 256 + Math.floor(Math.floor(pixel / 32) / 2) * 16 + Math.floor((pixel % 32) / 2)];
    }
    joined.set(hidden[0], 32 * AREA);
    convolution(model, model.layers[3], joined, 32, hidden[3], 32);
    convolution(model, model.layers[4], hidden[3], 32, epsilon, 32);
    for (let index = 0; index < sample.length; index += 1) {
      const predicted = round(round(sample[index] - round(model.weights[schedule + 1] * epsilon[index])) / model.weights[schedule]);
      const clean = Math.max(-1, Math.min(1, predicted));
      sample[index] = round(round(model.weights[schedule + 2] * clean) + round(model.weights[schedule + 3] * epsilon[index]));
    }
  }
  const pixels = new Float32Array(width * height * 4);
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const source = Math.floor(row * RESOLUTION / height) * RESOLUTION + Math.floor(column * RESOLUTION / width);
      const offset = (row * width + column) * 4;
      for (let channel = 0; channel < 3; channel += 1) pixels[offset + channel] = round(round(sample[channel * AREA + source] + 1) * 0.5);
      pixels[offset + 3] = 1;
      const plateColumn = settings[8 + column];
      const plateRow = settings[8 + width + row];
      if (settings[2] && plateColumn < 7 && plateRow < 9) {
        const ink = plateColumn >= 1 && plateColumn <= 5 && plateRow >= 1 && plateRow <= 7 && ((GLYPH_ROWS[plateRow - 1] >> (5 - plateColumn)) & 1) === 1;
        const colorOffset = model.brandOffset + (ink ? 0 : 3);
        pixels.set(model.weights.subarray(colorOffset, colorOffset + 3), offset);
      }
    }
  }
  if (!pixels.every(Number.isFinite)) throw new Error('Diffusion CPU reference produced nonfinite pixels');
  return pixels;
}

const BINDINGS = `
@group(0) @binding(0) var<storage, read> weights: array<f32>;
@group(0) @binding(1) var<storage, read> source: array<f32>;
@group(0) @binding(2) var<storage, read> skip: array<f32>;
@group(0) @binding(3) var<storage, read> settings: array<u32>;
@group(0) @binding(4) var<storage, read_write> destination: array<f32>;
`;

function convolutionShader(model, index) {
  const layer = model.layers[index];
  const outputSize = index === 1 || index === 2 ? 16 : 32;
  const inputSize = index === 2 ? 16 : 32;
  let featureRead = `return source[feature * ${inputSize * inputSize}u + u32(row) * ${inputSize}u + u32(column)];`;
  if (index === 0) featureRead = `
  if (feature < 3u) { return source[feature * 1024u + u32(row) * 32u + u32(column)]; }
  if (feature < 7u) { return bitcast<f32>(settings[feature + 1u]); }
  if (feature == 7u) { return weights[${model.scheduleOffset + 4}u + settings[3] * 5u]; }
  if (feature == 8u) { return (f32(column) + 0.5) / 32.0; }
  return (f32(row) + 0.5) / 32.0;`;
  if (index === 3) featureRead = `
  if (feature < 32u) { return source[feature * 256u + (u32(row) / 2u) * 16u + u32(column) / 2u]; }
  return skip[(feature - 32u) * 1024u + u32(row) * 32u + u32(column)];`;
  return `${BINDINGS}
fn readFeature(feature: u32, row: i32, column: i32) -> f32 {
  ${featureRead}
}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let outputIndex = invocation.x;
  if (outputIndex >= ${layer.output * outputSize * outputSize}u) { return; }
  let channel = outputIndex / ${outputSize * outputSize}u;
  let row = i32((outputIndex / ${outputSize}u) % ${outputSize}u) * ${layer.stride};
  let column = i32(outputIndex % ${outputSize}u) * ${layer.stride};
  var total = weights[${layer.biasOffset}u + channel];
  for (var feature = 0u; feature < ${layer.input}u; feature += 1u) {
    for (var kernelRow = 0u; kernelRow < 3u; kernelRow += 1u) {
      let sourceRow = row + i32(kernelRow) - 1;
      if (sourceRow < 0 || sourceRow >= ${inputSize}) { continue; }
      for (var kernelColumn = 0u; kernelColumn < 3u; kernelColumn += 1u) {
        let sourceColumn = column + i32(kernelColumn) - 1;
        if (sourceColumn < 0 || sourceColumn >= ${inputSize}) { continue; }
        let weightIndex = ${layer.weightsOffset}u + ((channel * ${layer.input}u + feature) * 3u + kernelRow) * 3u + kernelColumn;
        total += weights[weightIndex] * readFeature(feature, sourceRow, sourceColumn);
      }
    }
  }
  destination[outputIndex] = ${index === 4 ? 'total' : 'max(total, 0.0)'};
}`;
}

function updateShader(model) {
  return `${BINDINGS}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let index = invocation.x;
  if (index >= 3072u) { return; }
  let schedule = ${model.scheduleOffset}u + settings[3] * 5u;
  let epsilon = skip[index];
  let clean = clamp((source[index] - weights[schedule + 1u] * epsilon) / weights[schedule], -1.0, 1.0);
  destination[index] = weights[schedule + 2u] * clean + weights[schedule + 3u] * epsilon;
}`;
}

function displayShader(model) {
  return `${BINDINGS}
@compute @workgroup_size(${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
  let pixel = invocation.x;
  let width = settings[0];
  let height = settings[1];
  if (pixel >= width * height) { return; }
  let column = pixel % width;
  let row = pixel / width;
  let nativePixel = (row * 32u / height) * 32u + column * 32u / width;
  for (var channel = 0u; channel < 3u; channel += 1u) {
    destination[pixel * 4u + channel] = (source[channel * 1024u + nativePixel] + 1.0) * 0.5;
  }
  destination[pixel * 4u + 3u] = 1.0;
  if (settings[2] != 0u) {
    let plateColumn = settings[8u + column];
    let plateRow = settings[8u + width + row];
    if (plateColumn < 7u && plateRow < 9u) {
      var colorOffset = ${model.brandOffset + 3}u;
      if (plateColumn >= 1u && plateColumn <= 5u && plateRow >= 1u && plateRow <= 7u) {
        let glyph = array<u32, 7>(14u, 17u, 17u, 31u, 17u, 17u, 17u);
        if (((glyph[plateRow - 1u] >> (5u - plateColumn)) & 1u) != 0u) { colorOffset = ${model.brandOffset}u; }
      }
      for (var channel = 0u; channel < 3u; channel += 1u) { destination[pixel * 4u + channel] = weights[colorOffset + channel]; }
    }
  }
}`;
}

function requireLimit(device, name, required) {
  if (!Number.isFinite(device.limits[name]) || device.limits[name] < required) throw new Error(`WebGPU ${name} must be at least ${required}`);
}

function bufferLimits(device, size, storage = true) {
  requireLimit(device, 'maxBufferSize', size);
  if (storage) requireLimit(device, 'maxStorageBufferBindingSize', size);
}

async function checkedGpu(device, operation) {
  const scopes = [];
  let result;
  let failure;
  try {
    for (const scope of ['out-of-memory', 'internal', 'validation']) {
      device.pushErrorScope(scope);
      scopes.push(scope);
    }
    result = await operation();
  } catch (error) {
    failure = error;
  }
  for (let index = scopes.length - 1; index >= 0; index -= 1) {
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

export async function createDiffusionGpuRenderer(artifact) {
  const model = snapshot(artifact);
  const gpu = globalThis.navigator?.gpu;
  if (!gpu) throw new Error('WebGPU is unavailable; no CPU fallback was used');
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter is available');
  if (adapter.info?.isFallbackAdapter === true || adapter.isFallbackAdapter === true) throw new Error('A fallback WebGPU adapter is not supported');
  const device = await adapter.requestDevice();
  const usage = globalThis.GPUBufferUsage;
  const mapMode = globalThis.GPUMapMode;
  const buffers = new Set();
  let closedError = null;
  let rendering = false;
  let layout;
  let pipelines;
  let weights;
  let samples;
  let hidden;
  let epsilon;
  let settingsBuffers;
  let convolutionGroups;
  let firstGroups;
  let updateGroups;
  let displayGroup;
  let output;
  let readback;
  let displayBytes = 0;

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

  function allocate(label, size, bufferUsage) {
    active();
    bufferLimits(device, size, Boolean(bufferUsage & usage.STORAGE));
    const buffer = device.createBuffer({ label, size, usage: bufferUsage });
    buffers.add(buffer);
    return buffer;
  }

  function bind(source, skip, settings, destination) {
    return device.createBindGroup({ layout, entries: [weights, source, skip, settings, destination].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }

  function releaseDisplay() {
    for (const buffer of [output, readback]) {
      if (buffer && buffers.delete(buffer)) buffer.destroy();
    }
    output = undefined;
    readback = undefined;
    displayGroup = undefined;
    displayBytes = 0;
  }

  try {
    device.addEventListener('uncapturederror', onError);
    device.lost.then(info => close(new Error(`WebGPU device lost: ${info.message || info.reason}`)), error => close(error));
    if (!usage || !mapMode || !globalThis.GPUShaderStage) throw new Error('WebGPU buffer/shader APIs are unavailable');
    requireLimit(device, 'maxComputeInvocationsPerWorkgroup', WORKGROUP_SIZE);
    requireLimit(device, 'maxComputeWorkgroupSizeX', WORKGROUP_SIZE);
    requireLimit(device, 'maxComputeWorkgroupsPerDimension', 256);
    requireLimit(device, 'maxStorageBuffersPerShaderStage', 5);
    requireLimit(device, 'maxBindGroups', 1);
    requireLimit(device, 'maxBindingsPerBindGroup', 5);
    await checkedGpu(device, async () => {
      layout = device.createBindGroupLayout({ entries: Array.from({ length: 5 }, (_, binding) => ({ binding, visibility: globalThis.GPUShaderStage.COMPUTE, buffer: { type: binding === 4 ? 'storage' : 'read-only-storage' } })) });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
      const shaders = [...LAYERS.map((_, index) => convolutionShader(model, index)), updateShader(model), displayShader(model)];
      pipelines = [];
      for (let index = 0; index < shaders.length; index += 1) {
        active();
        const label = `Diffusion ${index < 5 ? `conv${index}` : index === 5 ? 'DDIM' : 'display'}`;
        const module = device.createShaderModule({ label, code: shaders[index] });
        const compilation = await module.getCompilationInfo();
        const errors = compilation.messages.filter(message => message.type === 'error');
        if (errors.length) throw new Error(`Diffusion shader compilation failed: ${errors.map(error => `${error.lineNum}:${error.linePos} ${error.message}`).join('; ')}`);
        active();
        pipelines.push(await device.createComputePipelineAsync({ label, layout: pipelineLayout, compute: { module, entryPoint: 'main' } }));
      }
      weights = allocate('Diffusion weights and schedule', model.weights.byteLength, usage.STORAGE | usage.COPY_DST);
      device.queue.writeBuffer(weights, 0, model.weights);
      samples = [0, 1].map(index => allocate(`Diffusion sample ${index}`, 3 * AREA * 4, usage.STORAGE | usage.COPY_DST));
      hidden = [16 * AREA, 32 * 256, 32 * 256, 16 * AREA].map((length, index) => allocate(`Diffusion hidden ${index}`, length * 4, usage.STORAGE));
      epsilon = allocate('Diffusion epsilon', 3 * AREA * 4, usage.STORAGE);
      settingsBuffers = Array.from({ length: STEPS }, (_, step) => allocate(`Diffusion settings ${step}`, (8 + 2 * MAX_DISPLAY) * 4, usage.STORAGE | usage.COPY_DST));
      convolutionGroups = [bind(hidden[0], weights, settingsBuffers[0], hidden[1]), bind(hidden[1], weights, settingsBuffers[0], hidden[2]), bind(hidden[2], hidden[0], settingsBuffers[0], hidden[3]), bind(hidden[3], weights, settingsBuffers[0], epsilon)];
      firstGroups = settingsBuffers.map((settings, step) => bind(samples[step % 2], weights, settings, hidden[0]));
      updateGroups = settingsBuffers.map((settings, step) => bind(samples[step % 2], epsilon, settings, samples[(step + 1) % 2]));
    });
    active();
  } catch (error) {
    close(error);
    throw error;
  }

  return {
    async render(upl, width, height, options = {}) {
      active();
      const { composite, seed } = renderInputs(upl, width, height, options);
      if (rendering) throw new Error('A diffusion render is already in flight');
      const outputBytes = width * height * 16;
      bufferLimits(device, outputBytes);
      const workgroups = Math.ceil(width * height / WORKGROUP_SIZE);
      requireLimit(device, 'maxComputeWorkgroupsPerDimension', Math.max(256, workgroups));
      const settings = settingsFor(model, upl, width, height, composite);
      rendering = true;
      try {
        const result = await checkedGpu(device, async () => {
          const started = performance.now();
          if (displayBytes < outputBytes) {
            releaseDisplay();
            output = allocate('Diffusion display RGBA', outputBytes, usage.STORAGE | usage.COPY_SRC);
            readback = allocate('Diffusion readback', outputBytes, usage.COPY_DST | usage.MAP_READ);
            displayGroup = bind(samples[0], weights, settingsBuffers[0], output);
            displayBytes = outputBytes;
          }
          device.queue.writeBuffer(samples[0], 0, noise(seed));
          for (let step = 0; step < STEPS; step += 1) {
            settings[3] = step;
            device.queue.writeBuffer(settingsBuffers[step], 0, settings);
          }
          const encoder = device.createCommandEncoder({ label: 'Diffusion DDIM16 and display readback' });
          const pass = encoder.beginComputePass();
          for (let step = 0; step < STEPS; step += 1) {
            pass.setPipeline(pipelines[0]);
            pass.setBindGroup(0, firstGroups[step]);
            pass.dispatchWorkgroups(256);
            for (let index = 1; index < 5; index += 1) {
              pass.setPipeline(pipelines[index]);
              pass.setBindGroup(0, convolutionGroups[index - 1]);
              pass.dispatchWorkgroups(index === 1 || index === 2 ? 128 : index === 3 ? 256 : 48);
            }
            pass.setPipeline(pipelines[5]);
            pass.setBindGroup(0, updateGroups[step]);
            pass.dispatchWorkgroups(48);
          }
          pass.setPipeline(pipelines[6]);
          pass.setBindGroup(0, displayGroup);
          pass.dispatchWorkgroups(workgroups);
          pass.end();
          encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
          device.queue.submit([encoder.finish()]);
          await readback.mapAsync(mapMode.READ, 0, outputBytes);
          let pixels;
          try {
            active();
            pixels = new Float32Array(readback.getMappedRange(0, outputBytes).slice(0));
          } finally {
            readback.unmap();
          }
          if (!pixels.every(Number.isFinite)) throw new Error('Diffusion GPU produced nonfinite pixels');
          return { pixels, milliseconds: performance.now() - started, seed, nativeResolution: RESOLUTION, steps: STEPS };
        });
        active();
        return result;
      } catch (error) {
        releaseDisplay();
        throw error;
      } finally {
        rendering = false;
      }
    },
    dispose() {
      close(new Error('Diffusion GPU renderer is disposed'));
    },
  };
}
