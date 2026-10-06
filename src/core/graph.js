export const GRAPH_VERSION = 'angelos.graph.v1';
export const ENCODER = Object.freeze({ id: 'structured-demo-v1', dimension: 4 });

const identifier = /^[A-Za-z_][A-Za-z0-9_]*$/;
const forbiddenNames = new Set(['__proto__', 'prototype', 'constructor', 'campaign', 'output']);
const finiteRange = (value, minimum, maximum) => typeof value === 'number' && Number.isFinite(value) && value >= minimum && value <= maximum;
const vector = (value, size) => Array.isArray(value) && value.length === size && value.every(item => finiteRange(item, 0, 1));
const rgb = value => vector(value, 3);
const unit = value => finiteRange(value, 0, 1);
const signedUnit = value => finiteRange(value, -1, 1);
const promptText = value => typeof value === 'string' && value.trim().length > 0 && value.length <= 2048 && !/[{}]/.test(value.replaceAll('{condition}', ''));
const rasterSize = value => Number.isInteger(value) && value >= 256 && value <= 512 && value % 32 === 0;
const generationParams = { steps: value => Number.isInteger(value) && value >= 1 && value <= 100, guidance: value => finiteRange(value, 0, 20), width: rasterSize, height: rasterSize };
export const GENERATIVE_OPERATIONS = Object.freeze(['prompt', 'generate_image', 'generate_video', 'frame']);

export const OPERATIONS = Object.freeze({
  upl: { output: 'upl', inputs: {}, params: {} },
  feature: { output: 'scalar', inputs: { input: 'upl' }, params: { index: value => Number.isInteger(value) && unit(value / 3) } },
  constant: { output: 'scalar', inputs: {}, params: { value: unit } },
  prompt: { output: 'prompt', inputs: { condition: 'scalar' }, params: { low: promptText, high: promptText, threshold: unit } },
  generate_image: { output: 'image', inputs: { prompt: 'prompt' }, params: { model: value => ['tiny-sd', 'flux2-klein-4b'].includes(value), ...generationParams } },
  generate_video: { output: 'video', inputs: { prompt: 'prompt' }, params: { model: value => value === 'ltx-video', ...generationParams, num_frames: value => Number.isInteger(value) && value >= 1 && value <= 121 && (value - 1) % 8 === 0 } },
  frame: { output: 'image', inputs: { video: 'video' }, params: { index: value => Number.isInteger(value) && value >= 0 && value <= 120 } },
  gradient: { output: 'image', inputs: { condition: 'scalar' }, params: { low: rgb, high: rgb, angle: signedUnit } },
  disk: { output: 'image', inputs: { image: 'image', condition: 'scalar' }, params: { color: rgb, center: value => vector(value, 2), radius: value => finiteRange(value, 0.02, 1), softness: value => finiteRange(value, 0.001, 1) } },
  exposure: { output: 'image', inputs: { image: 'image', condition: 'scalar' }, params: { amount: signedUnit } },
  vignette: { output: 'image', inputs: { image: 'image' }, params: { strength: unit } },
  brand: { output: 'image', inputs: { image: 'image' }, params: { x: unit, y: unit, cell: value => finiteRange(value, 0.005, 0.05), color: rgb, background: rgb } },
});

function record(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value) && [Object.prototype, null].includes(Object.getPrototypeOf(value));
}

function exactKeys(value, expected, location) {
  if (!record(value)) throw new Error(`${location} must be an object`);
  const actual = Object.keys(value).sort();
  if (JSON.stringify(actual) !== JSON.stringify([...expected].sort())) throw new Error(`${location} requires exactly: ${expected.join(', ') || '(no fields)'}`);
}

function validId(value) {
  return typeof value === 'string' && value.length <= 64 && identifier.test(value) && !forbiddenNames.has(value);
}

export function validateGraph(graph) {
  exactKeys(graph, ['version', 'id', 'encoder', 'nodes', 'output'], 'graph');
  if (graph.version !== GRAPH_VERSION) throw new Error('Unsupported graph version');
  if (!validId(graph.id)) throw new Error('Invalid campaign ID');
  exactKeys(graph.encoder, ['id', 'dimension'], 'encoder');
  if (graph.encoder.id !== ENCODER.id || graph.encoder.dimension !== ENCODER.dimension) throw new Error('Unsupported encoder contract');
  if (!Array.isArray(graph.nodes) || graph.nodes.length < 1 || graph.nodes.length > 64) throw new Error('Graph needs 1–64 nodes');
  const nodes = new Map();
  for (const node of graph.nodes) {
    exactKeys(node, ['id', 'op', 'inputs', 'params'], 'node');
    if (!validId(node.id) || nodes.has(node.id)) throw new Error(`Invalid or duplicate node ID: ${node.id}`);
    if (!Object.hasOwn(OPERATIONS, node.op)) throw new Error(`Unsupported operation: ${node.op}`);
    const specification = OPERATIONS[node.op];
    exactKeys(node.inputs, Object.keys(specification.inputs), `${node.id} inputs`);
    exactKeys(node.params, Object.keys(specification.params), `${node.id} params`);
    for (const [name, check] of Object.entries(specification.params)) {
      if (!check(node.params[name])) throw new Error(`${node.id}.${name}: invalid value or range`);
    }
    if (node.op === 'brand' && (node.params.x + node.params.cell * 7 > 1 || node.params.y + node.params.cell * 9 > 1)) throw new Error('Brand plate must fit within image');
    if (node.op === 'brand' && node.id !== graph.output) throw new Error('Brand must be the terminal output');
    nodes.set(node.id, node);
  }
  if (graph.nodes.filter(node => node.op === 'upl').length !== 1) throw new Error('Exactly one UPL source is required');
  if (!nodes.has(graph.output) || OPERATIONS[nodes.get(graph.output).op].output !== 'image') throw new Error('Output must reference an image');
  for (const node of graph.nodes) {
    for (const [port, expectedType] of Object.entries(OPERATIONS[node.op].inputs)) {
      const source = nodes.get(node.inputs[port]);
      if (!source) throw new Error(`${node.id}.${port}: unknown reference ${node.inputs[port]}`);
      if (OPERATIONS[source.op].output !== expectedType) throw new Error(`${node.id}.${port}: expected ${expectedType}, got ${OPERATIONS[source.op].output}`);
      if (node.op === 'frame' && node.params.index >= source.params.num_frames) throw new Error(`${node.id}.index exceeds video frame count`);
    }
  }
  const ordered = [];
  const visiting = new Set();
  const visited = new Set();
  function visit(nodeId) {
    if (visiting.has(nodeId)) throw new Error(`Cycle at ${nodeId}`);
    if (visited.has(nodeId)) return;
    visiting.add(nodeId);
    const node = nodes.get(nodeId);
    for (const source of [...new Set(Object.values(node.inputs))].sort()) visit(source);
    visiting.delete(nodeId);
    visited.add(nodeId);
    ordered.push(node);
  }
  for (const nodeId of [...nodes.keys()].sort()) visit(nodeId);
  const reachable = new Set();
  function reach(nodeId) {
    if (reachable.has(nodeId)) return;
    reachable.add(nodeId);
    for (const source of Object.values(nodes.get(nodeId).inputs)) reach(source);
  }
  reach(graph.output);
  if (reachable.size !== nodes.size) throw new Error(`Unreachable nodes: ${[...nodes.keys()].filter(nodeId => !reachable.has(nodeId)).join(', ')}`);
  return { graph, ordered };
}

export function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (record(value)) return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  if (typeof value === 'number' && !Number.isFinite(value)) throw new Error('Non-finite JSON number');
  const result = JSON.stringify(value);
  if (result === undefined) throw new Error('Unsupported JSON value');
  return result;
}

export async function graphHash(graph) {
  validateGraph(graph);
  const canonical = canonicalJson({ ...graph, nodes: [...graph.nodes].sort((left, right) => left.id < right.id ? -1 : left.id > right.id ? 1 : 0) });
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(canonical));
  return [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2, '0')).join('');
}

function tokenize(source) {
  if (typeof source !== 'string' || source.length > 100_000) throw new Error('Source must be text, at most 100KB');
  const tokens = [];
  let offset = 0;
  while (offset < source.length) {
    const remaining = source.slice(offset);
    const skipped = remaining.match(/^(?:\s+|\/\/[^\n]*)/);
    if (skipped) { offset += skipped[0].length; continue; }
    const match = remaining.match(/^(?:"(?:[^"\\]|\\.)*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z0-9_]*|[=(),:\[\]])/);
    if (!match) throw new Error(`Unexpected token at character ${offset + 1}`);
    tokens.push({ value: match[0], offset });
    offset += match[0].length;
    if (tokens.length > 12000) throw new Error('Too many source tokens');
  }
  return tokens;
}

export function parseSource(source) {
  const tokens = tokenize(source);
  let position = 0;
  const peek = () => tokens[position]?.value;
  function take(expected) {
    const token = tokens[position++];
    if (!token || (expected !== undefined && token.value !== expected)) throw new Error(`Expected ${expected ?? 'token'} at character ${(token?.offset ?? source.length) + 1}`);
    return token.value;
  }
  function name() {
    const value = take();
    if (!validId(value)) throw new Error(`Expected identifier, got ${value}`);
    return value;
  }
  function literal() {
    if (peek() === '[') {
      take('[');
      const values = [];
      while (peek() !== ']') {
        const value = take();
        const number = Number(value);
        if (!Number.isFinite(number)) throw new Error('Array elements must be finite numbers');
        values.push(number);
        if (values.length > 16) throw new Error('Array literal too long');
        if (peek() !== ']') take(',');
      }
      take(']');
      return values;
    }
    const value = take();
    if (value.startsWith('"')) return JSON.parse(value);
    const number = Number(value);
    if (!Number.isFinite(number)) throw new Error(`Expected literal, got ${value}`);
    return number;
  }
  take('campaign');
  const campaignId = name();
  const nodes = [];
  while (peek() && peek() !== 'output') {
    const nodeId = name();
    take('=');
    const operation = name();
    if (!Object.hasOwn(OPERATIONS, operation)) throw new Error(`Unsupported operation: ${operation}`);
    take('(');
    const inputs = {};
    const params = {};
    const seen = new Set();
    while (peek() !== ')') {
      const key = name();
      if (seen.has(key)) throw new Error(`Duplicate argument ${key}`);
      seen.add(key);
      take(':');
      if (Object.hasOwn(OPERATIONS[operation].inputs, key)) inputs[key] = name();
      else if (Object.hasOwn(OPERATIONS[operation].params, key)) params[key] = literal();
      else throw new Error(`Unknown argument ${operation}.${key}`);
      if (peek() !== ')') take(',');
    }
    take(')');
    nodes.push({ id: nodeId, op: operation, inputs, params });
    if (nodes.length > 64) throw new Error('Too many nodes');
  }
  take('output');
  const output = name();
  if (position !== tokens.length) throw new Error('Unexpected content after output');
  const graph = { version: GRAPH_VERSION, id: campaignId, encoder: { ...ENCODER }, nodes, output };
  validateGraph(graph);
  return graph;
}

export function printSource(graph) {
  const { ordered } = validateGraph(graph);
  const statements = ordered.map(node => {
    const args = [...Object.entries(node.inputs).map(([name, reference]) => `${name}: ${reference}`), ...Object.entries(node.params).map(([name, value]) => `${name}: ${JSON.stringify(value)}`)];
    return `${node.id} = ${node.op}(${args.join(', ')})`;
  });
  return [`campaign ${graph.id}`, ...statements, `output ${graph.output}`, ''].join('\n');
}

const clamp = value => Math.max(0, Math.min(1, value));
const mix = (left, right, ratio) => left.map((value, channel) => value * (1 - ratio) + right[channel] * ratio);

export function brandColor(brand, horizontal, vertical) {
  const column = Math.floor((horizontal - brand.x) / brand.cell);
  const row = Math.floor((vertical - brand.y) / brand.cell);
  if (column < 0 || row < 0 || column >= 7 || row >= 9) return null;
  const rows = [14, 17, 17, 31, 17, 17, 17];
  const ink = row >= 1 && row <= 7 && column >= 1 && column <= 5 && ((rows[row - 1] >> (5 - column)) & 1) === 1;
  return ink ? brand.color : brand.background;
}

export function createReference(graph) {
  const { ordered } = validateGraph(graph);
  assertDeterministicGraph(graph, 'Browser reference');
  return (upl, horizontal, vertical) => {
    if (!vector(upl, 4) || !unit(horizontal) || !unit(vertical)) throw new Error('Invalid reference inputs');
    const values = new Map();
    for (const node of ordered) {
      const params = node.params;
      const input = port => values.get(node.inputs[port]);
      let value;
      switch (node.op) {
        case 'upl': value = upl; break;
        case 'feature': value = input('input')[params.index]; break;
        case 'constant': value = params.value; break;
        case 'gradient': value = mix(params.low, params.high, clamp(0.65 * horizontal + params.angle * (vertical - 0.5) + 0.35 * input('condition'))); break;
        case 'disk': {
          const center = params.center[0] + 0.12 * (input('condition') - 0.5);
          const distance = Math.hypot(horizontal - center, vertical - params.center[1]);
          value = mix(input('image'), params.color, clamp((params.radius - distance) / params.softness + 0.5));
          break;
        }
        case 'exposure': value = input('image').map(channel => clamp(channel * (1 + params.amount * (2 * input('condition') - 1)))); break;
        case 'vignette': value = input('image').map(channel => channel * (1 - params.strength * clamp(2 * ((horizontal - 0.5) ** 2 + (vertical - 0.5) ** 2)))); break;
        case 'brand': value = brandColor(params, horizontal, vertical) ?? input('image'); break;
        default: throw new Error(`No reference implementation for ${node.op}`);
      }
      values.set(node.id, value);
    }
    return values.get(graph.output);
  };
}

export function assertDeterministicGraph(graph, backend = 'Browser compiler') {
  const unsupported = graph.nodes.find(node => GENERATIVE_OPERATIONS.includes(node.op));
  if (unsupported) throw new Error(`${backend} cannot execute ${unsupported.op}; use the Python RasterTeacher / angelos_ml.dataset with explicitly downloaded, pinned local models. No procedural fallback is available.`);
}
