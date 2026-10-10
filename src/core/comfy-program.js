const MAX_BYTES = 262144;
const integer = (minimum, maximum) => value => Number.isSafeInteger(value) && value >= minimum && value <= maximum;
const number = (minimum, maximum) => value => typeof value === 'number' && Number.isFinite(value) && value >= minimum && value <= maximum;
const text = value => typeof value === 'string' && value.length <= 8192;
const choice = values => value => values.includes(value);
const size = value => integer(16, 4096)(value) && value % 8 === 0;

export const COMFY_NODES = Object.freeze({
  CheckpointLoaderSimple: { outputs: ['MODEL', 'CLIP', 'VAE'], inputs: { ckpt_name: value => text(value) && value.trim().length > 0 } },
  CLIPTextEncode: { outputs: ['CONDITIONING'], inputs: { text, clip: 'CLIP' } },
  EmptyLatentImage: { outputs: ['LATENT'], inputs: { width: size, height: size, batch_size: integer(1, 8) } },
  KSampler: { outputs: ['LATENT'], inputs: { model: 'MODEL', seed: integer(0, Number.MAX_SAFE_INTEGER), steps: integer(1, 100), cfg: number(0, 100), sampler_name: choice(['euler', 'euler_ancestral', 'heun', 'dpmpp_2m']), scheduler: choice(['normal', 'karras', 'exponential', 'simple']), positive: 'CONDITIONING', negative: 'CONDITIONING', latent_image: 'LATENT', denoise: number(0, 1) } },
  VAEDecode: { outputs: ['IMAGE'], inputs: { samples: 'LATENT', vae: 'VAE' } },
  ImageScale: { outputs: ['IMAGE'], inputs: { image: 'IMAGE', upscale_method: choice(['nearest-exact', 'bilinear', 'area', 'bicubic', 'lanczos']), width: integer(0, 4096), height: integer(0, 4096), crop: choice(['disabled', 'center']) } },
  SaveImage: { outputs: [], inputs: { images: 'IMAGE', filename_prefix: text } },
  PreviewImage: { outputs: [], inputs: { images: 'IMAGE' } },
});

function record(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function exactKeys(value, keys, location) {
  if (!record(value) || Object.keys(value).sort().join(',') !== [...keys].sort().join(',')) throw new Error(`${location}: expected exactly ${keys.join(', ')}`);
}

function tokensFor(source) {
  if (typeof source !== 'string' || new TextEncoder().encode(source).length > MAX_BYTES) throw new Error('Input must be text no larger than 256 KiB');
  const tokens = [];
  let offset = 0;
  while (offset < source.length) {
    const whitespace = source.slice(offset).match(/^\s+/);
    if (whitespace) { offset += whitespace[0].length; continue; }
    const token = source.slice(offset).match(/^(?:"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"|-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z0-9_]*|[{}\[\](),:=;])/);
    if (!token) throw new Error(`Unexpected character at ${offset + 1}`);
    tokens.push(token[0]);
    offset += token[0].length;
    if (tokens.length > 32768) throw new Error('Too many input tokens');
  }
  return tokens;
}

function reader(source) {
  const tokens = tokensFor(source);
  let cursor = 0;
  const peek = () => tokens[cursor];
  function take(expected) {
    const value = tokens[cursor++];
    if (value === undefined || (expected !== undefined && value !== expected)) throw new Error(`Expected ${expected ?? 'value'}, got ${value ?? 'end of input'}`);
    return value;
  }
  function literal(depth = 0) {
    if (depth > 16) throw new Error('Input nesting exceeds 16 levels');
    if (peek() === '{') {
      take('{');
      const value = Object.create(null);
      if (peek() !== '}') do {
        const keyToken = take();
        if (!keyToken.startsWith('"')) throw new Error('Expected a quoted JSON key');
        const key = JSON.parse(keyToken);
        if (Object.hasOwn(value, key)) throw new Error(`Duplicate key: ${key}`);
        take(':');
        value[key] = literal(depth + 1);
        if (peek() !== ',') break;
        take(',');
      } while (true);
      take('}');
      return value;
    }
    if (peek() === '[') {
      take('[');
      const value = [];
      if (peek() !== ']') do {
        value.push(literal(depth + 1));
        if (peek() !== ',') break;
        take(',');
      } while (true);
      take(']');
      return value;
    }
    const token = take();
    if (!/^(?:"|-?\d|true$|false$|null$)/.test(token)) throw new Error(`Not a JSON literal: ${token}`);
    const value = JSON.parse(token);
    if (typeof value === 'number' && (!Number.isFinite(value) || (Number.isInteger(value) && !Number.isSafeInteger(value)))) throw new Error('Number cannot be represented exactly; use a safe-integer seed');
    return value;
  }
  return { peek, take, literal };
}

export function validateComfyGraph(workflow) {
  if (!record(workflow) || Array.isArray(workflow.nodes)) throw new Error('Use ComfyUI API-format JSON, not the canvas/layout JSON. Export the workflow in API format.');
  const entries = Object.entries(workflow);
  if (entries.length < 1 || entries.length > 64) throw new Error('Accepts 1–64 nodes');
  const graph = Object.create(null);
  for (const [id, node] of entries) {
    if (!/^[1-9]\d{0,19}$/.test(id)) throw new Error(`Node ID ${id}: use positive numeric Comfy node IDs`);
    exactKeys(node, Object.hasOwn(node ?? {}, '_meta') ? ['class_type', 'inputs', '_meta'] : ['class_type', 'inputs'], `Node ${id}`);
    if (!Object.hasOwn(COMFY_NODES, node.class_type)) throw new Error(`Node ${id}: unsupported class ${node.class_type}. No nodes were converted or dropped.`);
    if (Object.hasOwn(node, '_meta')) {
      exactKeys(node._meta, ['title'], `Node ${id} metadata`);
      if (!text(node._meta.title)) throw new Error(`Node ${id}: invalid title`);
    }
    exactKeys(node.inputs, Object.keys(COMFY_NODES[node.class_type].inputs), `Node ${id} (${node.class_type}) inputs`);
    graph[id] = { class_type: node.class_type, inputs: Object.fromEntries(Object.entries(node.inputs)) };
  }
  for (const [id, node] of Object.entries(graph)) {
    for (const [name, rule] of Object.entries(COMFY_NODES[node.class_type].inputs)) {
      const value = node.inputs[name];
      if (typeof rule === 'function') {
        if (!rule(value)) throw new Error(`Node ${id}.${name}: unsupported literal or range`);
      } else {
        if (!Array.isArray(value) || value.length !== 2 || typeof value[0] !== 'string' || !integer(0, 2)(value[1]) || !Object.hasOwn(graph, value[0])) throw new Error(`Node ${id}.${name}: expected a connected ${rule} output`);
        const actual = COMFY_NODES[graph[value[0]].class_type].outputs[value[1]];
        if (actual !== rule) throw new Error(`Node ${id}.${name}: expected ${rule}, got ${actual ?? 'missing output'}`);
      }
    }
  }
  const ordered = [];
  const active = new Set();
  const visited = new Set();
  function visit(id) {
    if (active.has(id)) throw new Error(`Cycle at node ${id}`);
    if (visited.has(id)) return;
    active.add(id);
    for (const value of Object.values(graph[id].inputs)) if (Array.isArray(value)) visit(value[0]);
    active.delete(id);
    visited.add(id);
    ordered.push(id);
  }
  const outputs = Object.keys(graph).filter(id => COMFY_NODES[graph[id].class_type].outputs.length === 0);
  if (!outputs.length) throw new Error('A SaveImage or PreviewImage output is required');
  for (const id of outputs.sort()) visit(id);
  if (visited.size !== entries.length) throw new Error(`Disconnected nodes: ${Object.keys(graph).filter(id => !visited.has(id)).join(', ')}. Remove them explicitly; nothing is silently dropped.`);
  return { graph, ordered, outputs };
}

export function printComfyProgram(workflow) {
  const { graph, ordered } = validateComfyGraph(workflow);
  const lines = ['comfy_program v1'];
  for (const id of ordered) {
    const node = graph[id];
    const args = Object.keys(COMFY_NODES[node.class_type].inputs).map(name => {
      const value = node.inputs[name];
      return `${name}: ${Array.isArray(value) ? `node_${value[0]}[${value[1]}]` : JSON.stringify(value)}`;
    });
    lines.push(`node_${id} = ${node.class_type}(${args.join(', ')});`);
  }
  return lines.join('\n') + '\n';
}

export function importComfyWorkflow(source) {
  const input = reader(source);
  const workflow = input.literal();
  if (input.peek() !== undefined) throw new Error('Unexpected content after workflow JSON');
  return printComfyProgram(workflow);
}

export function compileComfyProgram(source) {
  const input = reader(source);
  input.take('comfy_program');
  input.take('v1');
  const workflow = Object.create(null);
  while (input.peek() !== undefined) {
    const name = input.take();
    if (!/^node_[1-9]\d{0,19}$/.test(name)) throw new Error('Expected node_NUMERIC_ID');
    const id = name.slice(5);
    if (Object.hasOwn(workflow, id)) throw new Error(`Duplicate node ${id}`);
    input.take('=');
    const classType = input.take();
    input.take('(');
    const inputs = Object.create(null);
    if (input.peek() !== ')') do {
      const port = input.take();
      if (Object.hasOwn(inputs, port)) throw new Error(`Duplicate input ${port}`);
      input.take(':');
      if (/^node_[1-9]\d{0,19}$/.test(input.peek())) {
        const reference = input.take().slice(5);
        input.take('[');
        const slot = input.literal();
        input.take(']');
        inputs[port] = [reference, slot];
      } else inputs[port] = input.literal();
      if (input.peek() !== ',') break;
      input.take(',');
    } while (true);
    input.take(')');
    input.take(';');
    workflow[id] = { class_type: classType, inputs };
    if (Object.keys(workflow).length > 64) throw new Error('Accepts 1–64 nodes');
  }
  return validateComfyGraph(workflow).graph;
}
