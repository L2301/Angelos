import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseSource, printSource, validateGraph, graphHash, createReference, brandColor, assertDeterministicGraph } from '../src/core/graph.js';

const source = readFileSync(new URL('../examples/aurora.ang', import.meta.url), 'utf8');
const fixture = () => parseSource(source);

test('typed source round trips without changing computational identity', async () => {
  const graph = fixture();
  const rebuilt = parseSource(printSource(graph));
  assert.equal(await graphHash(graph), await graphHash(rebuilt));
  assert.equal(validateGraph(graph).ordered.at(-1).id, 'mark');
});

test('hash ignores JSON key and node ordering, changes with semantics', async () => {
  const graph = fixture();
  const reordered = { ...graph, nodes: [...graph.nodes].reverse() };
  assert.equal(await graphHash(graph), await graphHash(reordered));
  reordered.nodes.find(node => node.id === 'field').params.angle = 0.1;
  assert.notEqual(await graphHash(fixture()), await graphHash(reordered));
});

test('bad types, references, cycles, orphan nodes and parameters fail loudly', () => {
  const changes = [
    graph => { graph.nodes[3].inputs.condition = 'person'; },
    graph => { graph.nodes[3].inputs.condition = 'missing'; },
    graph => { graph.nodes[4].inputs.image = 'shade'; },
    graph => { graph.nodes.push({ id: 'unused', op: 'constant', inputs: {}, params: { value: 0.5 } }); },
    graph => { graph.nodes[3].params.extra = 1; },
    graph => { graph.nodes[1].params.index = 4; },
    graph => { graph.nodes[3].params.angle = NaN; },
    graph => { graph.nodes.at(-1).params.x = 0.99; },
    graph => { graph.encoder.dimension = 8; },
    graph => { graph.nodes[0].id = 'output'; },
  ];
  for (const mutate of changes) { const graph = fixture(); mutate(graph); assert.throws(() => validateGraph(graph)); }
});

test('parser rejects executable input, prototype names and ambiguous arguments', () => {
  for (const bad of [source + ' process.exit()', source.replace('upl()', 'upl(remote: "url")'), source.replace('index: 2', 'index: 2, index: 3'), source.replace('person =', '__proto__ ='), source.replace('0.55', 'Infinity')]) assert.throws(() => parseSource(bad));
});

test('reference graph responds to UPL and preserves the exact brand plate', () => {
  const graph = fixture();
  const render = createReference(graph);
  assert.notDeepEqual(render([0, 0, 0, 0], 0.8, 0.5), render([0, 0, 1, 1], 0.8, 0.5));
  const brand = graph.nodes.at(-1).params;
  const horizontal = brand.x + 2.5 * brand.cell;
  const vertical = brand.y + 1.5 * brand.cell;
  assert.deepEqual(render([0, 0, 0, 0], horizontal, vertical), brand.color);
  assert.deepEqual(brandColor(brand, brand.x + 0.5 * brand.cell, brand.y + 0.5 * brand.cell), brand.background);
  assert.equal(brandColor(brand, 1, 1), null);
  assert.throws(() => render([0, 1], 0, 0));
});

test('generative examples preserve typed source identity but reject browser execution', async () => {
  for (const name of ['aurora-generative', 'aurora-flux', 'aurora-ltx']) {
    const graph = parseSource(readFileSync(new URL(`../examples/${name}.ang`, import.meta.url), 'utf8'));
    assert.equal(graph.id, 'aurora');
    assert.equal(await graphHash(graph), await graphHash(parseSource(printSource(graph))));
    assert.throws(() => createReference(graph), /Python RasterTeacher.*No procedural fallback/);
    assert.throws(() => assertDeterministicGraph(graph), /Browser compiler/);
  }
});

test('generative contracts reject arbitrary models, bad prompts, parameters, and frame types', () => {
  const text = readFileSync(new URL('../examples/aurora-generative.ang', import.meta.url), 'utf8');
  for (const bad of [
    text.replace('"tiny-sd"', '"https://example.com/model"'),
    text.replace('"tiny-sd"', '"ltx-video"'),
    text.replace('steps: 12', 'steps: 0'),
    text.replace('guidance: 7.5', 'guidance: 21'),
    text.replace('width: 256', 'width: 32'),
    text.replace('height: 256', 'height: 257'),
    text.replace('prompt: scene', 'prompt: openness'),
    text.replace('threshold: 0.5', 'threshold: 2'),
    text.replace('A serene aurora', '{condition.__class__}'),
    text.replace('steps: 12', 'seed: 4, steps: 12'),
  ]) assert.throws(() => parseSource(bad));
  const interpolation = parseSource(text.replace('A serene aurora', 'Condition {condition}'));
  assert.equal(interpolation.nodes.find(node => node.op === 'prompt').params.low.startsWith('Condition {condition}'), true);
  const video = readFileSync(new URL('../examples/aurora-ltx.ang', import.meta.url), 'utf8');
  for (const bad of [video.replace('num_frames: 9', 'num_frames: 8'), video.replace('index: 4', 'index: 9'), video.replace('video: clip', 'video: scene')]) assert.throws(() => parseSource(bad));
});

test('text nodes are distinct and composable while output remains image-only', async () => {
  const text = readFileSync(new URL('../examples/aurora-text.ang', import.meta.url), 'utf8');
  const graph = parseSource(text);
  assert.equal(await graphHash(graph), await graphHash(parseSource(printSource(graph))));
  assert.throws(() => createReference(graph), /Python RasterTeacher/);
  const chained = text.replace('picture =', 'rewrite = generate_text(prompt: copy, model: "qwen3-8b", system: "Shorter {literal} copy 🌌", max_tokens: 8, temperature: 0)\npicture =').replace('generate_image(prompt: copy', 'generate_image(prompt: rewrite');
  assert.doesNotThrow(() => parseSource(chained));
  const video = text.replace('picture = generate_image(prompt: copy, model: "tiny-sd", steps: 12, guidance: 7.5, width: 256, height: 256)', 'clip = generate_video(prompt: copy, model: "ltx-video", steps: 12, guidance: 7.5, width: 256, height: 256, num_frames: 9)\npicture = frame(video: clip, index: 0)');
  assert.doesNotThrow(() => parseSource(video));
  for (const invalid of [
    text.replace('"qwen3-0.6b"', '"tiny-sd"'),
    text.replace('prompt: scene', 'prompt: openness'),
    text.replace('output picture', 'output copy'),
    text.replace('max_tokens: 48', 'max_tokens: 0'),
    text.replace('max_tokens: 48', 'max_tokens: 513'),
    text.replace('max_tokens: 48', 'max_tokens: 1.5'),
    text.replace('temperature: 0.7', 'temperature: 2.1'),
    text.replace('temperature: 0.7', 'temperature: -0.1'),
    text.replace(/system: "[^"]*"/, 'system: " "'),
    text.replace(/system: "[^"]*"/, `system: ${JSON.stringify('🌌'.repeat(1025))}`),
    text.replace(/system: "[^"]*"/, `system: ${JSON.stringify('\ud800')}`),
  ]) assert.throws(() => parseSource(invalid));
});

test('legacy deterministic prompt semantics retain pre-text graph hashes', async () => {
  const originals = {
    aurora: '960931297bd07971f08aaf99098a4ea157dae17be7ada26367698b67502c07ec',
    'aurora-generative': 'f0a99add82b7caab74c1ea45c5af801ed4cecfbae208c2ec7972e06b644987bd',
    'aurora-ltx': '24db0f58e99dd445b5779e5cbf13f8a517f6e574c70486c100108d3f26d46f0e',
  };
  for (const [name, expected] of Object.entries(originals)) {
    assert.equal(await graphHash(parseSource(readFileSync(new URL(`../examples/${name}.ang`, import.meta.url), 'utf8'))), expected);
  }
});
