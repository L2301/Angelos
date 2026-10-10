import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { importComfyWorkflow, compileComfyProgram, printComfyProgram } from '../src/core/comfy-program.js';

const fixture = JSON.parse(await readFile(new URL('../examples/aurora-comfy-api.json', import.meta.url), 'utf8'));
const plain = value => JSON.parse(JSON.stringify(value));

test('Comfy API graph translates to typed source and back without changing execution inputs', () => {
  const source = importComfyWorkflow(JSON.stringify(fixture));
  assert.match(source, /^comfy_program v1\n/);
  assert.match(source, /model: node_1\[0\]/);
  assert.match(source, /negative: node_3\[0\]/);
  assert.match(source, /width: 1920, height: 1088/);
  assert.deepEqual(plain(compileComfyProgram(source)), fixture);
  assert.equal(printComfyProgram(compileComfyProgram(source)), source);
});

test('deterministic image resize and multiple output nodes survive round-trip', () => {
  const graph = structuredClone(fixture);
  graph['8'] = { class_type: 'ImageScale', inputs: { image: ['6', 0], upscale_method: 'lanczos', width: 1920, height: 1080, crop: 'center' } };
  graph['9'] = { class_type: 'PreviewImage', inputs: { images: ['8', 0] } };
  assert.deepEqual(plain(compileComfyProgram(printComfyProgram(graph))), graph);
});

test('titles are explicitly non-executable metadata, while strings remain data', () => {
  const graph = structuredClone(fixture);
  graph['1']._meta = { title: 'A display label' };
  graph['2'].inputs.text = 'Quotes " and newline\n); process.exit(); <script>alert(1)</script>';
  const expected = structuredClone(graph);
  delete expected['1']._meta;
  assert.deepEqual(plain(compileComfyProgram(printComfyProgram(graph))), expected);
});

test('rejects unsupported nodes, settings, unknown keys, bad types and missing outputs', () => {
  const cases = [
    [graph => { graph['1'].class_type = 'UnreviewedLoader'; }, /unsupported class/],
    [graph => { graph['2'].inputs.clip = ['1', 0]; }, /expected CLIP, got MODEL/],
    [graph => { graph['2'].inputs.clip = ['missing', 1]; }, /connected CLIP/],
    [graph => { graph['2'].inputs.clip = ['1', 7]; }, /connected CLIP/],
    [graph => { graph['2'].inputs.clip = ['1', true]; }, /connected CLIP/],
    [graph => { graph['5'].inputs.seed = Number.MAX_SAFE_INTEGER + 1; }, /range/],
    [graph => { graph['5'].inputs.steps = true; }, /range/],
    [graph => { graph['5'].inputs.sampler_name = 'invented'; }, /range/],
    [graph => { graph['5'].inputs.scheduler = 'invented'; }, /range/],
    [graph => { graph['5'].inputs.denoise = NaN; }, /range/],
    [graph => { graph['4'].inputs.width = 1081; }, /range/],
    [graph => { graph['4'].inputs.batch_size = 9; }, /range/],
    [graph => { graph['2'].inputs.extra = 'ignored?'; }, /expected exactly/],
    [graph => { delete graph['5'].inputs.negative; }, /expected exactly/],
    [graph => { graph['1'].extra = 'ignored?'; }, /expected exactly/],
    [graph => { graph['8'] = structuredClone(graph['2']); }, /Disconnected/],
    [graph => { delete graph['7']; }, /output is required/],
  ];
  for (const [mutate, pattern] of cases) {
    const graph = structuredClone(fixture);
    mutate(graph);
    assert.throws(() => printComfyProgram(graph), pattern);
  }
});

test('rejects cycles without recursion overflow or silently choosing a branch', () => {
  const graph = structuredClone(fixture);
  graph['8'] = { class_type: 'ImageScale', inputs: { image: ['8', 0], upscale_method: 'bilinear', width: 1920, height: 1080, crop: 'center' } };
  graph['7'].inputs.images = ['8', 0];
  assert.throws(() => printComfyProgram(graph), /Cycle/);
});

test('strict bounded parsing rejects duplicate keys, unsafe integers, layout JSON and executable code', () => {
  for (const source of ['{"1":{},"1":{}}', '{"seed":9007199254740993}', '{"value":1e999}', '{"value":NaN}', '{"trailing":1,}', '{"nodes":[]}', 'process.exit()', ' '.repeat(262145), '['.repeat(18) + '0' + ']'.repeat(18)]) {
    assert.throws(() => importComfyWorkflow(source));
  }
  assert.throws(() => compileComfyProgram('comfy_program v1\nnode_1 = eval(code: "alert(1)");'), /unsupported class/);
  assert.throws(() => compileComfyProgram(printComfyProgram(fixture) + 'process.exit();'));
  assert.throws(() => compileComfyProgram(printComfyProgram(fixture).replace('seed: 2026', 'seed: 2026, seed: 3')), /Duplicate input/);
  assert.throws(() => compileComfyProgram(printComfyProgram(fixture) + 'node_1 = PreviewImage(images: node_6[0]);'), /Duplicate node/);
  assert.throws(() => compileComfyProgram('comfy_program v2'), /Expected v1/);
});

test('CLI performs offline round-trip and refuses to overwrite output', async context => {
  const directory = await mkdtemp(join(tmpdir(), 'angelos-comfy-program-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const program = join(directory, 'campaign.comfy.ang');
  const output = join(directory, 'roundtrip.json');
  const script = new URL('../scripts/comfy-program.js', import.meta.url).pathname;
  const input = new URL('../examples/aurora-comfy-api.json', import.meta.url).pathname;
  execFileSync(process.execPath, [script, 'import', input, '--output', program]);
  execFileSync(process.execPath, [script, 'compile', program, '--output', output]);
  assert.deepEqual(JSON.parse(await readFile(output, 'utf8')), fixture);
  assert.throws(() => execFileSync(process.execPath, [script, 'compile', program, '--output', output], { stdio: 'pipe' }), /EEXIST/);
});
