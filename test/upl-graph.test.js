import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { validateGraph, graphHash, canonicalJson, parseSource, printSource, createReference, assertDeterministicGraph } from '../src/core/graph.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const python = process.env.ANGELOS_TEST_PYTHON || `${root}/.venv/bin/python`;
const revision = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3';
const legacySource = readFileSync(new URL('../examples/aurora.ang', import.meta.url), 'utf8');

function semanticGraph(dimension = 128) {
  return {
    version: 'angelos.graph.v2', id: 'synthetic_campaign',
    encoder: { id: 'qwen3-embedding-0.6b-v1', dimension, modelId: 'Qwen/Qwen3-Embedding-0.6B',
      revision, pooling: 'last-token', normalization: 'l2', templateVersion: 'angelos.upl-text.v1' },
    nodes: [
      { id: 'person', op: 'upl', inputs: {}, params: {} },
      { id: 'coordinate', op: 'feature', inputs: { input: 'person' }, params: { index: dimension - 1 } },
      { id: 'image', op: 'gradient', inputs: { condition: 'coordinate' }, params: { low: [0, 0, 0], high: [1, 1, 1], angle: 0 } },
    ],
    output: 'image',
  };
}

test('v2 admits all pinned semantic dimensions with graph-local feature bounds', () => {
  for (const dimension of [128, 256, 512, 1024]) {
    const graph = semanticGraph(dimension);
    assert.equal(validateGraph(graph).ordered.at(-1).id, 'image');
    for (const index of [-1, dimension, 1.5, true]) {
      const invalid = structuredClone(graph);
      invalid.nodes[1].params.index = index;
      assert.throws(() => validateGraph(invalid), /index/);
    }
  }
});

test('v2 rejects partial, unpinned, or incompatible semantic metadata', () => {
  for (const [field, value] of [
    ['revision', 'a'.repeat(40)], ['dimension', 4], ['dimension', 129], ['dimension', '128'],
    ['modelId', 'other/model'], ['pooling', 'mean'], ['normalization', 'none'], ['templateVersion', 'future'],
  ]) {
    const graph = semanticGraph();
    graph.encoder[field] = value;
    assert.throws(() => validateGraph(graph), /encoder/);
  }
  for (const field of Object.keys(semanticGraph().encoder)) {
    const graph = semanticGraph();
    delete graph.encoder[field];
    assert.throws(() => validateGraph(graph), /encoder/);
  }
  const extra = semanticGraph();
  extra.encoder.remoteCode = true;
  assert.throws(() => validateGraph(extra), /encoder/);
});

test('v1 syntax, strict encoder boundary, and historical hash stay unchanged', async () => {
  const legacy = parseSource(legacySource);
  assert.equal(legacy.version, 'angelos.graph.v1');
  assert.equal(await graphHash(legacy), '960931297bd07971f08aaf99098a4ea157dae17be7ada26367698b67502c07ec');
  assert.equal(await graphHash(parseSource(printSource(legacy))), await graphHash(legacy));
  const wrongVersion = semanticGraph();
  wrongVersion.version = 'angelos.graph.v1';
  assert.throws(() => validateGraph(wrongVersion), /encoder/);
  const wrongEncoder = structuredClone(legacy);
  wrongEncoder.version = 'angelos.graph.v2';
  assert.throws(() => validateGraph(wrongEncoder), /encoder/);
  legacy.nodes.find(node => node.op === 'feature').params.index = 4;
  assert.throws(() => validateGraph(legacy), /index/);
});

test('v2 cannot silently decompile or execute through legacy backends', () => {
  const graph = semanticGraph();
  assert.throws(() => printSource(graph), /Textual source\/decompilation.*graph.v1 only/);
  assert.throws(() => createReference(graph), /cannot execute graph.v2/);
  assert.throws(() => assertDeterministicGraph(graph), /cannot execute graph.v2/);
});

test('v2 graph hashes are stable across ordering and bind semantic dimensions', async () => {
  const graph = semanticGraph();
  const before = canonicalJson(graph);
  const reordered = structuredClone(graph);
  reordered.nodes.reverse();
  assert.equal(await graphHash(graph), await graphHash(reordered));
  assert.notEqual(await graphHash(graph), await graphHash(semanticGraph(256)));
  assert.equal(canonicalJson(graph), before);
});

test('Python and browser-compatible JS agree on v1/v2 validity and canonical hashes', {
  skip: existsSync(python) ? false : 'Python unavailable; set ANGELOS_TEST_PYTHON to an interpreter path',
}, async () => {
  const valid = [parseSource(legacySource), ...[128, 256, 512, 1024].map(semanticGraph)];
  const badPin = semanticGraph();
  badPin.encoder.revision = '0'.repeat(40);
  const badIndex = semanticGraph();
  badIndex.nodes[1].params.index = 128;
  const badVersion = semanticGraph();
  badVersion.version = 'angelos.graph.v3';
  const graphs = [...valid, badPin, badIndex, badVersion];
  const program = [
    'import json, sys',
    'from angelos_ml.graph import graph_hash, validate_graph',
    'results = []',
    'for graph in json.load(sys.stdin):',
    '    try:',
    '        validate_graph(graph)',
    '        results.append({"valid": True, "hash": graph_hash(graph)})',
    '    except ValueError:',
    '        results.append({"valid": False})',
    'print(json.dumps(results))',
  ].join('\n');
  const result = spawnSync(python, ['-c', program], {
    cwd: root, input: JSON.stringify(graphs), encoding: 'utf8', env: { ...process.env, PYTHONPATH: `${root}/ml` },
  });
  assert.equal(result.status, 0, result.stderr);
  const expected = await Promise.all(graphs.map(async graph => {
    try { return { valid: true, hash: await graphHash(graph) }; }
    catch { return { valid: false }; }
  }));
  assert.deepEqual(JSON.parse(result.stdout), expected);
});

test('graph module stays browser-compatible without Node-only imports', () => {
  const source = readFileSync(new URL('../src/core/graph.js', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /(?:from\s*|import\s*\()\s*['"](?:node:|\.\/upl\.js)/);
});
