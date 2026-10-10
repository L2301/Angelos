import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { createLab } from '../src/server.js';
import { request as httpRequest } from 'node:http';

async function laboratory(context, options = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'angelos-test-'));
  const artifactDirectory = join(directory, 'artifacts');
  await mkdir(artifactDirectory);
  const { server } = await createLab({ dataDirectory: join(directory, 'data'), artifactDirectory, sourcePath: 'examples/aurora.ang', ...options });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(async () => {
    await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    await rm(directory, { recursive: true, force: true });
  });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const get = path => fetch(`${origin}${path}`);
  const post = (path, value, headers = {}) => fetch(`${origin}${path}`, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(value) });
  return { get, post, origin, artifactDirectory };
}

function syntheticTestArtifact(graphHash) {
  return { format: 'angelos.student.v1', id: 'test-student', campaignId: 'aurora', graphHash,
    encoder: { id: 'structured-demo-v1', dimension: 4 }, inputOrder: ['x', 'y', 'upl.0', 'upl.1', 'upl.2', 'upl.3'],
    layers: [{ input: 6, output: 1, activation: 'relu', weights: [0, 0, 0, 0, 0, 0], bias: [0] },
      { input: 1, output: 1, activation: 'relu', weights: [0], bias: [0] },
      { input: 1, output: 3, activation: 'sigmoid', weights: [0, 0, 0], bias: [0, 0, 0] }],
    brand: null, metrics: { baselineMse: 0.2, heldoutMse: 0.1, brandMaxError: null, brandTolerance: 0.06 }, training: { seed: 1, steps: 1, device: 'cpu', fixtureOnly: true } };
}

test('loopback server exposes only explicit static files and rejects cross-origin writes', async context => {
  const lab = await laboratory(context);
  assert.equal((await lab.get('/')).status, 200);
  assert.equal((await lab.get('/src/server.js')).status, 404);
  assert.equal((await lab.get('/data/angelos.sqlite')).status, 404);
  assert.equal((await lab.get('/../package.json')).status, 404);
  const refused = await lab.post('/api/snapshot', {}, { Origin: 'https://attacker.invalid' });
  assert.equal(refused.status, 403);
  const forgedHostStatus = await new Promise((resolve, reject) => {
    const request = httpRequest(`${lab.origin}/api/health`, { headers: { Host: 'attacker.invalid' } }, response => { response.resume(); resolve(response.statusCode); });
    request.on('error', reject);
    request.end();
  });
  assert.equal(forgedHostStatus, 403);
  assert.equal((await fetch(`${lab.origin}/api/snapshot`, { method: 'POST', body: '{}' })).status, 415);
  assert.equal((await lab.get('/api/health')).headers.get('cache-control'), 'no-store');
});

test('concept site and laboratory are distinct routes with explicit research links', async context => {
  const lab = await laboratory(context);
  const homepage = await (await lab.get('/')).text();
  const workspace = await (await lab.get('/lab')).text();
  assert.match(homepage, /story\.css/);
  assert.doesNotMatch(homepage, /src="\/web\/app.js"/);
  assert.match(workspace, /id="text-form"/);
  assert.match(workspace, /generate_text\(\)/);
  assert.equal((await lab.get('/lab/')).status, 200);
  assert.equal((await lab.get('/web/text-lab.js')).status, 200);
  assert.equal((await lab.get('/docs/MODEL-COVERAGE.md')).status, 200);
  assert.equal((await lab.get('/src/text-runner.js')).status, 404);
});

test('text preview validates its boundary, rejects remote origins and creates no profile or exposure records', async context => {
  const calls = [];
  const lab = await laboratory(context, { textRunner: async payload => { calls.push(payload); return { text: 'Injected HTTP fixture, not model output.', stored: false }; } });
  const before = await (await lab.get('/api/state')).json();
  const payload = { model: 'qwen3-0.6b', prompt: 'A fictional brand', system: 'Write copy.', max_tokens: 32, temperature: 0, seed: 4 };
  assert.equal((await lab.post('/api/text-preview', { ...payload, model: '../model' })).status, 400);
  assert.equal((await lab.post('/api/text-preview', { ...payload, max_tokens: 100000 })).status, 400);
  assert.equal((await lab.post('/api/text-preview', payload, { Origin: 'https://untrusted.invalid' })).status, 403);
  assert.equal(calls.length, 0);
  const response = await lab.post('/api/text-preview', payload);
  assert.equal(response.status, 200);
  assert.match((await response.json()).text, /Injected HTTP fixture/);
  assert.deepEqual(calls, [payload]);
  const after = await (await lab.get('/api/state')).json();
  assert.deepEqual(after.profiles, before.profiles);
  assert.deepEqual({ ...after.metrics, asOf: 0 }, { ...before.metrics, asOf: 0 });
  assert.equal(after.graphHash, before.graphHash);
});

test('invalid source does not replace the saved graph and student compatibility follows content', async context => {
  const lab = await laboratory(context);
  const before = await (await lab.get('/api/state')).json();
  assert.equal(before.profiles.length, 3);
  assert.equal((await lab.post('/api/graph', { source: 'process.exit()' })).status, 400);
  const after = await (await lab.get('/api/state')).json();
  assert.equal(after.graphHash, before.graphHash);
  await writeFile(join(lab.artifactDirectory, 'test.student.json'), JSON.stringify(syntheticTestArtifact(before.graphHash)));
  await writeFile(join(lab.artifactDirectory, 'preview.json'), JSON.stringify({ file: 'test.student.json' }));
  assert.equal((await (await lab.get('/api/state')).json()).artifactCompatible, true);
  const changed = await lab.post('/api/graph', { source: before.source.replace('angle: 0.55', 'angle: 0.2') });
  assert.equal(changed.status, 200);
  assert.equal((await changed.json()).artifactCompatible, false);
});

test('a corrupt preview cannot disable graph authoring or authorize an impression', async context => {
  const lab = await laboratory(context);
  await writeFile(join(lab.artifactDirectory, 'preview.json'), '{broken');
  const response = await lab.get('/api/state');
  assert.equal(response.status, 200);
  const state = await response.json();
  assert.equal(state.artifactAvailable, false);
  assert.equal(state.artifactCompatible, false);
  assert.match(state.artifactError, /unavailable/);
  const saved = await lab.post('/api/graph', { source: state.source });
  assert.equal(saved.status, 200);
  assert.equal((await saved.json()).artifactCompatible, false);
  assert.equal((await lab.get('/api/artifact')).status, 400);
});

test('HTTP exposure pins snapshots, rejects unqualified exposure and attributes clicks without a model update', async context => {
  const lab = await laboratory(context);
  const state = await (await lab.get('/api/state')).json();
  await writeFile(join(lab.artifactDirectory, 'test.student.json'), JSON.stringify(syntheticTestArtifact(state.graphHash)));
  await writeFile(join(lab.artifactDirectory, 'preview.json'), JSON.stringify({ file: 'test.student.json' }));
  const profile = state.profiles[0];
  const snapshot = await (await lab.post('/api/snapshot', { profileId: profile.id, campaignId: 'aurora' })).json();
  const exposure = { visibleFraction: 1, visibleMilliseconds: 1100, documentVisible: true };
  const payload = { profileId: profile.id, campaignId: 'aurora', snapshotId: snapshot.id, artifactId: 'test-student', graphHash: state.graphHash, eventId: randomUUID(), exposure, seed: 19 };
  assert.equal((await lab.post('/api/impressions', { ...payload, seed: -1 })).status, 400);
  assert.equal((await lab.post('/api/impressions', { ...payload, exposure: { ...exposure, visibleMilliseconds: 0 } })).status, 400);
  const accepted = await lab.post('/api/impressions', payload);
  assert.equal(accepted.status, 200, await accepted.clone().text());
  const impression = await accepted.json();
  assert.ok(impression.context.creativeId.endsWith(':19'));
  assert.equal((await lab.post('/api/impressions', payload)).status, 200);
  assert.equal((await lab.post('/api/impressions', { ...payload, seed: 20 })).status, 409);
  const click = { impressionId: impression.id, campaignId: 'aurora', eventId: randomUUID() };
  assert.equal((await lab.post('/api/click', click)).status, 200);
  assert.equal((await lab.post('/api/click', click)).status, 200);
  const metrics = await (await lab.get('/api/metrics')).json();
  assert.equal(metrics.impressions, 1);
  assert.equal(metrics.clicks, 1);
  assert.equal(metrics.trainingRows, 0);
  assert.equal(metrics.ctr, 1);
  assert.equal((await lab.post('/api/delete-profile', { profileId: profile.id })).status, 200);
  assert.equal((await (await lab.get('/api/metrics')).json()).impressions, 0);
});
