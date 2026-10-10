import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createLab } from '../src/server.js';
import { semanticFixture } from './upl-fixture.js';

async function laboratory(context, uplRunner) {
  const directory = await mkdtemp(join(tmpdir(), 'angelos-upl-'));
  const { server, store } = await createLab({ dataDirectory: join(directory, 'data'), artifactDirectory: join(directory, 'artifacts'), uplRunner });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  context.after(async () => {
    await new Promise(resolve => server.close(resolve));
    await rm(directory, { recursive: true, force: true });
  });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const get = path => fetch(origin + path);
  const post = (path, payload, headers = {}) => fetch(origin + path, { method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers }, body: JSON.stringify(payload) });
  const profile = store.listProfiles()[0];
  return { get, post, store, profile, input: { profileId: profile.id, campaignId: 'aurora' } };
}

test('UPL HTTP path updates rich facts and intent, freezes semantic result and retains legacy preview', async context => {
  const requests = [];
  const lab = await laboratory(context, async request => { requests.push(request); return semanticFixture(request.ingredients, request.dimension); });
  const before = lab.store.getProfile(lab.profile.id);
  assert.equal((await lab.post('/api/upl/fact', { ...lab.input, field: 'interests', value: ['climbing'] })).status, 200);
  assert.equal((await lab.post('/api/upl/event', { ...lab.input, kind: 'save', topic: 'alpine backpack' })).status, 200);
  const ingredients = await (await lab.get(`/api/upl/ingredients?${new URLSearchParams(lab.input)}`)).json();
  assert.deepEqual(ingredients.facts.interests, ['climbing']);
  assert.equal(ingredients.events.length, 1);
  assert.equal(ingredients.factRevision, before.factRevision + 1);
  const response = await lab.post('/api/upl/encode', { ...lab.input, dimension: 512 });
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.bundle.upl.length, 512);
  assert.equal(result.snapshot.hash.length, 64);
  assert.deepEqual(result.bundle, result.snapshot.bundle);
  assert.equal(requests.length, 1);
  const legacy = await (await lab.post('/api/snapshot', lab.input)).json();
  assert.equal(legacy.upl.length, 4);
  assert.equal('interests' in legacy.facts, false);
  assert.equal(lab.store.metrics({ campaignId: 'aurora' }).exposures, 0);
});

test('UPL HTTP gates dimensions, permissions, campaign and origin before model execution', async context => {
  const requests = [];
  const lab = await laboratory(context, async request => { requests.push(request); return semanticFixture(request.ingredients, request.dimension); });
  assert.equal((await lab.post('/api/upl/encode', { ...lab.input, dimension: 4 })).status, 400);
  assert.equal((await lab.post('/api/upl/encode', { ...lab.input, campaignId: 'other', dimension: 512 })).status, 400);
  assert.equal((await lab.post('/api/upl/encode', { ...lab.input, dimension: 512, model: 'arbitrary' })).status, 400);
  assert.equal((await lab.post('/api/upl/encode', { ...lab.input, dimension: 512 }, { Origin: 'https://foreign.invalid' })).status, 403);
  assert.equal((await lab.post('/api/upl/fact', { ...lab.input, field: 'interests', value: ['duplicate', 'duplicate'] })).status, 400);
  await lab.post('/api/consent', { ...lab.input, personalization: false, telemetry: false, training: false });
  assert.equal((await lab.post('/api/upl/encode', { ...lab.input, dimension: 512 })).status, 403);
  assert.equal((await lab.get(`/api/upl/ingredients?${new URLSearchParams(lab.input)}`)).status, 403);
  assert.equal(requests.length, 0);
});

test('UPL HTTP rejects state races and fails unavailable model without padding or fallback', async context => {
  let update;
  const lab = await laboratory(context, async ({ ingredients, dimension }) => {
    if (update) update();
    return semanticFixture(ingredients, dimension);
  });
  update = () => lab.store.recordActivity({ ...lab.input, eventId: 'during-encode', kind: 'click', topic: 'new intent' });
  assert.equal((await lab.post('/api/upl/encode', { ...lab.input, dimension: 512 })).status, 409);
  const missing = await laboratory(context, async () => { throw Object.assign(new Error('No cached embedding checkpoint; no download or fallback'), { status: 503 }); });
  const response = await missing.post('/api/upl/encode', { ...missing.input, dimension: 512 });
  assert.equal(response.status, 503);
  assert.match((await response.json()).error, /No cached/);
  assert.equal(missing.store.metrics({ campaignId: 'aurora' }).exposures, 0);
});

test('incomplete adult facts block encoding but can still be repaired with permission', async context => {
  const lab = await laboratory(context, async ({ ingredients, dimension }) => semanticFixture(ingredients, dimension));
  assert.equal((await lab.post('/api/upl/fact', { ...lab.input, field: 'ageBand', value: null })).status, 200);
  assert.equal((await lab.post('/api/upl/encode', { ...lab.input, dimension: 512 })).status, 400);
  assert.equal((await lab.post('/api/upl/fact', { ...lab.input, field: 'ageBand', value: '35-44' })).status, 200);
  assert.equal((await lab.post('/api/upl/encode', { ...lab.input, dimension: 512 })).status, 200);
  await lab.post('/api/consent', { ...lab.input, personalization: false, telemetry: false, training: false });
  assert.equal((await lab.post('/api/upl/fact', { ...lab.input, field: 'ageBand', value: '45-54' })).status, 403);
});
