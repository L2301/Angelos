import { test, expect } from '@playwright/test';
import { createHash, randomUUID } from 'node:crypto';
import { UPL_ENCODER_REVISION, uplHash, validateSemanticBundle } from '../../src/core/upl.js';
import { validateEncodedUpl } from '../../web/upl-lab.js';

function makeWireFixture(ingredients, { profileId, dimension = 512 }) {
  const current = structuredClone(ingredients);
  const bundle = { format: 'angelos.upl-input.v2', synthetic: true, adult: true,
    factsSchema: current.factsSchema, facts: current.facts,
    encoder: { id: 'qwen3-embedding-0.6b-v1', modelId: 'Qwen/Qwen3-Embedding-0.6B', revision: UPL_ENCODER_REVISION,
      dimension, pooling: 'last-token', normalization: 'l2', templateVersion: 'angelos.upl-text.v1' },
    upl: Array.from({ length: dimension }, (_, index) => (index % 2 ? -1 : 1) / Math.sqrt(dimension)),
    context: { campaignId: current.campaignId, factRevision: current.factRevision, eventRevision: current.eventRevision,
      consentVersion: current.consentVersion, asOf: current.asOf, ingredientsHash: uplHash(current) },
    recentIntent: current.events.map(({ kind, topic, occurredAt, expiresAt }) => ({ kind, topic, occurredAt, expiresAt })) };
  validateSemanticBundle(bundle, current);
  const snapshot = { format: 'angelos.snapshot.v2', id: randomUUID(), profileId, campaignId: current.campaignId,
    facts: bundle.facts, factRevision: current.factRevision, eventRevision: current.eventRevision,
    factSchemaVersion: current.factsSchema, encoder: bundle.encoder, upl: bundle.upl, vector: bundle.upl,
    consentVersion: current.consentVersion, consentSchemaVersion: 'angelos.consent.v1', createdAt: current.asOf, bundle };
  snapshot.hash = uplHash(snapshot);
  return { snapshot, bundle };
}

function wireIngredientsFixture() {
  const asOf = Date.now();
  const facts = { ageBand: '35-44', region: 'NA', openness: 0.5, energy: 0.5,
    interests: ['hiking'], goals: ['weekend walk'], preferredStyles: ['natural light'], preferredLanguage: 'en' };
  return { format: 'angelos.upl-ingredients.v2', synthetic: true, adult: true, factsSchema: 'angelos.facts.v2', facts,
    provenance: Object.keys(facts).sort().map(field => ({ field, source: 'synthetic-declared', occurredAt: asOf - 1000, receivedAt: asOf - 1000 })),
    eventsSchema: 'angelos.events.v1', events: [{ id: 'fixture-event', kind: 'save', topic: 'weekend hiking',
      occurredAt: asOf - 500, receivedAt: asOf - 500, expiresAt: asOf + 3600000, source: 'synthetic-operator', campaignId: 'aurora' }],
    factRevision: 8, eventRevision: 1, consentVersion: 1, campaignId: 'aurora', asOf };
}

test('wire contract: injected normalized fixtures satisfy actual server and browser contracts at all widths', async () => {
  const ingredients = wireIngredientsFixture();
  for (const dimension of [128, 256, 512, 1024]) {
    const response = makeWireFixture(ingredients, { profileId: 'wire-fixture-profile', dimension });
    expect(validateSemanticBundle(response.bundle, ingredients)).toEqual(response.bundle);
    await validateEncodedUpl(response, { ...ingredients, asOf: ingredients.asOf + 1000 },
      { profile: { id: 'wire-fixture-profile' }, campaignId: 'aurora' }, dimension);
    expect(response.bundle.encoder.revision).toBe(UPL_ENCODER_REVISION);
    expect(Object.keys(response.bundle).sort()).toEqual(['adult', 'context', 'encoder', 'facts', 'factsSchema', 'format', 'recentIntent', 'synthetic', 'upl']);
  }
});

test('wire contract: browser rejects altered pin, hashes, facts, intent and extra fields even with a recomputed snapshot hash', async () => {
  const ingredients = wireIngredientsFixture();
  const context = { profile: { id: 'wire-fixture-profile' }, campaignId: 'aurora' };
  const mutations = [
    response => { response.bundle.encoder.revision = 'a'.repeat(40); },
    response => { response.bundle.encoder.pooling = 'mean'; },
    response => { response.bundle.context.ingredientsHash = '0'.repeat(64); },
    response => { response.bundle.facts.interests = ['changed facts']; },
    response => { response.bundle.recentIntent[0].topic = 'changed intent'; },
    response => { response.bundle.recentIntent[0].kind = 'purchase'; },
    response => { response.bundle.fixtureOnly = true; },
    response => { response.bundle.upl[0] = Number.POSITIVE_INFINITY; },
    response => { response.bundle.context.extra = 'unexpected'; },
  ];
  for (const mutate of mutations) {
    const response = makeWireFixture(ingredients, { profileId: context.profile.id });
    mutate(response);
    response.snapshot.hash = uplHash(Object.fromEntries(Object.entries(response.snapshot).filter(([key]) => key !== 'hash')));
    await expect(validateEncodedUpl(response, ingredients, context, 512)).rejects.toThrow('current, bound v2 snapshot');
  }
  const response = makeWireFixture(ingredients, { profileId: context.profile.id });
  response.snapshot.hash = '0'.repeat(64);
  await expect(validateEncodedUpl(response, ingredients, context, 512)).rejects.toThrow('current, bound v2 snapshot');
});

async function installFixture(page, { unavailable = false, clipboard = false } = {}) {
  test.info().annotations.push({ type: 'fixture', description: 'Injected normalized coordinates; valid pinned wire metadata does not mean Qwen inference ran.' });
  const state = await (await page.request.get('/api/state')).json();
  const profiles = structuredClone(state.profiles);
  const facts = new Map(profiles.map(profile => [profile.id, { ...profile.facts,
    interests: ['fixture hiking'], goals: ['fixture weekend'], preferredStyles: ['fixture minimal'], preferredLanguage: 'en' }]));
  const fixture = { state, profiles, facts, events: [], calls: [], responses: [], eventRevision: 0, unavailable, pending: undefined };
  const observedAt = Date.now() - 1000;
  const ingredients = (profileId, campaignId) => ({ format: 'angelos.upl-ingredients.v2', synthetic: true, adult: true,
    factsSchema: 'angelos.facts.v2', facts: facts.get(profileId),
    provenance: Object.keys(facts.get(profileId)).sort().map(field => ({ field, source: 'synthetic-declared', occurredAt: observedAt, receivedAt: observedAt })),
    eventsSchema: 'angelos.events.v1', events: fixture.events, factRevision: profiles.find(profile => profile.id === profileId).factRevision,
    eventRevision: fixture.eventRevision, consentVersion: profiles.find(profile => profile.id === profileId).consentVersion,
    campaignId, asOf: Date.now() });
  fixture.response = payload => makeWireFixture(ingredients(payload.profileId, payload.campaignId), payload);
  await page.addInitScript(available => {
    window.fixtureClipboard = [];
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async text => {
      if (!available) throw new Error('Explicit browser fixture: clipboard unavailable');
      window.fixtureClipboard.push(text);
    } } });
  }, clipboard);
  await page.route('**/api/artifact', route => route.fulfill({ status: 404, json: { error: 'Explicit UI fixture: no student artifact' } }));
  await page.route('**/api/profiles', route => route.fulfill({ json: profiles }));
  await page.route('**/api/upl/**', async route => {
    const url = new URL(route.request().url());
    const payload = route.request().method() === 'POST' ? route.request().postDataJSON() : Object.fromEntries(url.searchParams);
    fixture.calls.push({ path: url.pathname, payload });
    if (url.pathname === '/api/upl/ingredients') return route.fulfill({ json: ingredients(payload.profileId, payload.campaignId) });
    if (url.pathname === '/api/upl/fact') {
      facts.get(payload.profileId)[payload.field] = payload.value;
      const profile = profiles.find(profile => profile.id === payload.profileId);
      profile.factRevision += 1;
      profile.facts = structuredClone(facts.get(profile.id));
      return route.fulfill({ json: { factRevision: profile.factRevision } });
    }
    if (url.pathname === '/api/upl/event') {
      fixture.eventRevision += 1;
      fixture.events.push({ id: `fixture-event-${fixture.eventRevision}`, kind: payload.kind, topic: payload.topic,
        occurredAt: Date.now(), receivedAt: Date.now(), expiresAt: Date.now() + 3600000,
        source: 'synthetic-operator', campaignId: payload.campaignId });
      return route.fulfill({ json: { eventRevision: fixture.eventRevision } });
    }
    if (url.pathname === '/api/upl/encode') {
      const response = fixture.response(payload);
      fixture.responses.push(response);
      if (fixture.pending) await fixture.pending;
      if (fixture.unavailable) return route.fulfill({ status: 503, json: { error: 'Pinned local Qwen3-Embedding weights are not cached (UI test fixture).' } });
      return route.fulfill({ json: response });
    }
    return route.fulfill({ status: 404, json: { error: 'No other endpoints belong to this explicit fixture.' } });
  });
  await page.goto('/lab');
  await expect(page.locator('#upl-v2-status')).toContainText('Ingredients ready');
  return fixture;
}

test('UPL UI fixture loads ingredients only, separates legacy copy, and stays narrow-screen accessible', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const fixture = await installFixture(page);
  await expect(page.locator('#upl-dimension')).toHaveValue('512');
  await expect(page.locator('#upl-copy-v2')).toBeDisabled();
  await expect(page.locator('#upl-encode')).toBeEnabled();
  await expect(page.locator('#copy-upl')).toHaveText('Copy legacy 4D UPL JSON');
  await expect(page.locator('#upl-interests')).toHaveValue('fixture hiking');
  await expect(page.locator('#upl-events')).toContainText('Missing observations are not negative feedback');
  expect(fixture.calls.every(call => call.path === '/api/upl/ingredients')).toBe(true);
  for (const width of [390, 760]) {
    await page.setViewportSize({ width, height: 900 });
    await page.locator('#upl-heading').scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  }
  await expect(page.getByLabel('Interests', { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test('mocked audience saves and intent events use separate explicit APIs without click labels or auto-encoding', async ({ page }) => {
  const fixture = await installFixture(page);
  await page.locator('#upl-interests').fill('hiking, night photography');
  await page.locator('#upl-goals').fill('find a trail');
  await page.locator('#upl-preferredStyles').fill('soft light, minimal');
  await page.locator('#upl-preferredLanguage').fill('fr');
  await expect(page.locator('#upl-encode')).toBeDisabled();
  await page.locator('#upl-save-facts').click();
  await expect(page.locator('#upl-v2-status')).toContainText('Ingredients ready');
  const saves = fixture.calls.filter(call => call.path === '/api/upl/fact');
  expect(saves.map(call => [call.payload.field, call.payload.value])).toEqual([
    ['interests', ['hiking', 'night photography']], ['goals', ['find a trail']],
    ['preferredStyles', ['soft light', 'minimal']], ['preferredLanguage', 'fr'],
  ]);
  await page.locator('#upl-event-kind').selectOption('dismiss');
  await page.locator('#upl-event-topic').fill('<img src=x onerror=alert(1)>');
  await page.locator('#upl-add-event').click();
  await expect(page.locator('#upl-event-revision')).toHaveText('1');
  await expect(page.locator('#upl-events')).toContainText('dismiss · <img src=x onerror=alert(1)>');
  expect(await page.locator('#upl-events img').count()).toBe(0);
  expect(fixture.calls.filter(call => call.path === '/api/upl/encode')).toHaveLength(0);
  const event = fixture.calls.find(call => call.path === '/api/upl/event').payload;
  expect(event.kind).toBe('dismiss');
  expect(Object.keys(event).sort()).toEqual(['campaignId', 'kind', 'profileId', 'topic']);
});

test('explicit fixture encode copies only public v2, reports dimensions/hash, then invalidates on edits', async ({ page }) => {
  const fixture = await installFixture(page, { clipboard: true });
  await page.locator('#upl-encode').click();
  await expect(page.locator('#upl-copy-v2')).toBeEnabled();
  await expect(page.locator('#upl-encoded-dimension')).toHaveText('512 semantic dimensions');
  await expect(page.locator('#upl-bundle-hash')).toHaveText(/^[0-9a-f]{64}$/);
  await page.locator('#upl-copy-v2').click();
  await expect(page.locator('#upl-v2-status')).toContainText('public v2 bundle copied');
  const copied = await page.evaluate(() => JSON.parse(window.fixtureClipboard[0]));
  const encodeCall = fixture.calls.find(call => call.path === '/api/upl/encode');
  expect(encodeCall.payload.dimension).toBe(512);
  expect(copied).toEqual(fixture.responses[0].bundle);
  expect(copied.snapshot).toBeUndefined();
  expect(copied.fixtureOnly).toBeUndefined();
  expect(copied.evidence).toBeUndefined();
  expect(copied.encoder.revision).toBe(UPL_ENCODER_REVISION);
  await page.locator('#upl-interests').fill('changed but unsaved');
  await expect(page.locator('#upl-copy-v2')).toBeDisabled();
  await expect(page.locator('#upl-bundle-hash')).toHaveText('No current bundle');
  await expect(page.locator('#upl-encode')).toBeDisabled();
});

test('503 never fabricates or retains an old embedding, and output widths are explicit', async ({ page }) => {
  const fixture = await installFixture(page);
  await page.locator('#upl-dimension').selectOption('128');
  await page.locator('#upl-encode').click();
  await expect(page.locator('#upl-copy-v2')).toBeEnabled();
  fixture.unavailable = true;
  await page.locator('#upl-dimension').selectOption('1024');
  await expect(page.locator('#upl-copy-v2')).toBeDisabled();
  await page.locator('#upl-encode').click();
  await expect(page.locator('#upl-v2-status')).toContainText('503');
  await expect(page.locator('#upl-v2-status')).toContainText('No fallback vector');
  await expect(page.locator('#upl-v2-json')).toHaveValue('');
  await expect(page.locator('#upl-encoded-dimension')).toHaveText('Not encoded');
  await expect(page.locator('#upl-copy-v2')).toBeDisabled();
  expect(fixture.calls.filter(call => call.path === '/api/upl/encode').map(call => call.payload.dimension)).toEqual([128, 1024]);
});

test('clipboard fallback contains only current fixture bundle and permission changes clear it', async ({ page }) => {
  const fixture = await installFixture(page);
  await page.locator('#upl-encode').click();
  await expect(page.locator('#upl-copy-v2')).toBeEnabled();
  await page.locator('#upl-copy-v2').click();
  await expect(page.locator('#upl-v2-fallback')).toBeVisible();
  expect(JSON.parse(await page.locator('#upl-v2-json').inputValue())).toEqual(fixture.responses[0].bundle);
  await page.getByRole('button', { name: 'People & permissions', exact: true }).click();
  await page.locator('#allow-personalization').uncheck();
  await page.getByRole('button', { name: 'Build an ad', exact: true }).click();
  await expect(page.locator('#upl-copy-v2')).toBeDisabled();
  await expect(page.locator('#upl-v2-fallback')).toBeHidden();
  await expect(page.locator('#upl-v2-json')).toHaveValue('');
  await expect(page.locator('#upl-interests')).toHaveValue('');
});

test('a late fixture encode cannot become current after a profile switch', async ({ page }) => {
  const fixture = await installFixture(page);
  let release;
  fixture.pending = new Promise(resolve => { release = resolve; });
  await page.locator('#upl-encode').click();
  await expect.poll(() => fixture.calls.filter(call => call.path === '/api/upl/encode').length).toBe(1);
  const previous = await page.locator('#profile').inputValue();
  const next = fixture.profiles.find(profile => profile.id !== previous);
  await page.locator('#profile').selectOption(next.id);
  await expect(page.locator('#upl-scope')).toContainText(next.name);
  release();
  await expect(page.locator('#upl-v2-status')).toContainText('Ingredients ready');
  await expect(page.locator('#upl-copy-v2')).toBeDisabled();
  await expect(page.locator('#upl-bundle-hash')).toHaveText('No current bundle');
});

test('copy rechecks server revisions instead of exporting a silently stale fixture bundle', async ({ page }) => {
  const fixture = await installFixture(page, { clipboard: true });
  await page.locator('#upl-encode').click();
  await expect(page.locator('#upl-copy-v2')).toBeEnabled();
  fixture.eventRevision += 1;
  await page.locator('#upl-copy-v2').click();
  await expect(page.locator('#upl-v2-status')).toContainText('changed');
  await expect(page.locator('#upl-copy-v2')).toBeDisabled();
  expect(await page.evaluate(() => window.fixtureClipboard)).toEqual([]);
});

test('a mismatched fixture snapshot cannot be copied as a current successful encode', async ({ page }) => {
  const fixture = await installFixture(page);
  const validResponse = fixture.response;
  fixture.response = payload => {
    const response = validResponse(payload);
    response.snapshot.factRevision += 1;
    return response;
  };
  await page.locator('#upl-encode').click();
  await expect(page.locator('#upl-v2-status')).toContainText('not a current, bound v2 snapshot');
  await expect(page.locator('#upl-copy-v2')).toBeDisabled();
});

test('numeric intent expiry clears a previously successful UI-fixture encode', async ({ page }) => {
  const fixture = await installFixture(page);
  fixture.eventRevision = 1;
  fixture.events.push({ id: 'expiring-fixture', kind: 'view', topic: 'fixture short-lived intent',
    occurredAt: Date.now(), receivedAt: Date.now(), expiresAt: Date.now() + 5000,
    source: 'synthetic-operator', campaignId: fixture.state.graph.id });
  await page.locator('#upl-refresh').click();
  await expect(page.locator('#upl-events')).toContainText('fixture short-lived intent');
  await page.locator('#upl-encode').click();
  await expect(page.locator('#upl-copy-v2')).toBeEnabled();
  await expect(page.locator('#upl-v2-status')).toContainText('Event freshness expired', { timeout: 10000 });
  await expect(page.locator('#upl-copy-v2')).toBeDisabled();
});

test('real local ingredient routes refresh richer facts and record intent separately from the click ledger', async ({ page }) => {
  await page.route('**/api/artifact', route => route.fulfill({ status: 404, json: { error: 'UI integration fixture: no student' } }));
  await page.route('**/api/upl/encode', route => route.fulfill({ status: 503,
    json: { error: 'Explicit encoder-unavailable test fixture; no model inference attempted.' } }));
  const before = await (await page.request.get('/api/metrics')).json();
  await page.goto('/lab');
  await expect(page.locator('#upl-v2-status')).toContainText('Ingredients ready');
  await page.locator('#upl-interests').fill('integration hiking, integration photography');
  await page.locator('#upl-save-facts').click();
  await expect(page.locator('#upl-v2-status')).toContainText('Ingredients ready');
  await expect(page.locator('#upl-interests')).toHaveValue('integration hiking, integration photography');
  await page.locator('#upl-event-kind').selectOption('click');
  await page.locator('#upl-event-topic').fill('explicit synthetic UI test');
  await page.locator('#upl-add-event').click();
  await expect(page.locator('#upl-events')).toContainText('synthetic-operator');
  await expect(page.locator('#upl-events')).toContainText('explicit synthetic UI test');
  const after = await (await page.request.get('/api/metrics')).json();
  expect(after.clicks).toBe(before.clicks);
  expect(after.impressions).toBe(before.impressions);
  await page.locator('#upl-encode').click();
  await expect(page.locator('#upl-v2-status')).toContainText('503');
  await expect(page.locator('#upl-copy-v2')).toBeDisabled();
});

test('static public UPL explanation makes no inference or privacy promises and makes no API calls', async ({ page }) => {
  const requests = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) requests.push(request.url()); });
  await page.goto('/');
  await expect(page.locator('#stage-0')).toContainText('Embeddings are not anonymous');
  await expect(page.locator('#stage-0')).toContainText('Trainable adapter infrastructure is not a trained conditioning model');
  await expect(page.locator('#stage-0')).toContainText('4D WebGPU path remains a legacy fixture');
  expect(requests).toEqual([]);
});

test('real cached CPU encoder completes a browser encode-to-clipboard round trip', async ({ page, context }, testInfo) => {
  test.skip(process.env.ANGELOS_TEST_REAL_UPL !== '1', 'Explicit opt-in required: real cached model inference, not a mocked UI fixture.');
  test.setTimeout(240000);
  expect(process.env.ANGELOS_UPL_DEVICE).toBe('cpu');
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: 'http://127.0.0.1:4311' });
  await page.goto('/lab');
  await expect(page.locator('#upl-v2-status')).toContainText('Ingredients ready', { timeout: 30000 });
  await expect(page.locator('#upl-dimension')).toHaveValue('512');
  const completed = page.waitForResponse(response => new URL(response.url()).pathname === '/api/upl/encode'
    && response.request().method() === 'POST', { timeout: 190000 });
  const started = Date.now();
  await page.locator('#upl-encode').click();
  await expect(page.locator('#upl-v2-status')).toContainText('first cold load');
  const response = await completed;
  const result = await response.json();
  expect(response.status(), JSON.stringify(result)).toBe(200);
  const elapsedMilliseconds = Date.now() - started;
  expect(result.bundle.format).toBe('angelos.upl-input.v2');
  expect(result.bundle.encoder.id).toBe('qwen3-embedding-0.6b-v1');
  expect(result.bundle.encoder.dimension).toBe(512);
  expect(result.bundle.upl).toHaveLength(512);
  expect(result.bundle.upl.every(Number.isFinite)).toBe(true);
  const negativeCoordinates = result.bundle.upl.filter(value => value < 0).length;
  const norm = Math.hypot(...result.bundle.upl);
  expect(negativeCoordinates).toBeGreaterThan(0);
  expect(norm).toBeCloseTo(1, 5);
  expect(result.snapshot.bundle).toEqual(result.bundle);
  await expect(page.locator('#upl-copy-v2')).toBeEnabled();
  await expect(page.locator('#upl-snapshot-hash')).toHaveText(result.snapshot.hash);
  await page.locator('#upl-copy-v2').click();
  await expect(page.locator('#upl-v2-status')).toContainText('public v2 bundle copied');
  const copied = await page.evaluate(() => navigator.clipboard.readText());
  expect(JSON.parse(copied)).toEqual(result.bundle);
  const copiedJsonSha256 = createHash('sha256').update(copied).digest('hex');
  await expect(page.locator('#upl-bundle-hash')).toHaveText(copiedJsonSha256);
  const report = { evidence: 'Real browser and cached CPU encoder; no mocked routes or model outputs.',
    elapsedMilliseconds, dimension: result.bundle.upl.length, negativeCoordinates, norm,
    encoder: result.bundle.encoder, snapshotHash: result.snapshot.hash, copiedJsonSha256,
    factRevision: result.snapshot.factRevision, eventRevision: result.snapshot.eventRevision,
    consentVersion: result.snapshot.consentVersion,
    limitation: 'One local worker round trip, including startup; not a GPU benchmark or personalization-quality result.' };
  console.log('Real UPL browser round trip:', JSON.stringify(report));
  await testInfo.attach('real-upl-browser-roundtrip.json', { body: JSON.stringify(report, null, 2), contentType: 'application/json' });
  expect(errors).toEqual([]);
});
