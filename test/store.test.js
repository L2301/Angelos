import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import test from 'node:test';
import { Store, ENCODER, encodeFacts } from '../src/core/store.js';

const BASE_FACTS = Object.freeze({ ageBand: '35-44', region: 'EU', openness: 0.75, energy: 0.25 });
const BASE_CONSENT = Object.freeze({ personalization: true, telemetry: true, training: true,
  campaignIds: Object.freeze(['aurora', 'borealis']) });

function setup(context, options = {}) {
  let now = 1_000_000;
  const store = new Store(':memory:', { clock: () => now, clickWindowMs: 100, allowedLatenessMs: 50, ...options });
  context.after(() => store.close());
  const profile = store.createProfile({ name: 'Synthetic adult', facts: BASE_FACTS, consent: BASE_CONSENT });
  return { store, profile, advance: duration => { now += duration; }, now: () => now };
}

function descriptor(campaignId = 'aurora') {
  return { id: `${campaignId}-student-v1`, campaignId, hash: 'a'.repeat(64), graphHash: 'b'.repeat(64), encoder: ENCODER };
}

function impression(store, profileId, eventId = 'exposure-1', campaignId = 'aurora') {
  return { eventId, profileId, campaignId, snapshotId: store.snapshot(profileId, campaignId).id,
    artifact: descriptor(campaignId), context: { creativeId: 'creative-v1', placementId: 'laboratory',
      actionSetId: 'actions-v1', policyVersion: 'uniform-v1', propensityVersion: 'discrete-v1', propensity: 0.5 },
    exposure: { visibleFraction: 0.5, visibleMilliseconds: 1000, documentVisible: true } };
}

function changeConsent(store, profileId, eventId, changes) {
  const profile = store.getProfile(profileId);
  return store.setConsent({ profileId, eventId, expectedVersion: profile.consentVersion, ...profile.consent, ...changes });
}

function review(store, profileId, eventId = 'review-1', source = 'operator') {
  return { eventId, profileId, campaignId: 'aurora', snapshotId: store.snapshot(profileId, 'aurora').id,
    artifact: descriptor(), creativeId: 'creative-v1', qualityRating: 4, brandPass: true, source };
}

function throwsCode(action, code) {
  assert.throws(action, error => error.code === code);
}

test('profiles retain raw facts and encoder rejects missing required facts', context => {
  const { store, profile } = setup(context);
  assert.equal(profile.synthetic, true);
  assert.equal(Object.hasOwn(profile, 'upl'), false);
  assert.deepEqual(profile.facts, BASE_FACTS);
  assert.deepEqual(encodeFacts(profile.facts), [0.4, 1 / 3, 0.75, 0.25]);
  assert.deepEqual(encodeFacts({ ageBand: '65+', region: 'OTHER', openness: 0, energy: 1 }), [1, 1, 0, 1]);
  const partial = store.createProfile({ name: 'Incomplete fixture', facts: { ageBand: '18-24' }, consent: BASE_CONSENT });
  throwsCode(() => store.snapshot(partial.id, 'aurora'), 'MISSING_FACTS');
  throwsCode(() => encodeFacts({ ...BASE_FACTS, energy: null }), 'MISSING_FACTS');
  assert.equal(store.listProfiles().length, 2);
  profile.facts.energy = 1;
  assert.equal(store.getProfile(profile.id).facts.energy, 0.25);
});

test('pure allowlists reject unknown keys, invalid scalars, getters and prototype payloads', context => {
  const { store, profile } = setup(context);
  for (const badFacts of [
    { ...BASE_FACTS, ageBand: 'under-18' }, { ...BASE_FACTS, region: 'US' },
    { ...BASE_FACTS, energy: '0.5' }, { ...BASE_FACTS, energy: NaN },
    { ...BASE_FACTS, openness: Infinity }, { ...BASE_FACTS, energy: -0.1 },
    { ...BASE_FACTS, personality: {} }, JSON.parse('{"__proto__": {"polluted":true}}'),
  ]) {
    throwsCode(() => store.createProfile({ name: 'Invalid', facts: badFacts }), 'INVALID_INPUT');
  }
  throwsCode(() => store.createProfile({ name: 'Invalid', synthetic: false }), 'INVALID_INPUT');
  throwsCode(() => store.createProfile({ name: 'Invalid', consent: { ...BASE_CONSENT, training: 'true' } }), 'INVALID_INPUT');
  throwsCode(() => store.createProfile({ name: 'Invalid', consent: { ...BASE_CONSENT, campaignIds: ['aurora', 'aurora'] } }), 'INVALID_INPUT');
  const getter = Object.defineProperty({}, 'name', { get() { throw new Error('Getter executed'); } });
  throwsCode(() => store.createProfile(getter), 'INVALID_INPUT');
  const invalid = impression(store, profile.id);
  invalid.context.extra = 'forbidden';
  throwsCode(() => store.recordImpression(invalid), 'INVALID_INPUT');
  assert.equal(store.listProfiles().length, 1);
});

test('fact events are idempotent, revision checked and cannot backdate newer values', context => {
  const { store, profile, advance, now } = setup(context);
  advance(10);
  const input = { profileId: profile.id, eventId: 'fact-1', expectedRevision: profile.factRevision,
    field: 'energy', value: 0.9, occurredAt: now() };
  const result = store.updateFact(input);
  advance(5);
  assert.deepEqual(store.updateFact(input), result);
  assert.equal(store.getProfile(profile.id).factRevision, profile.factRevision + 1);
  throwsCode(() => store.updateFact({ ...input, value: 0.8 }), 'CONFLICT');
  throwsCode(() => store.updateFact({ ...input, eventId: 'fact-stale' }), 'CONFLICT');
  throwsCode(() => store.updateFact({ ...input, eventId: 'fact-late', expectedRevision: result.factRevision,
    occurredAt: now() - 6 }), 'CONFLICT');
  throwsCode(() => store.updateFact({ ...input, eventId: 'fact-future', expectedRevision: result.factRevision,
    occurredAt: now() + 1 }), 'INVALID_INPUT');
  store.updateFact({ ...input, eventId: 'fact-unset', expectedRevision: result.factRevision, value: null });
  throwsCode(() => store.snapshot(profile.id, 'aurora'), 'MISSING_FACTS');
});

test('snapshots are immutable, independently versioned and stale snapshots cannot create new exposures', context => {
  const { store, profile, advance } = setup(context);
  const input = impression(store, profile.id);
  const exposed = store.recordImpression(input);
  const original = exposed.snapshot;
  const frozen = store.snapshot(profile.id, 'aurora');
  assert.equal(Object.isFrozen(frozen.facts), true);
  assert.throws(() => { frozen.upl[0] = 1; }, TypeError);
  advance(1);
  store.updateFact({ profileId: profile.id, eventId: 'fact-change', expectedRevision: profile.factRevision,
    field: 'energy', value: 1 });
  throwsCode(() => store.recordImpression({ ...input, eventId: 'stale-exposure' }), 'STALE_SNAPSHOT');
  const latest = store.snapshot(profile.id, 'aurora');
  assert.equal(latest.upl[3], 1);
  assert.equal(original.upl[3], 0.25);
  assert.equal(latest.encoder.id, original.encoder.id);
  assert.notEqual(latest.factSchemaVersion, latest.encoder.id);
  assert.notEqual(latest.hash, original.hash);
  store.recordClick({ eventId: 'click-after-edit', campaignId: 'aurora', impressionId: exposed.id });
  advance(150);
  assert.deepEqual(store.maturedRows({ campaignId: 'aurora' })[0].upl, original.upl);
});

test('default consent denies serving and telemetry remains independent of personalization', context => {
  const { store } = setup(context);
  const profile = store.createProfile({ name: 'Default permission', facts: BASE_FACTS });
  throwsCode(() => store.snapshot(profile.id, 'aurora'), 'CONSENT_DENIED');
  changeConsent(store, profile.id, 'grant-personalization', { personalization: true, campaignIds: ['aurora'] });
  const input = impression(store, profile.id);
  throwsCode(() => store.recordImpression(input), 'CONSENT_DENIED');
  throwsCode(() => store.snapshot(profile.id, 'borealis'), 'CONSENT_DENIED');
  changeConsent(store, profile.id, 'grant-telemetry', { telemetry: true });
  throwsCode(() => store.recordImpression(input), 'STALE_SNAPSHOT');
  assert.equal(store.recordImpression(impression(store, profile.id)).trainingEligibleAtExposure, false);
});

test('consent updates deduplicate, conflict on revision and invalidate old snapshots after regrant', context => {
  const { store, profile } = setup(context);
  const pending = impression(store, profile.id);
  const input = { profileId: profile.id, eventId: 'revoke', expectedVersion: profile.consentVersion,
    ...profile.consent, personalization: false };
  const revoked = store.setConsent(input);
  assert.deepEqual(store.setConsent(input), revoked);
  throwsCode(() => store.setConsent({ ...input, training: false }), 'CONFLICT');
  throwsCode(() => store.setConsent({ ...input, eventId: 'stale-consent' }), 'CONFLICT');
  throwsCode(() => store.recordImpression(pending), 'CONSENT_DENIED');
  changeConsent(store, profile.id, 'regrant', { personalization: true });
  throwsCode(() => store.recordImpression(pending), 'STALE_SNAPSHOT');
  store.recordImpression(impression(store, profile.id));
});

test('campaign and profile attribution cannot be crossed', context => {
  const { store, profile } = setup(context);
  const input = impression(store, profile.id);
  throwsCode(() => store.recordImpression({ ...input, campaignId: 'borealis', artifact: descriptor('borealis') }), 'INVALID_ATTRIBUTION');
  const other = store.createProfile({ name: 'Other fixture', facts: BASE_FACTS, consent: BASE_CONSENT });
  throwsCode(() => store.recordImpression({ ...input, profileId: other.id }), 'INVALID_ATTRIBUTION');
  throwsCode(() => store.recordImpression({ ...input, artifact: descriptor('borealis') }), 'INCOMPATIBLE_ARTIFACT');
  store.recordImpression(input);
  throwsCode(() => store.recordClick({ eventId: 'wrong-campaign', campaignId: 'borealis', impressionId: input.eventId }), 'INVALID_ATTRIBUTION');
  assert.equal(store.metrics({ campaignId: 'borealis' }).exposures, 0);
});

test('artifact, encoder and policy identities are pinned and conflicting artifact reuse is rejected', context => {
  const { store, profile } = setup(context);
  const input = impression(store, profile.id);
  const wrongEncoder = structuredClone(input);
  wrongEncoder.artifact.encoder.id = 'future-encoder';
  throwsCode(() => store.recordImpression(wrongEncoder), 'INCOMPATIBLE_ARTIFACT');
  const first = store.recordImpression(input);
  assert.deepEqual(first.artifact, descriptor());
  assert.equal(first.modelVersion, descriptor().id);
  assert.equal(first.context.propensityVersion, 'discrete-v1');
  const altered = structuredClone(input);
  altered.eventId = 'altered-artifact';
  altered.artifact.hash = 'c'.repeat(64);
  throwsCode(() => store.recordImpression(altered), 'INCOMPATIBLE_ARTIFACT');
  assert.equal(store.metrics({ campaignId: 'aurora' }).exposures, 1);
});

test('impression and click retries deduplicate and multiple clicks remain one positive', context => {
  const { store, profile, advance } = setup(context);
  const input = impression(store, profile.id);
  const exposed = store.recordImpression(input);
  advance(10);
  assert.deepEqual(store.recordImpression(input), exposed);
  const changed = structuredClone(input);
  changed.context.propensity = 0.25;
  throwsCode(() => store.recordImpression(changed), 'CONFLICT');
  const click = { eventId: 'click-1', campaignId: 'aurora', impressionId: exposed.id };
  const receipt = store.recordClick(click);
  assert.deepEqual(store.recordClick(click), receipt);
  throwsCode(() => store.recordClick({ ...click, occurredAt: receipt.occurredAt }), 'CONFLICT');
  store.recordClick({ ...click, eventId: 'click-2' });
  assert.deepEqual(receipt.artifact, exposed.artifact);
  assert.deepEqual(receipt.context, exposed.context);
  assert.equal(receipt.snapshotId, exposed.snapshotId);
  assert.equal(store.maturedRows({ campaignId: 'aurora' }).length, 0);
  advance(140);
  const rows = store.maturedRows({ campaignId: 'aurora' });
  assert.equal(rows.length, 1);
  assert.equal(rows[0].label, 1);
  const metrics = store.metrics({ campaignId: 'aurora' });
  assert.equal(metrics.clickEvents, 2);
  assert.equal(metrics.clickedExposures, 1);
  assert.equal(metrics.maturedCtr, 1);
});

test('only true exposures become mature negatives; generated snapshots and failed exposures do not', context => {
  const { store, profile, advance } = setup(context);
  store.snapshot(profile.id, 'aurora');
  const unexposed = impression(store, profile.id, 'unexposed');
  for (const exposure of [
    { visibleFraction: 0.49, visibleMilliseconds: 1000, documentVisible: true },
    { visibleFraction: 1, visibleMilliseconds: 999, documentVisible: true },
    { visibleFraction: 1, visibleMilliseconds: 1000, documentVisible: false },
  ]) throwsCode(() => store.recordImpression({ ...unexposed, exposure }), 'NOT_EXPOSED');
  assert.equal(store.metrics({ campaignId: 'aurora' }).ctr, null);
  store.recordImpression(impression(store, profile.id, 'exposed'));
  advance(149);
  assert.deepEqual(store.maturedRows({ campaignId: 'aurora' }), []);
  advance(1);
  const rows = store.maturedRows({ campaignId: 'aurora' });
  assert.deepEqual(rows.map(row => [row.impressionId, row.label]), [['exposed', 0]]);
  assert.equal(store.metrics({ campaignId: 'aurora' }).maturedCtr, 0);
});

test('click attribution uses half-open event and receipt windows', context => {
  const { store, profile, advance, now } = setup(context);
  const exposed = store.recordImpression(impression(store, profile.id));
  throwsCode(() => store.recordClick({ eventId: 'before', campaignId: 'aurora', impressionId: exposed.id,
    occurredAt: now() - 1 }), 'INVALID_ATTRIBUTION');
  throwsCode(() => store.recordClick({ eventId: 'future', campaignId: 'aurora', impressionId: exposed.id,
    occurredAt: now() + 1 }), 'INVALID_INPUT');
  advance(100);
  throwsCode(() => store.recordClick({ eventId: 'at-end', campaignId: 'aurora', impressionId: exposed.id }), 'INVALID_ATTRIBUTION');
  advance(49);
  store.recordClick({ eventId: 'late-valid', campaignId: 'aurora', impressionId: exposed.id,
    occurredAt: exposed.occurredAt + 99 });
  advance(1);
  throwsCode(() => store.recordClick({ eventId: 'too-late', campaignId: 'aurora', impressionId: exposed.id,
    occurredAt: exposed.occurredAt + 99 }), 'INVALID_ATTRIBUTION');
  assert.equal(store.maturedRows({ campaignId: 'aurora' })[0].label, 1);
  assert.equal(store.metrics({ campaignId: 'aurora', asOf: exposed.occurredAt + 100 }).clickEvents, 0);
  throwsCode(() => store.maturedRows({ campaignId: 'aurora', asOf: now() + 1 }), 'INVALID_INPUT');
});

test('current telemetry consent gates clicks even when the exposure was authorized', context => {
  const { store, profile } = setup(context);
  const exposed = store.recordImpression(impression(store, profile.id));
  changeConsent(store, profile.id, 'stop-telemetry', { telemetry: false });
  throwsCode(() => store.recordClick({ eventId: 'blocked-click', campaignId: 'aurora', impressionId: exposed.id }), 'CONSENT_DENIED');
  assert.equal(store.metrics({ campaignId: 'aurora' }).exposures, 0);
});

for (const withdrawal of [
  { name: 'training', changes: { training: false } },
  { name: 'telemetry', changes: { telemetry: false } },
  { name: 'campaign', changes: { campaignIds: ['borealis'] } },
]) {
  test(`${withdrawal.name} regrant never resurrects old training exposures`, context => {
    const { store, profile, advance } = setup(context);
    store.recordImpression(impression(store, profile.id, 'before-revoke'));
    advance(150);
    assert.equal(store.maturedRows({ campaignId: 'aurora' }).length, 1);
    changeConsent(store, profile.id, 'withdraw', withdrawal.changes);
    assert.deepEqual(store.maturedRows({ campaignId: 'aurora' }), []);
    changeConsent(store, profile.id, 'restore', BASE_CONSENT);
    assert.deepEqual(store.maturedRows({ campaignId: 'aurora' }), []);
    store.recordImpression(impression(store, profile.id, 'after-regrant'));
    advance(150);
    assert.deepEqual(store.maturedRows({ campaignId: 'aurora' }).map(row => row.impressionId), ['after-regrant']);
  });
}

test('training disabled at exposure cannot be enabled retroactively', context => {
  const { store, profile, advance } = setup(context);
  changeConsent(store, profile.id, 'disable-training', { training: false });
  store.recordImpression(impression(store, profile.id));
  changeConsent(store, profile.id, 'enable-training', { training: true });
  advance(150);
  assert.deepEqual(store.maturedRows({ campaignId: 'aurora' }), []);
});

test('reviews are immutable, source separated and never converted into clicks or exposures', context => {
  const { store, profile } = setup(context);
  const input = review(store, profile.id);
  const stored = store.recordReview(input);
  assert.equal(Object.hasOwn(stored, 'brandRating'), false);
  assert.deepEqual(store.recordReview(input), stored);
  throwsCode(() => store.recordReview({ ...input, qualityRating: 2 }), 'CONFLICT');
  throwsCode(() => store.recordReview({ ...input, eventId: 'wrong-review', campaignId: 'borealis' }), 'INVALID_ATTRIBUTION');
  throwsCode(() => store.recordReview({ ...input, source: 'real-human-verified' }), 'INVALID_INPUT');
  store.recordReview(review(store, profile.id, 'synthetic-review', 'synthetic'));
  const metrics = store.metrics({ campaignId: 'aurora' });
  assert.deepEqual(metrics.reviews, { operator: 1, synthetic: 1 });
  assert.equal(metrics.exposures, 0);
  assert.equal(metrics.clickEvents, 0);
  assert.equal(metrics.ctr, null);
  assert.deepEqual(store.maturedRows({ campaignId: 'aurora' }), []);
});

test('reviews preserve optional integer brand grades independently from the pass gate', context => {
  const { store, profile } = setup(context);
  const input = { ...review(store, profile.id), brandRating: 3, brandPass: false };
  const stored = store.recordReview(input);
  assert.equal(stored.brandRating, 3);
  assert.equal(stored.brandPass, false);
  assert.deepEqual(store.recordReview(input), stored);
  throwsCode(() => store.recordReview({ ...input, brandRating: 4 }), 'CONFLICT');
  for (const brandRating of [1, 5]) {
    const result = store.recordReview({ ...input, eventId: `brand-grade-${brandRating}`, brandRating });
    assert.equal(result.brandRating, brandRating);
    assert.equal(result.brandPass, false);
  }
  for (const brandRating of [0, 6, 2.5, '4', null, undefined, NaN, Infinity]) {
    throwsCode(() => store.recordReview({ ...input, eventId: 'invalid-grade', brandRating }), 'INVALID_INPUT');
  }
  assert.equal(store.metrics({ campaignId: 'aurora' }).reviews.operator, 3);
});

test('campaign training and metrics remain partitioned when one campaign is revoked', context => {
  const { store, profile, advance } = setup(context);
  store.recordImpression(impression(store, profile.id, 'aurora-exposure'));
  store.recordImpression(impression(store, profile.id, 'borealis-exposure', 'borealis'));
  advance(150);
  changeConsent(store, profile.id, 'remove-aurora', { campaignIds: ['borealis'] });
  assert.equal(store.maturedRows({ campaignId: 'borealis' }).length, 1);
  assert.deepEqual(store.maturedRows({ campaignId: 'aurora' }), []);
  assert.equal(store.metrics({ campaignId: 'borealis' }).exposures, 1);
  changeConsent(store, profile.id, 'restore-aurora', { campaignIds: ['aurora', 'borealis'] });
  assert.deepEqual(store.maturedRows({ campaignId: 'aurora' }), []);
  assert.equal(store.maturedRows({ campaignId: 'borealis' }).length, 1);
});

test('deletion cascades through persisted personal data and cannot resurrect through retries or reopening', context => {
  const directory = mkdtempSync(join(tmpdir(), 'angelos-store-'));
  const path = join(directory, 'fixture.sqlite');
  let now = 1_000_000;
  const store = new Store(path, { clock: () => now, clickWindowMs: 100, allowedLatenessMs: 50 });
  context.after(() => rmSync(directory, { recursive: true, force: true }));
  try {
    const profile = store.createProfile({ name: 'Delete synthetic fixture', facts: BASE_FACTS, consent: BASE_CONSENT });
    const input = impression(store, profile.id);
    store.recordImpression(input);
    store.recordClick({ eventId: 'delete-click', campaignId: 'aurora', impressionId: input.eventId });
    store.recordReview(review(store, profile.id));
    now += 150;
    assert.equal(store.maturedRows({ campaignId: 'aurora' }).length, 1);
    const deleted = store.deleteProfile(profile.id);
    assert.deepEqual(deleted.affectedArtifactIds, [descriptor().id]);
    assert.equal(store.deleteProfile(profile.id).alreadyDeleted, true);
    assert.deepEqual(store.listProfiles(), []);
    assert.deepEqual(store.maturedRows({ campaignId: 'aurora' }), []);
    throwsCode(() => store.recordImpression(input), 'DELETED');
    throwsCode(() => store.snapshot(profile.id, 'aurora'), 'DELETED');
    throwsCode(() => store.setConsent({ profileId: profile.id, eventId: 'resurrect', expectedVersion: 1,
      ...BASE_CONSENT }), 'DELETED');
    const inspection = new DatabaseSync(path);
    try {
      for (const table of ['profiles', 'fact_events', 'consent_events', 'snapshots', 'impressions', 'clicks', 'reviews', 'receipts']) {
        assert.equal(inspection.prepare(`SELECT COUNT(*) AS count FROM ${table}`).get().count, 0, table);
      }
      assert.deepEqual(inspection.prepare('PRAGMA foreign_key_check').all(), []);
    } finally {
      inspection.close();
    }
    const reopened = new Store(path, { clock: () => now });
    try {
      throwsCode(() => reopened.getProfile(profile.id), 'DELETED');
      assert.deepEqual(reopened.maturedRows({ campaignId: 'aurora' }), []);
    } finally {
      reopened.close();
    }
  } finally {
    store.close();
  }
});

test('persisted label windows do not change when a store reopens with different defaults', context => {
  const directory = mkdtempSync(join(tmpdir(), 'angelos-store-'));
  context.after(() => rmSync(directory, { recursive: true, force: true }));
  const path = join(directory, 'fixture.sqlite');
  let now = 1_000_000;
  const original = new Store(path, { clock: () => now, clickWindowMs: 100, allowedLatenessMs: 50 });
  let profile;
  try {
    profile = original.createProfile({ name: 'Persistent fixture', facts: BASE_FACTS, consent: BASE_CONSENT });
    original.recordImpression(impression(original, profile.id));
  } finally {
    original.close();
  }
  now += 150;
  const reopened = new Store(path, { clock: () => new Date(now) });
  try {
    const rows = reopened.maturedRows({ campaignId: 'aurora' });
    assert.equal(rows.length, 1);
    assert.equal(rows[0].labelPolicy.clickWindowMs, 100);
    assert.equal(rows[0].labelPolicy.allowedLatenessMs, 50);
    assert.deepEqual(reopened.getProfile(profile.id).facts, BASE_FACTS);
  } finally {
    reopened.close();
  }
});
