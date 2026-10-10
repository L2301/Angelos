import test from 'node:test';
import assert from 'node:assert/strict';
import { Store, ENCODER } from '../src/core/store.js';
import { validateSemanticBundle, validateUplFact } from '../src/core/upl.js';
import { semanticFixture } from './upl-fixture.js';

function setup(context) {
  let now = 1000000;
  const store = new Store(':memory:', { clock: () => now, clickWindowMs: 100, allowedLatenessMs: 50 });
  context.after(() => store.close());
  const profile = store.createProfile({ name: 'Synthetic test adult',
    facts: { ageBand: '25-34', region: 'EU', openness: 0.2, energy: 0.8, interests: ['hiking', 'photography'],
      goals: ['weekend outdoors'], preferredStyles: ['natural light'], preferredLanguage: 'English' },
    consent: { personalization: true, telemetry: true, training: true, campaignIds: ['aurora', 'other'] } });
  const activity = (changes = {}) => store.recordActivity({ profileId: profile.id, eventId: 'activity-1',
    campaignId: 'aurora', kind: 'view', topic: 'lightweight backpacks', ...changes });
  const encode = () => {
    const ingredients = store.uplIngredients(profile.id, 'aurora');
    return store.saveSemanticSnapshot(profile.id, 'aurora', ingredients, semanticFixture(ingredients));
  };
  const impression = snapshot => ({ eventId: 'impression-1', profileId: profile.id, campaignId: 'aurora',
    snapshotId: snapshot.id, artifact: { id: 'synthetic-semantic-test-artifact', campaignId: 'aurora',
      hash: 'a'.repeat(64), graphHash: 'b'.repeat(64), encoder: snapshot.encoder },
    context: { creativeId: 'fixture:1', placementId: 'local', actionSetId: 'single', policyVersion: 'test', propensityVersion: 'single', propensity: 1 },
    exposure: { visibleFraction: 1, visibleMilliseconds: 1000, documentVisible: true } });
  const permission = changes => {
    const current = store.getProfile(profile.id);
    return store.setConsent({ profileId: profile.id, eventId: `consent-${current.consentVersion}`, expectedVersion: current.consentVersion, ...current.consent, ...changes });
  };
  return { store, profile, activity, encode, impression, permission, advance: duration => { now += duration; } };
}

test('UPL ingredients preserve richer facts independently of legacy coordinates and encoder', context => {
  const { store, profile } = setup(context);
  const ingredients = store.uplIngredients(profile.id, 'aurora');
  assert.equal(ingredients.format, 'angelos.upl-ingredients.v2');
  assert.deepEqual(ingredients.facts.interests, ['hiking', 'photography']);
  assert.equal(ingredients.provenance.length, 8);
  assert.equal(ingredients.provenance[0].source, 'synthetic-declared');
  assert.equal('profileId' in ingredients, false);
  assert.equal('name' in ingredients, false);
  assert.equal('upl' in ingredients, false);
  assert.ok(Object.isFrozen(ingredients.facts.interests));
  const legacy = store.snapshot(profile.id, 'aurora');
  assert.equal(legacy.upl.length, 4);
  assert.deepEqual(Object.keys(legacy.facts).sort(), ['ageBand', 'energy', 'openness', 'region']);
  for (const value of [['duplicate', 'duplicate'], new Array(17).fill('x'), [''], ['x'.repeat(81)], ['bad\ntext'], [123]]) {
    assert.throws(() => validateUplFact('interests', value));
  }
});

test('live intent is idempotent, campaign scoped, time bounded and separate from declared facts', context => {
  const { store, profile, activity, advance } = setup(context);
  const first = activity({ expiresAt: 1000050 });
  assert.deepEqual(activity({ expiresAt: 1000050 }), first);
  assert.throws(() => activity({ topic: 'changed' }), error => error.code === 'CONFLICT');
  activity({ eventId: 'other-event', campaignId: 'other', kind: 'dismiss', topic: 'irrelevant item' });
  const ingredients = store.uplIngredients(profile.id, 'aurora');
  assert.equal(ingredients.events.length, 1);
  assert.equal(ingredients.eventRevision, 1);
  assert.deepEqual(store.getProfile(profile.id).facts, profile.facts);
  advance(51);
  assert.deepEqual(store.uplIngredients(profile.id, 'aurora').events, []);
  assert.throws(() => activity({ eventId: 'future', occurredAt: 1001000 }));
  assert.throws(() => activity({ eventId: 'wrong', kind: 'purchase' }));
});

test('v2 snapshots bind complete encoder, signed vector and context without claiming model inference', context => {
  const { store, profile, activity, encode } = setup(context);
  activity();
  const snapshot = encode();
  assert.equal(snapshot.upl.length, 512);
  assert.ok(snapshot.upl.some(value => value < 0));
  assert.equal(snapshot.bundle.context.eventRevision, 1);
  assert.equal(snapshot.bundle.recentIntent[0].topic, 'lightweight backpacks');
  assert.equal(snapshot.encoder.normalization, 'l2');
  assert.ok(Object.isFrozen(snapshot.bundle.recentIntent[0]));
  const ingredients = store.uplIngredients(profile.id, 'aurora');
  for (const change of [bundle => { bundle.upl[0] = NaN; }, bundle => { bundle.upl.fill(0); },
    bundle => { bundle.encoder.dimension = 128; }, bundle => { bundle.context.ingredientsHash = '0'.repeat(64); },
    bundle => { bundle.recentIntent = []; }, bundle => { bundle.context.campaignId = 'other'; },
    bundle => { bundle.facts.interests = ['invented']; }]) {
    const bundle = semanticFixture(ingredients);
    change(bundle);
    assert.throws(() => validateSemanticBundle(bundle, ingredients));
  }
});

test('facts, events, permissions and expiry changes reject in-flight semantic encoding', context => {
  const { store, profile, activity, permission, advance } = setup(context);
  const ingredients = store.uplIngredients(profile.id, 'aurora');
  activity({ expiresAt: 1000050 });
  assert.throws(() => store.saveSemanticSnapshot(profile.id, 'aurora', ingredients, semanticFixture(ingredients)), /changed during encoding/);
  const withEvent = store.uplIngredients(profile.id, 'aurora');
  advance(51);
  assert.throws(() => store.saveSemanticSnapshot(profile.id, 'aurora', withEvent, semanticFixture(withEvent)), /changed during encoding/);
  const current = store.uplIngredients(profile.id, 'aurora');
  permission({ training: false });
  assert.throws(() => store.saveSemanticSnapshot(profile.id, 'aurora', current, semanticFixture(current)), /changed during encoding/);
  const beforeFact = store.uplIngredients(profile.id, 'aurora');
  store.updateFact({ profileId: profile.id, eventId: 'fact', expectedRevision: profile.factRevision, field: 'interests', value: ['camping'] });
  assert.throws(() => store.saveSemanticSnapshot(profile.id, 'aurora', beforeFact, semanticFixture(beforeFact)), /changed during encoding/);
});

test('revocation prevents intent collection and regrant does not resurrect withdrawn activity', context => {
  const { store, profile, activity, permission } = setup(context);
  activity();
  permission({ telemetry: false });
  assert.deepEqual(store.uplIngredients(profile.id, 'aurora').events, []);
  assert.throws(() => activity({ eventId: 'denied' }), error => error.code === 'CONSENT_DENIED');
  permission({ telemetry: true });
  assert.deepEqual(store.uplIngredients(profile.id, 'aurora').events, []);
  permission({ personalization: false });
  assert.throws(() => store.uplIngredients(profile.id, 'aurora'), error => error.code === 'CONSENT_DENIED');
  store.deleteProfile(profile.id);
  assert.throws(() => store.uplIngredients(profile.id, 'aurora'), error => error.code === 'DELETED');
});

test('semantic exposure rejects legacy model, preserves frozen full bundle and updates click intent', context => {
  const { store, profile, activity, encode, impression, advance } = setup(context);
  activity();
  const snapshot = encode();
  const input = impression(snapshot);
  assert.throws(() => store.recordImpression({ ...input, artifact: { ...input.artifact, encoder: ENCODER } }), /encoder contracts differ/);
  store.recordImpression(input);
  advance(1);
  store.recordClick({ eventId: 'click-1', campaignId: 'aurora', impressionId: input.eventId });
  assert.equal(store.uplIngredients(profile.id, 'aurora').events.at(-1).kind, 'click');
  assert.equal(store.getProfile(profile.id).eventRevision, 2);
  store.recordClick({ eventId: 'click-1', campaignId: 'aurora', impressionId: input.eventId });
  assert.equal(store.getProfile(profile.id).eventRevision, 2);
  assert.throws(() => store.recordImpression({ ...input, eventId: 'stale' }), error => error.code === 'STALE_SNAPSHOT');
  advance(200);
  const feedback = store.learningFeedback({ campaignId: 'aurora' });
  assert.equal(feedback.clicks[0].label, 1);
  assert.deepEqual(feedback.clicks[0].uplBundle, snapshot.bundle);
  assert.equal(feedback.clicks[0].uplBundle.recentIntent.length, 1);
});

test('a late valid attributed click does not become fresh intent or lose its label', context => {
  const day = 86400000;
  let now = 1000000;
  const store = new Store(':memory:', { clock: () => now, clickWindowMs: 30 * day, allowedLatenessMs: 30 * day });
  context.after(() => store.close());
  const profile = store.createProfile({ name: 'Synthetic delayed-click fixture',
    facts: { ageBand: '25-34', region: 'NA', openness: 0.5, energy: 0.5 },
    consent: { personalization: true, telemetry: true, training: true, campaignIds: ['aurora'] } });
  const snapshot = store.snapshot(profile.id, 'aurora');
  store.recordImpression({ eventId: 'impression', profileId: profile.id, campaignId: 'aurora', snapshotId: snapshot.id,
    artifact: { id: 'legacy-fixture', campaignId: 'aurora', encoder: ENCODER, hash: 'a'.repeat(64), graphHash: 'b'.repeat(64) },
    context: { creativeId: 'fixture', placementId: 'local', actionSetId: 'single', policyVersion: 'test', propensityVersion: 'single', propensity: 1 },
    exposure: { visibleFraction: 1, visibleMilliseconds: 1000, documentVisible: true } });
  now += 40 * day;
  const click = store.recordClick({ eventId: 'delayed', campaignId: 'aurora', impressionId: 'impression', occurredAt: 1000001 });
  assert.equal(click.intentUpdate, 'outside-recent-window');
  assert.deepEqual(store.uplIngredients(profile.id, 'aurora').events, []);
  now += 21 * day;
  assert.equal(store.maturedRows({ campaignId: 'aurora' })[0].label, 1);
});
