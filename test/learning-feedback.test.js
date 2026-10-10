import test from 'node:test';
import assert from 'node:assert/strict';
import { Store, ENCODER } from '../src/core/store.js';

test('learning export binds frozen UPL and honors maturity, consent revocation and deletion', context => {
  let now = 1000;
  const store = new Store(':memory:', { clock: () => now, clickWindowMs: 100, allowedLatenessMs: 50 });
  context.after(() => store.close());
  const consent = { personalization: true, telemetry: true, training: true, campaignIds: ['aurora'] };
  const profile = store.createProfile({ name: 'Synthetic adult', facts: { ageBand: '25-34', region: 'NA', openness: 0.7, energy: 0.2 }, consent });
  const snapshot = store.snapshot(profile.id, 'aurora');
  const artifact = { id: 'student-v1', campaignId: 'aurora', hash: 'a'.repeat(64), graphHash: 'b'.repeat(64), encoder: ENCODER };
  store.recordImpression({ eventId: 'impression-1', profileId: profile.id, campaignId: 'aurora', snapshotId: snapshot.id, artifact,
    context: { creativeId: 'creative:abc:1', placementId: 'local', actionSetId: 'single', policyVersion: 'operator', propensityVersion: 'single', propensity: 1 },
    exposure: { visibleFraction: 1, visibleMilliseconds: 1000, documentVisible: true } });
  store.recordReview({ eventId: 'review-1', profileId: profile.id, campaignId: 'aurora', snapshotId: snapshot.id, artifact, creativeId: 'creative:abc:1', qualityRating: 4, brandPass: true, source: 'operator' });
  assert.equal(store.learningFeedback({ campaignId: 'aurora' }).clicks.length, 0);
  now += 200;
  store.updateFact({ profileId: profile.id, eventId: 'fact-1', expectedRevision: profile.factRevision, field: 'openness', value: 0.1 });
  const feedback = store.learningFeedback({ campaignId: 'aurora' });
  assert.equal(feedback.clicks[0].label, 0);
  assert.deepEqual(feedback.clicks[0].upl, snapshot.upl);
  assert.deepEqual(feedback.reviews[0].upl, snapshot.upl);
  assert.equal(feedback.outcome, 'click');
  store.setConsent({ profileId: profile.id, eventId: 'revoke', expectedVersion: profile.consentVersion, ...consent, training: false });
  assert.equal(store.learningFeedback({ campaignId: 'aurora' }).reviews.length, 0);
  assert.equal(store.learningFeedback({ campaignId: 'aurora' }).clicks.length, 0);
  store.setConsent({ profileId: profile.id, eventId: 'regrant', expectedVersion: profile.consentVersion + 1, ...consent });
  assert.equal(store.learningFeedback({ campaignId: 'aurora' }).reviews.length, 0);
  store.deleteProfile(profile.id);
  assert.equal(store.learningFeedback({ campaignId: 'aurora' }).reviews.length, 0);
});
