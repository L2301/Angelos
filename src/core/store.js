import { createHash, randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import { UPL_FIELDS, UPL_FACT_SCHEMA, UPL_EVENT_SCHEMA, UPL_EVENT_KINDS, ADULT_AGE_BANDS,
  validateUplFact, validateSemanticEncoder, validateSemanticBundle, uplText } from './upl.js';

export const ENCODER = Object.freeze({ id: 'structured-demo-v1', dimension: 4 });
export const FACT_SCHEMA_VERSION = 'angelos.facts.v1';
export const CONSENT_SCHEMA_VERSION = 'angelos.consent.v1';

const AGE_BANDS = Object.freeze(['18-24', '25-34', '35-44', '45-54', '55-64', '65+']);
const REGIONS = Object.freeze(['NA', 'EU', 'APAC', 'OTHER']);
const LEGACY_FIELDS = Object.freeze(['ageBand', 'region', 'openness', 'energy']);
const FIELDS = UPL_FIELDS;
const DAY = 86_400_000;
const MAX_ROWS = 10_000;

function fail(code, message) {
  throw Object.assign(new Error(message), { code });
}

function object(value, allowed, required = allowed) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || ![Object.prototype, null].includes(Object.getPrototypeOf(value))) {
    fail('INVALID_INPUT', 'Expected a plain object');
  }
  for (const key of Reflect.ownKeys(value)) {
    if (typeof key !== 'string' || !allowed.includes(key)
        || !Object.hasOwn(Object.getOwnPropertyDescriptor(value, key), 'value')) {
      fail('INVALID_INPUT', `Unexpected property: ${String(key)}`);
    }
  }
  for (const key of required) {
    if (!Object.hasOwn(value, key)) fail('INVALID_INPUT', `Missing property: ${key}`);
  }
}

function identifier(value, label = 'identifier') {
  if (typeof value !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(value)) {
    fail('INVALID_INPUT', `Invalid ${label}`);
  }
}

function integer(value, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    fail('INVALID_INPUT', `Expected an integer in [${minimum}, ${maximum}]`);
  }
}

function unit(value) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) {
    fail('INVALID_INPUT', 'Expected a finite number in [0, 1]');
  }
}

function boolean(value) {
  if (typeof value !== 'boolean') fail('INVALID_INPUT', 'Expected a boolean');
}

function fact(field, value) {
  validateUplFact(field, value);
}

function facts(value) {
  object(value, FIELDS, []);
  for (const [field, entry] of Object.entries(value)) fact(field, entry);
}

function consent(value) {
  object(value, ['personalization', 'telemetry', 'training', 'campaignIds']);
  for (const scope of ['personalization', 'telemetry', 'training']) boolean(value[scope]);
  if (!Array.isArray(value.campaignIds) || value.campaignIds.length > 100) {
    fail('INVALID_INPUT', 'Expected at most 100 campaign IDs');
  }
  for (const campaignId of value.campaignIds) identifier(campaignId, 'campaign ID');
  if (new Set(value.campaignIds).size !== value.campaignIds.length) {
    fail('INVALID_INPUT', 'Duplicate campaign ID');
  }
}

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function freeze(value) {
  if (value && typeof value === 'object') {
    for (const entry of Object.values(value)) freeze(entry);
    Object.freeze(value);
  }
  return value;
}

function artifact(value) {
  object(value, ['id', 'campaignId', 'hash', 'graphHash', 'encoder']);
  identifier(value.id, 'artifact ID');
  identifier(value.campaignId, 'campaign ID');
  for (const key of ['hash', 'graphHash']) {
    if (typeof value[key] !== 'string' || !/^[a-f0-9]{64}$/.test(value[key])) {
      fail('INVALID_INPUT', `Invalid ${key}`);
    }
  }
  if (value.encoder?.id === ENCODER.id) {
    object(value.encoder, ['id', 'dimension']);
    if (value.encoder.dimension !== ENCODER.dimension) fail('INCOMPATIBLE_ARTIFACT', 'Unsupported encoder contract');
  } else {
    try { validateSemanticEncoder(value.encoder); }
    catch { fail('INCOMPATIBLE_ARTIFACT', 'Unsupported encoder contract'); }
  }
}

export function encodeFacts(value) {
  facts(value);
  const missing = LEGACY_FIELDS.filter(field => !Object.hasOwn(value, field) || value[field] === null);
  if (missing.length) fail('MISSING_FACTS', `Missing encoder facts: ${missing.join(', ')}`);
  return [AGE_BANDS.indexOf(value.ageBand) / 5, REGIONS.indexOf(value.region) / 3,
    value.openness, value.energy];
}

export class Store {
  #database;
  #clock;
  #clickWindowMs;
  #allowedLatenessMs;

  constructor(path = ':memory:', options = {}) {
    object(options, ['clock', 'clickWindowMs', 'allowedLatenessMs'], []);
    if (typeof path !== 'string' || !path.length) fail('INVALID_INPUT', 'Expected database path');
    this.#clock = options.clock ?? Date.now;
    if (typeof this.#clock !== 'function') fail('INVALID_INPUT', 'Expected clock function');
    this.#clickWindowMs = options.clickWindowMs ?? DAY;
    this.#allowedLatenessMs = options.allowedLatenessMs ?? DAY;
    integer(this.#clickWindowMs, 1, 30 * DAY);
    integer(this.#allowedLatenessMs, 0, 30 * DAY);
    this.#database = new DatabaseSync(path, { enableForeignKeyConstraints: true, allowExtension: false });
    try {
      const version = this.#database.prepare('PRAGMA user_version').get().user_version;
      if (![0, 1, 2].includes(version)) fail('INCOMPATIBLE_STORE', 'Unsupported database schema');
      this.#database.exec(`
        PRAGMA foreign_keys = ON;
        PRAGMA secure_delete = ON;
        PRAGMA busy_timeout = 1000;
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS profiles (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, facts TEXT NOT NULL,
          fact_revision INTEGER NOT NULL, consent TEXT NOT NULL,
          consent_version INTEGER NOT NULL, created_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS fact_events (
          id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
          revision INTEGER NOT NULL, field TEXT NOT NULL, value TEXT NOT NULL,
          occurred_at INTEGER NOT NULL, received_at INTEGER NOT NULL,
          UNIQUE(profile_id, revision)
        );
        CREATE TABLE IF NOT EXISTS consent_events (
          id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
          version INTEGER NOT NULL, consent TEXT NOT NULL, received_at INTEGER NOT NULL,
          UNIQUE(profile_id, version)
        );
        CREATE TABLE IF NOT EXISTS snapshots (
          id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
          campaign_id TEXT NOT NULL, payload TEXT NOT NULL,
          UNIQUE(id, profile_id, campaign_id)
        );
        CREATE TABLE IF NOT EXISTS artifacts (
          id TEXT PRIMARY KEY, descriptor TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS impressions (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
          profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
          campaign_id TEXT NOT NULL, snapshot_id TEXT NOT NULL,
          artifact_id TEXT NOT NULL REFERENCES artifacts(id), occurred_at INTEGER NOT NULL,
          received_at INTEGER NOT NULL, click_deadline INTEGER NOT NULL, matures_at INTEGER NOT NULL,
          training_eligible INTEGER NOT NULL CHECK(training_eligible IN (0, 1)), payload TEXT NOT NULL,
          UNIQUE(id, profile_id, campaign_id),
          FOREIGN KEY(snapshot_id, profile_id, campaign_id)
            REFERENCES snapshots(id, profile_id, campaign_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS impressions_campaign ON impressions(campaign_id, matures_at);
        CREATE TABLE IF NOT EXISTS clicks (
          id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
          campaign_id TEXT NOT NULL, impression_id TEXT NOT NULL,
          occurred_at INTEGER NOT NULL, received_at INTEGER NOT NULL, payload TEXT NOT NULL,
          FOREIGN KEY(impression_id, profile_id, campaign_id)
            REFERENCES impressions(id, profile_id, campaign_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS clicks_impression ON clicks(impression_id, received_at);
        CREATE TABLE IF NOT EXISTS reviews (
          id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
          campaign_id TEXT NOT NULL, snapshot_id TEXT NOT NULL,
          artifact_id TEXT NOT NULL REFERENCES artifacts(id), received_at INTEGER NOT NULL,
          training_eligible INTEGER NOT NULL CHECK(training_eligible IN (0, 1)), payload TEXT NOT NULL,
          FOREIGN KEY(snapshot_id, profile_id, campaign_id)
            REFERENCES snapshots(id, profile_id, campaign_id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS receipts (
          profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
          kind TEXT NOT NULL, event_id TEXT NOT NULL, input TEXT NOT NULL, result TEXT NOT NULL,
          PRIMARY KEY(profile_id, kind, event_id)
        );
        CREATE TABLE IF NOT EXISTS deleted_profiles (id TEXT PRIMARY KEY, deleted_at INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS upl_events (
          id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
          campaign_id TEXT NOT NULL, revision INTEGER NOT NULL, kind TEXT NOT NULL,
          topic TEXT NOT NULL, occurred_at INTEGER NOT NULL, received_at INTEGER NOT NULL,
          expires_at INTEGER NOT NULL, source TEXT NOT NULL, UNIQUE(profile_id, revision)
        );
        CREATE INDEX IF NOT EXISTS upl_events_profile ON upl_events(profile_id, campaign_id, occurred_at);
        PRAGMA user_version = 2;
        COMMIT;
      `);
    } catch (error) {
      this.#database.close();
      throw error;
    }
  }

  #now() {
    const reading = this.#clock();
    const timestamp = reading instanceof Date ? reading.getTime() : reading;
    integer(timestamp, 0, Number.MAX_SAFE_INTEGER - 60 * DAY);
    return timestamp;
  }

  #transaction(action) {
    this.#database.exec('BEGIN IMMEDIATE');
    try {
      const result = action();
      this.#database.exec('COMMIT');
      return result;
    } catch (error) {
      this.#database.exec('ROLLBACK');
      throw error;
    }
  }

  #profile(profileId) {
    const row = this.#database.prepare('SELECT * FROM profiles WHERE id = ?').get(profileId);
    if (!row) {
      if (this.#database.prepare('SELECT id FROM deleted_profiles WHERE id = ?').get(profileId)) {
        fail('DELETED', 'Profile was deleted');
      }
      fail('NOT_FOUND', 'Profile not found');
    }
    const eventRevision = this.#database.prepare('SELECT COALESCE(MAX(revision), 0) AS revision FROM upl_events WHERE profile_id = ?').get(profileId).revision;
    return { id: row.id, name: row.name, synthetic: true, facts: JSON.parse(row.facts), eventRevision,
      factRevision: row.fact_revision, consent: JSON.parse(row.consent),
      consentVersion: row.consent_version, createdAt: row.created_at };
  }

  #permit(profile, campaignId, scopes) {
    if (!profile.consent.campaignIds.includes(campaignId)
        || scopes.some(scope => !profile.consent[scope])) {
      fail('CONSENT_DENIED', 'Campaign or scope is not permitted');
    }
  }

  #campaignEventRevision(profileId, campaignId) {
    return this.#database.prepare('SELECT COUNT(*) AS revision FROM upl_events WHERE profile_id = ? AND campaign_id = ?').get(profileId, campaignId).revision;
  }

  #event(profileId, kind, eventId, input, action) {
    return this.#transaction(() => {
      const profile = this.#profile(profileId);
      const serialized = canonical(input);
      const previous = this.#database.prepare(
        'SELECT input, result FROM receipts WHERE profile_id = ? AND kind = ? AND event_id = ?',
      ).get(profileId, kind, eventId);
      if (previous) {
        if (previous.input !== serialized) fail('CONFLICT', 'Idempotency key reused with different payload');
        return JSON.parse(previous.result);
      }
      const result = action(profile, this.#now());
      this.#database.prepare('INSERT INTO receipts VALUES (?, ?, ?, ?, ?)')
        .run(profileId, kind, eventId, serialized, canonical(result));
      return JSON.parse(canonical(result));
    });
  }

  #occurred(input, now) {
    const timestamp = Object.hasOwn(input, 'occurredAt') ? input.occurredAt : now;
    integer(timestamp, 0, now);
    return timestamp;
  }

  #snapshot(snapshotId, profile, campaignId, fresh) {
    const row = this.#database.prepare('SELECT * FROM snapshots WHERE id = ?').get(snapshotId);
    if (!row || row.profile_id !== profile.id || row.campaign_id !== campaignId) {
      fail('INVALID_ATTRIBUTION', 'Snapshot does not belong to this profile and campaign');
    }
    const result = JSON.parse(row.payload);
    if (fresh && (result.factRevision !== profile.factRevision
        || result.consentVersion !== profile.consentVersion
        || (result.bundle && (result.eventRevision !== this.#campaignEventRevision(profile.id, campaignId)
          || result.bundle.recentIntent.some(event => event.expiresAt <= this.#now()))))) {
      fail('STALE_SNAPSHOT', 'Create a snapshot for the current facts and consent');
    }
    return result;
  }

  #artifact(descriptor, campaignId) {
    if (descriptor.campaignId !== campaignId) fail('INCOMPATIBLE_ARTIFACT', 'Artifact campaign mismatch');
    const serialized = canonical(descriptor);
    const previous = this.#database.prepare('SELECT descriptor FROM artifacts WHERE id = ?').get(descriptor.id);
    if (previous && previous.descriptor !== serialized) {
      fail('INCOMPATIBLE_ARTIFACT', 'Artifact ID is already bound to another descriptor');
    }
    if (!previous) this.#database.prepare('INSERT INTO artifacts VALUES (?, ?)').run(descriptor.id, serialized);
  }

  createProfile(input) {
    object(input, ['name', 'facts', 'consent'], ['name']);
    if (typeof input.name !== 'string' || !input.name.trim() || input.name.length > 100
        || /[\u0000-\u001f\u007f]/.test(input.name)) fail('INVALID_INPUT', 'Invalid profile name');
    const initialFacts = Object.hasOwn(input, 'facts') ? input.facts : {};
    const initialConsent = Object.hasOwn(input, 'consent') ? input.consent
      : { personalization: false, telemetry: false, training: false, campaignIds: [] };
    facts(initialFacts);
    consent(initialConsent);
    return this.#transaction(() => {
      const profileId = randomUUID();
      const now = this.#now();
      const entries = Object.entries(initialFacts).sort(([left], [right]) => left.localeCompare(right));
      this.#database.prepare('INSERT INTO profiles VALUES (?, ?, ?, ?, ?, ?, ?)')
        .run(profileId, input.name, canonical(initialFacts), entries.length, canonical(initialConsent), 1, now);
      for (const [index, [field, value]] of entries.entries()) {
        this.#database.prepare('INSERT INTO fact_events VALUES (?, ?, ?, ?, ?, ?, ?)')
          .run(randomUUID(), profileId, index + 1, field, canonical(value), now, now);
      }
      this.#database.prepare('INSERT INTO consent_events VALUES (?, ?, ?, ?, ?)')
        .run(randomUUID(), profileId, 1, canonical(initialConsent), now);
      return this.#profile(profileId);
    });
  }

  listProfiles() {
    const rows = this.#database.prepare('SELECT id FROM profiles ORDER BY created_at, id LIMIT ?').all(MAX_ROWS + 1);
    if (rows.length > MAX_ROWS) fail('LIMIT_EXCEEDED', 'Demo profile limit exceeded');
    return rows.map(row => this.#profile(row.id));
  }

  getProfile(profileId) {
    identifier(profileId, 'profile ID');
    return this.#profile(profileId);
  }

  updateFact(input) {
    object(input, ['profileId', 'eventId', 'expectedRevision', 'field', 'value', 'occurredAt', 'campaignId'],
      ['profileId', 'eventId', 'expectedRevision', 'field', 'value']);
    identifier(input.profileId);
    identifier(input.eventId);
    integer(input.expectedRevision);
    fact(input.field, input.value);
    if (Object.hasOwn(input, 'campaignId')) identifier(input.campaignId);
    if (Object.hasOwn(input, 'occurredAt')) integer(input.occurredAt);
    return this.#event(input.profileId, 'fact', input.eventId, input, (profile, now) => {
      if (input.campaignId) this.#permit(profile, input.campaignId, ['personalization']);
      if (profile.factRevision !== input.expectedRevision) fail('CONFLICT', 'Stale fact revision');
      const occurredAt = this.#occurred(input, now);
      const previous = this.#database.prepare(
        'SELECT occurred_at FROM fact_events WHERE profile_id = ? AND field = ? ORDER BY revision DESC LIMIT 1',
      ).get(profile.id, input.field);
      if (previous && occurredAt < previous.occurred_at) fail('CONFLICT', 'Backdated fact update');
      const revision = profile.factRevision + 1;
      const updated = { ...profile.facts, [input.field]: input.value };
      this.#database.prepare('INSERT INTO fact_events VALUES (?, ?, ?, ?, ?, ?, ?)')
        .run(input.eventId, profile.id, revision, input.field, canonical(input.value), occurredAt, now);
      this.#database.prepare('UPDATE profiles SET facts = ?, fact_revision = ? WHERE id = ?')
        .run(canonical(updated), revision, profile.id);
      return { eventId: input.eventId, profileId: profile.id, factRevision: revision,
        field: input.field, value: input.value, occurredAt, receivedAt: now, schemaVersion: FACT_SCHEMA_VERSION };
    });
  }

  setConsent(input) {
    object(input, ['profileId', 'eventId', 'expectedVersion', 'personalization', 'telemetry', 'training', 'campaignIds']);
    identifier(input.profileId);
    identifier(input.eventId);
    integer(input.expectedVersion, 1);
    const updated = { personalization: input.personalization, telemetry: input.telemetry,
      training: input.training, campaignIds: input.campaignIds };
    consent(updated);
    return this.#event(input.profileId, 'consent', input.eventId, input, (profile, now) => {
      if (profile.consentVersion !== input.expectedVersion) fail('CONFLICT', 'Stale consent version');
      const version = profile.consentVersion + 1;
      const removed = profile.consent.campaignIds.filter(campaignId => !updated.campaignIds.includes(campaignId));
      if (!updated.personalization || !updated.telemetry) {
        this.#database.prepare('UPDATE upl_events SET expires_at = MIN(expires_at, ?) WHERE profile_id = ?').run(now, profile.id);
      } else {
        for (const campaignId of removed) {
          this.#database.prepare('UPDATE upl_events SET expires_at = MIN(expires_at, ?) WHERE profile_id = ? AND campaign_id = ?').run(now, profile.id, campaignId);
        }
      }
      for (const table of ['impressions', 'reviews']) {
        if (!updated.training || !updated.telemetry) {
          this.#database.prepare(`UPDATE ${table} SET training_eligible = 0 WHERE profile_id = ?`).run(profile.id);
        } else {
          for (const campaignId of removed) {
            this.#database.prepare(`UPDATE ${table} SET training_eligible = 0 WHERE profile_id = ? AND campaign_id = ?`)
              .run(profile.id, campaignId);
          }
        }
      }
      this.#database.prepare('INSERT INTO consent_events VALUES (?, ?, ?, ?, ?)')
        .run(input.eventId, profile.id, version, canonical(updated), now);
      this.#database.prepare('UPDATE profiles SET consent = ?, consent_version = ? WHERE id = ?')
        .run(canonical(updated), version, profile.id);
      return { eventId: input.eventId, profileId: profile.id, consent: updated, consentVersion: version,
        receivedAt: now, schemaVersion: CONSENT_SCHEMA_VERSION };
    });
  }

  snapshot(profileId, campaignId) {
    identifier(profileId);
    identifier(campaignId);
    return this.#transaction(() => {
      const profile = this.#profile(profileId);
      this.#permit(profile, campaignId, ['personalization']);
      const result = { id: randomUUID(), profileId, campaignId,
        facts: Object.fromEntries(LEGACY_FIELDS.filter(field => Object.hasOwn(profile.facts, field)).map(field => [field, profile.facts[field]])),
        factRevision: profile.factRevision, factSchemaVersion: FACT_SCHEMA_VERSION,
        encoder: ENCODER, upl: encodeFacts(profile.facts), consentVersion: profile.consentVersion,
        consentSchemaVersion: CONSENT_SCHEMA_VERSION, createdAt: this.#now() };
      result.vector = result.upl;
      result.hash = createHash('sha256').update(canonical(result)).digest('hex');
      this.#database.prepare('INSERT INTO snapshots VALUES (?, ?, ?, ?)')
        .run(result.id, profileId, campaignId, canonical(result));
      return freeze(result);
    });
  }

  #ingredients(profile, campaignId, now) {
    this.#permit(profile, campaignId, ['personalization']);
    if (!ADULT_AGE_BANDS.includes(profile.facts.ageBand)) fail('MISSING_FACTS', 'An adult age band is required');
    const provenance = this.#database.prepare(`SELECT field, occurred_at, received_at FROM fact_events
      WHERE profile_id = ? AND revision IN
        (SELECT MAX(revision) FROM fact_events WHERE profile_id = ? GROUP BY field) ORDER BY field`)
      .all(profile.id, profile.id).map(row => ({ field: row.field, source: 'synthetic-declared', occurredAt: row.occurred_at, receivedAt: row.received_at }));
    const events = profile.consent.telemetry ? this.#database.prepare(`SELECT * FROM upl_events
      WHERE profile_id = ? AND campaign_id = ? AND occurred_at <= ? AND received_at <= ? AND expires_at > ?
      ORDER BY occurred_at DESC, revision DESC LIMIT 64`).all(profile.id, campaignId, now, now, now)
      .reverse().map(row => ({ id: row.id, kind: row.kind, topic: row.topic, occurredAt: row.occurred_at,
        receivedAt: row.received_at, expiresAt: row.expires_at, source: row.source, campaignId: row.campaign_id })) : [];
    return { format: 'angelos.upl-ingredients.v2', synthetic: true, adult: true,
      factsSchema: UPL_FACT_SCHEMA, facts: profile.facts, provenance, eventsSchema: UPL_EVENT_SCHEMA,
      events, factRevision: profile.factRevision, eventRevision: this.#campaignEventRevision(profile.id, campaignId),
      consentVersion: profile.consentVersion, campaignId, asOf: now };
  }

  uplIngredients(profileId, campaignId) {
    identifier(profileId);
    identifier(campaignId);
    return this.#transaction(() => freeze(this.#ingredients(this.#profile(profileId), campaignId, this.#now())));
  }

  #appendActivity(profile, input, now, source) {
    const revision = profile.eventRevision + 1;
    const occurredAt = this.#occurred(input, now);
    const expiresAt = input.expiresAt ?? occurredAt + 7 * DAY;
    integer(expiresAt, now + 1, occurredAt + 30 * DAY);
    if (this.#database.prepare('SELECT COUNT(*) AS count FROM upl_events WHERE profile_id = ?').get(profile.id).count >= MAX_ROWS) {
      fail('LIMIT_EXCEEDED', 'Synthetic event limit exceeded');
    }
    this.#database.prepare('INSERT INTO upl_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)')
      .run(input.eventId, profile.id, input.campaignId, revision, input.kind, input.topic, occurredAt, now, expiresAt, source);
    return { id: input.eventId, campaignId: input.campaignId, eventRevision: revision, kind: input.kind,
      topic: input.topic, occurredAt, receivedAt: now, expiresAt, source, schemaVersion: UPL_EVENT_SCHEMA };
  }

  recordActivity(input) {
    object(input, ['profileId', 'eventId', 'campaignId', 'kind', 'topic', 'occurredAt', 'expiresAt'],
      ['profileId', 'eventId', 'campaignId', 'kind', 'topic']);
    for (const field of ['profileId', 'eventId', 'campaignId']) identifier(input[field]);
    if (!UPL_EVENT_KINDS.includes(input.kind)) fail('INVALID_INPUT', 'Unsupported activity kind');
    uplText(input.topic);
    return this.#event(input.profileId, 'activity', input.eventId, input, (profile, now) => {
      this.#permit(profile, input.campaignId, ['personalization', 'telemetry']);
      return this.#appendActivity(profile, input, now, 'synthetic-operator');
    });
  }

  saveSemanticSnapshot(profileId, campaignId, ingredients, bundle) {
    identifier(profileId);
    identifier(campaignId);
    validateSemanticBundle(bundle, ingredients);
    return this.#transaction(() => {
      const profile = this.#profile(profileId);
      const now = this.#now();
      const current = this.#ingredients(profile, campaignId, ingredients.asOf);
      if (ingredients.asOf > now || canonical(current) !== canonical(ingredients)
          || bundle.recentIntent.some(event => event.expiresAt <= now)) {
        fail('STALE_SNAPSHOT', 'User state changed during encoding; encode the current ingredients again');
      }
      const result = { format: 'angelos.snapshot.v2', id: randomUUID(), profileId, campaignId,
        facts: bundle.facts, factRevision: profile.factRevision, eventRevision: ingredients.eventRevision,
        factSchemaVersion: UPL_FACT_SCHEMA, encoder: bundle.encoder, upl: bundle.upl, vector: bundle.upl,
        consentVersion: profile.consentVersion, consentSchemaVersion: CONSENT_SCHEMA_VERSION,
        createdAt: now, bundle };
      result.hash = createHash('sha256').update(canonical(result)).digest('hex');
      this.#database.prepare('INSERT INTO snapshots VALUES (?, ?, ?, ?)').run(result.id, profileId, campaignId, canonical(result));
      return freeze(result);
    });
  }

  recordImpression(input) {
    object(input, ['eventId', 'profileId', 'campaignId', 'snapshotId', 'artifact', 'context', 'exposure', 'occurredAt'],
      ['eventId', 'profileId', 'campaignId', 'snapshotId', 'artifact', 'context', 'exposure']);
    for (const key of ['eventId', 'profileId', 'campaignId', 'snapshotId']) identifier(input[key]);
    artifact(input.artifact);
    object(input.context, ['creativeId', 'placementId', 'actionSetId', 'policyVersion', 'propensityVersion', 'propensity']);
    for (const key of ['creativeId', 'placementId', 'actionSetId', 'policyVersion', 'propensityVersion']) {
      identifier(input.context[key]);
    }
    unit(input.context.propensity);
    if (input.context.propensity === 0) fail('INVALID_INPUT', 'Propensity must be positive');
    object(input.exposure, ['visibleFraction', 'visibleMilliseconds', 'documentVisible']);
    unit(input.exposure.visibleFraction);
    integer(input.exposure.visibleMilliseconds, 0, DAY);
    boolean(input.exposure.documentVisible);
    if (input.exposure.visibleFraction < 0.5 || input.exposure.visibleMilliseconds < 1000
        || !input.exposure.documentVisible) fail('NOT_EXPOSED', 'Exposure threshold not met');
    if (Object.hasOwn(input, 'occurredAt')) integer(input.occurredAt);
    return this.#event(input.profileId, 'impression', input.eventId, input, (profile, now) => {
      this.#permit(profile, input.campaignId, ['personalization', 'telemetry']);
      const snapshot = this.#snapshot(input.snapshotId, profile, input.campaignId, true);
      if (canonical(snapshot.encoder) !== canonical(input.artifact.encoder)) fail('INCOMPATIBLE_ARTIFACT', 'Snapshot and artifact encoder contracts differ');
      this.#artifact(input.artifact, input.campaignId);
      const occurredAt = this.#occurred(input, now);
      const clickDeadline = occurredAt + this.#clickWindowMs;
      const maturesAt = clickDeadline + this.#allowedLatenessMs;
      if (occurredAt < snapshot.createdAt || now >= maturesAt) {
        fail('INVALID_ATTRIBUTION', 'Exposure predates snapshot or arrived after its attribution deadline');
      }
      const result = { id: input.eventId, profileId: profile.id, campaignId: input.campaignId,
        snapshotId: snapshot.id, snapshot, artifact: input.artifact, modelVersion: input.artifact.id,
        encoderVersion: snapshot.encoder.id, consentVersion: profile.consentVersion,
        context: input.context, exposure: input.exposure, occurredAt, receivedAt: now,
        clickDeadline, maturesAt, trainingEligibleAtExposure: profile.consent.training,
        labelPolicy: { id: 'exposure-click-v1', clickWindowMs: this.#clickWindowMs,
          allowedLatenessMs: this.#allowedLatenessMs } };
      this.#database.prepare(`INSERT INTO impressions
        (id, profile_id, campaign_id, snapshot_id, artifact_id, occurred_at, received_at,
         click_deadline, matures_at, training_eligible, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
        .run(result.id, profile.id, input.campaignId, snapshot.id, input.artifact.id, occurredAt,
          now, clickDeadline, maturesAt, Number(profile.consent.training), canonical(result));
      return result;
    });
  }

  recordClick(input) {
    object(input, ['eventId', 'campaignId', 'impressionId', 'occurredAt'], ['eventId', 'campaignId', 'impressionId']);
    for (const key of ['eventId', 'campaignId', 'impressionId']) identifier(input[key]);
    if (Object.hasOwn(input, 'occurredAt')) integer(input.occurredAt);
    const impression = this.#database.prepare('SELECT * FROM impressions WHERE id = ? AND campaign_id = ?')
      .get(input.impressionId, input.campaignId);
    if (!impression) fail('INVALID_ATTRIBUTION', 'Impression not found in campaign');
    return this.#event(impression.profile_id, 'click', input.eventId, input, (profile, now) => {
      this.#permit(profile, input.campaignId, ['telemetry']);
      const occurredAt = this.#occurred(input, now);
      if (occurredAt < impression.occurred_at || occurredAt >= impression.click_deadline
          || now >= impression.matures_at) fail('INVALID_ATTRIBUTION', 'Click outside attribution window');
      const pinned = JSON.parse(impression.payload);
      const result = { id: input.eventId, profileId: profile.id, campaignId: input.campaignId,
        impressionId: input.impressionId, snapshotId: pinned.snapshotId, modelVersion: pinned.modelVersion,
        encoderVersion: pinned.encoderVersion, artifact: pinned.artifact, context: pinned.context,
        exposureConsentVersion: pinned.consentVersion, collectionConsentVersion: profile.consentVersion,
        occurredAt, receivedAt: now };
      const intentCapacity = this.#database.prepare('SELECT COUNT(*) AS count FROM upl_events WHERE profile_id = ?').get(profile.id).count < MAX_ROWS;
      result.intentUpdate = !profile.consent.personalization ? 'not-permitted'
        : occurredAt <= now - 7 * DAY ? 'outside-recent-window' : !intentCapacity ? 'capacity-limit' : 'recorded';
      if (result.intentUpdate === 'recorded') {
        this.#appendActivity(profile, { eventId: randomUUID(), campaignId: input.campaignId,
          kind: 'click', topic: 'Ad engagement', occurredAt, expiresAt: occurredAt + 7 * DAY }, now, 'exposure-click');
      }
      this.#database.prepare('INSERT INTO clicks VALUES (?, ?, ?, ?, ?, ?, ?)')
        .run(result.id, profile.id, input.campaignId, impression.id, occurredAt, now, canonical(result));
      return result;
    });
  }

  #query(input) {
    object(input, ['campaignId', 'asOf'], ['campaignId']);
    identifier(input.campaignId);
    const now = this.#now();
    const asOf = Object.hasOwn(input, 'asOf') ? input.asOf : now;
    integer(asOf, 0, now);
    return asOf;
  }

  #rows(campaignId, asOf) {
    const rows = this.#database.prepare(`SELECT impressions.*, profiles.consent AS current_consent,
      (SELECT COUNT(*) FROM clicks WHERE clicks.impression_id = impressions.id AND clicks.received_at <= ?) AS click_count
      FROM impressions JOIN profiles ON profiles.id = impressions.profile_id
      WHERE impressions.campaign_id = ? AND impressions.received_at <= ?
      ORDER BY impressions.sequence LIMIT ?`).all(asOf, campaignId, asOf, MAX_ROWS + 1);
    if (rows.length > MAX_ROWS) fail('LIMIT_EXCEEDED', 'Demo ledger limit exceeded');
    return rows.filter(row => {
      const current = JSON.parse(row.current_consent);
      return current.telemetry && current.campaignIds.includes(campaignId);
    });
  }

  maturedRows(input) {
    const asOf = this.#query(input);
    return this.#transaction(() => this.#rows(input.campaignId, asOf)
      .filter(row => row.matures_at <= asOf && row.training_eligible && JSON.parse(row.current_consent).training)
      .map(row => {
        const pinned = JSON.parse(row.payload);
        return { impressionId: row.id, sequence: row.sequence, profileId: row.profile_id,
          campaignId: row.campaign_id, snapshotId: pinned.snapshotId, upl: pinned.snapshot.upl,
          encoder: pinned.snapshot.encoder, artifact: pinned.artifact, context: pinned.context,
          consentVersion: pinned.consentVersion, labelPolicy: pinned.labelPolicy,
          occurredAt: row.occurred_at, maturesAt: row.matures_at, label: Number(row.click_count > 0),
          ...(pinned.snapshot.bundle ? { uplBundle: pinned.snapshot.bundle } : {}) };
      }));
  }

  trainingReviews(input) {
    const asOf = this.#query(input);
    return this.#transaction(() => {
      const rows = this.#database.prepare(`SELECT reviews.payload, reviews.profile_id, snapshots.payload AS snapshot,
        profiles.consent FROM reviews JOIN profiles ON profiles.id = reviews.profile_id
        JOIN snapshots ON snapshots.id = reviews.snapshot_id
        WHERE reviews.campaign_id = ? AND reviews.received_at <= ? AND reviews.training_eligible = 1
        ORDER BY reviews.received_at, reviews.id LIMIT ?`).all(input.campaignId, asOf, MAX_ROWS + 1);
      if (rows.length > MAX_ROWS) fail('LIMIT_EXCEEDED', 'Demo review limit exceeded');
      return rows.filter(row => {
        const current = JSON.parse(row.consent);
        return current.telemetry && current.training && current.campaignIds.includes(input.campaignId);
      }).map(row => {
        const review = JSON.parse(row.payload);
        const snapshot = JSON.parse(row.snapshot);
        return { ...review, upl: snapshot.upl, encoder: snapshot.encoder,
          ...(snapshot.bundle ? { uplBundle: snapshot.bundle } : {}) };
      });
    });
  }

  learningFeedback(input) {
    const asOf = this.#query(input);
    return {
      format: 'angelos.feedback.v1', campaignId: input.campaignId, asOf,
      outcome: 'click', population: 'local-synthetic-adults',
      clicks: this.maturedRows({ campaignId: input.campaignId, asOf }),
      reviews: this.trainingReviews({ campaignId: input.campaignId, asOf }),
      limitation: 'A consent-filtered point-in-time export, not perpetual permission or randomized conversion evidence. Re-export before training; exported files cannot observe later revocation.',
    };
  }

  metrics(input) {
    const asOf = this.#query(input);
    return this.#transaction(() => {
      const rows = this.#rows(input.campaignId, asOf);
      const mature = rows.filter(row => row.matures_at <= asOf);
      const clicked = rows.filter(row => row.click_count > 0).length;
      const matureClicked = mature.filter(row => row.click_count > 0).length;
      const reviewRows = this.#database.prepare(`SELECT reviews.payload, profiles.consent FROM reviews
        JOIN profiles ON profiles.id = reviews.profile_id WHERE reviews.campaign_id = ? AND reviews.received_at <= ?`)
        .all(input.campaignId, asOf);
      const reviewCounts = { operator: 0, synthetic: 0 };
      for (const row of reviewRows) {
        const current = JSON.parse(row.consent);
        if (current.telemetry && current.campaignIds.includes(input.campaignId)) {
          reviewCounts[JSON.parse(row.payload).source] += 1;
        }
      }
      return { campaignId: input.campaignId, asOf, exposures: rows.length,
        clickEvents: rows.reduce((total, row) => total + row.click_count, 0), clickedExposures: clicked,
        maturedExposures: mature.length, maturedClicks: matureClicked, pendingExposures: rows.length - mature.length,
        ctr: rows.length ? clicked / rows.length : null, maturedCtr: mature.length ? matureClicked / mature.length : null,
        trainingRows: mature.filter(row => row.training_eligible && JSON.parse(row.current_consent).training).length,
        reviews: reviewCounts };
    });
  }

  recordReview(input) {
    object(input, ['eventId', 'profileId', 'campaignId', 'snapshotId', 'artifact', 'creativeId', 'qualityRating', 'brandPass', 'source', 'brandRating'],
      ['eventId', 'profileId', 'campaignId', 'snapshotId', 'artifact', 'creativeId', 'qualityRating', 'brandPass', 'source']);
    for (const key of ['eventId', 'profileId', 'campaignId', 'snapshotId', 'creativeId']) identifier(input[key]);
    artifact(input.artifact);
    integer(input.qualityRating, 1, 5);
    if (Object.hasOwn(input, 'brandRating')) integer(input.brandRating, 1, 5);
    boolean(input.brandPass);
    if (!['operator', 'synthetic'].includes(input.source)) fail('INVALID_INPUT', 'Unknown review source');
    return this.#event(input.profileId, 'review', input.eventId, input, (profile, now) => {
      this.#permit(profile, input.campaignId, ['telemetry']);
      const snapshot = this.#snapshot(input.snapshotId, profile, input.campaignId, false);
      if (canonical(snapshot.encoder) !== canonical(input.artifact.encoder)) fail('INCOMPATIBLE_ARTIFACT', 'Snapshot and artifact encoder contracts differ');
      this.#artifact(input.artifact, input.campaignId);
      const result = { id: input.eventId, profileId: profile.id, campaignId: input.campaignId,
        snapshotId: snapshot.id, artifact: input.artifact, creativeId: input.creativeId,
        qualityRating: input.qualityRating, brandPass: input.brandPass, source: input.source,
        consentVersion: profile.consentVersion, receivedAt: now };
      if (Object.hasOwn(input, 'brandRating')) result.brandRating = input.brandRating;
      this.#database.prepare('INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?, ?)')
        .run(result.id, profile.id, input.campaignId, snapshot.id, input.artifact.id, now,
          Number(profile.consent.training && snapshot.consentVersion === profile.consentVersion), canonical(result));
      return result;
    });
  }

  deleteProfile(profileId) {
    identifier(profileId);
    return this.#transaction(() => {
      const previous = this.#database.prepare('SELECT deleted_at FROM deleted_profiles WHERE id = ?').get(profileId);
      if (previous) return { profileId, deleted: true, deletedAt: previous.deleted_at, alreadyDeleted: true, affectedArtifactIds: [] };
      this.#profile(profileId);
      const affectedArtifactIds = this.#database.prepare(`SELECT artifact_id FROM impressions WHERE profile_id = ?
        UNION SELECT artifact_id FROM reviews WHERE profile_id = ? ORDER BY artifact_id`)
        .all(profileId, profileId).map(row => row.artifact_id);
      const deletedAt = this.#now();
      this.#database.prepare('DELETE FROM profiles WHERE id = ?').run(profileId);
      this.#database.prepare('INSERT INTO deleted_profiles VALUES (?, ?)').run(profileId, deletedAt);
      return { profileId, deleted: true, deletedAt, alreadyDeleted: false, affectedArtifactIds };
    });
  }

  close() {
    this.#database.close();
  }
}
