import { createHash } from 'node:crypto';

export const UPL_DIMENSIONS = Object.freeze([128, 256, 512, 1024]);
export const UPL_ENCODER_REVISION = '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3';
export const UPL_FACT_SCHEMA = 'angelos.facts.v2';
export const UPL_EVENT_SCHEMA = 'angelos.events.v1';
export const UPL_FIELDS = Object.freeze(['ageBand', 'region', 'openness', 'energy', 'interests', 'goals', 'preferredStyles', 'preferredLanguage']);
export const ADULT_AGE_BANDS = Object.freeze(['18-24', '25-34', '35-44', '45-54', '55-64', '65+']);
export const UPL_EVENT_KINDS = Object.freeze(['view', 'click', 'save', 'dismiss']);

function invalid(message) {
  throw Object.assign(new Error(message), { code: 'INVALID_INPUT' });
}

export function uplObject(value, fields) {
  if (!value || typeof value !== 'object' || Array.isArray(value)
      || ![Object.prototype, null].includes(Object.getPrototypeOf(value))
      || Reflect.ownKeys(value).length !== fields.length
      || fields.some(field => !Object.hasOwn(value, field)
        || !Object.hasOwn(Object.getOwnPropertyDescriptor(value, field), 'value'))) invalid('Unexpected UPL object fields');
}

export function uplText(value, maximum = 80) {
  if (typeof value !== 'string' || !value.trim() || value !== value.trim() || value.length > maximum
      || /[\u0000-\u001f\u007f]/.test(value) || !value.isWellFormed()) invalid(`Expected nonempty UPL text, at most ${maximum} characters`);
  return value;
}

export function validateUplFact(field, value) {
  if (!UPL_FIELDS.includes(field)) invalid('Unknown UPL fact field');
  if (value === null) return;
  if (field === 'ageBand' && !ADULT_AGE_BANDS.includes(value)) invalid('Expected an adult age band');
  if (field === 'region' && !['NA', 'EU', 'APAC', 'OTHER'].includes(value)) invalid('Expected a coarse region');
  if (['openness', 'energy'].includes(field)
      && (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1)) invalid('Expected a fact in [0, 1]');
  if (['interests', 'goals', 'preferredStyles'].includes(field)) {
    if (!Array.isArray(value) || value.length > 16 || new Set(value).size !== value.length) invalid('Expected at most 16 unique fact values');
    value.forEach(entry => uplText(entry));
  }
  if (field === 'preferredLanguage') uplText(value);
}

export function uplCanonical(value) {
  if (Array.isArray(value)) return `[${value.map(uplCanonical).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${uplCanonical(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

export function uplHash(value) {
  return createHash('sha256').update(uplCanonical(value)).digest('hex');
}

export function validateSemanticEncoder(encoder) {
  uplObject(encoder, ['id', 'dimension', 'modelId', 'revision', 'pooling', 'normalization', 'templateVersion']);
  if (encoder.id !== 'qwen3-embedding-0.6b-v1' || encoder.modelId !== 'Qwen/Qwen3-Embedding-0.6B'
      || !UPL_DIMENSIONS.includes(encoder.dimension) || encoder.revision !== UPL_ENCODER_REVISION
      || encoder.pooling !== 'last-token' || encoder.normalization !== 'l2'
      || encoder.templateVersion !== 'angelos.upl-text.v1') invalid('Unsupported semantic encoder contract');
  return encoder;
}

export function validateSemanticBundle(bundle, ingredients) {
  uplObject(bundle, ['format', 'synthetic', 'adult', 'factsSchema', 'facts', 'encoder', 'upl', 'context', 'recentIntent']);
  if (bundle.format !== 'angelos.upl-input.v2' || bundle.synthetic !== true || bundle.adult !== true
      || bundle.factsSchema !== UPL_FACT_SCHEMA) invalid('Expected a synthetic adult UPL v2 bundle');
  if (!bundle.facts || typeof bundle.facts !== 'object' || Array.isArray(bundle.facts)) invalid('Expected UPL facts');
  for (const [field, value] of Object.entries(bundle.facts)) validateUplFact(field, value);
  if (!ADULT_AGE_BANDS.includes(bundle.facts.ageBand)) invalid('Adult age band is required');
  validateSemanticEncoder(bundle.encoder);
  if (!Array.isArray(bundle.upl) || bundle.upl.length !== bundle.encoder.dimension
      || bundle.upl.some(value => typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > 1)
      || Math.abs(Math.hypot(...bundle.upl) - 1) > 0.0001) invalid('Expected a signed, finite, L2-normalized UPL matching its dimension');
  uplObject(bundle.context, ['campaignId', 'factRevision', 'eventRevision', 'consentVersion', 'asOf', 'ingredientsHash']);
  if (!/^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/.test(bundle.context.campaignId)
      || !/^[a-f0-9]{64}$/.test(bundle.context.ingredientsHash)) invalid('Invalid UPL campaign or ingredients hash');
  for (const field of ['factRevision', 'eventRevision', 'consentVersion', 'asOf']) {
    if (!Number.isSafeInteger(bundle.context[field]) || bundle.context[field] < (['factRevision', 'consentVersion'].includes(field) ? 1 : 0)) invalid('Invalid UPL revision or timestamp');
  }
  if (!Array.isArray(bundle.recentIntent) || bundle.recentIntent.length > 64) invalid('Expected at most 64 recent events');
  for (const event of bundle.recentIntent) {
    uplObject(event, ['kind', 'topic', 'occurredAt', 'expiresAt']);
    if (!UPL_EVENT_KINDS.includes(event.kind)) invalid('Unsupported intent event');
    uplText(event.topic);
    if (!Number.isSafeInteger(event.occurredAt) || event.occurredAt < 0 || event.occurredAt > bundle.context.asOf
        || !Number.isSafeInteger(event.expiresAt) || event.expiresAt <= bundle.context.asOf) invalid('Expired or future intent event');
  }
  if (ingredients) {
    const expectedContext = Object.fromEntries(['campaignId', 'factRevision', 'eventRevision', 'consentVersion', 'asOf'].map(field => [field, ingredients[field]]));
    expectedContext.ingredientsHash = uplHash(ingredients);
    const recentIntent = ingredients.events.map(({ kind, topic, occurredAt, expiresAt }) => ({ kind, topic, occurredAt, expiresAt }));
    if (uplCanonical(bundle.context) !== uplCanonical(expectedContext)
        || uplCanonical(bundle.facts) !== uplCanonical(ingredients.facts)
        || uplCanonical(bundle.recentIntent) !== uplCanonical(recentIntent)) invalid('Semantic result is not bound to the requested ingredients');
  }
  return bundle;
}
