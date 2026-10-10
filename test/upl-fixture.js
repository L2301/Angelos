import { uplHash, UPL_ENCODER_REVISION } from '../src/core/upl.js';

export function semanticFixture(ingredients, dimension = 512) {
  return { format: 'angelos.upl-input.v2', synthetic: true, adult: true, factsSchema: 'angelos.facts.v2',
    facts: structuredClone(ingredients.facts), encoder: { id: 'qwen3-embedding-0.6b-v1', dimension,
      modelId: 'Qwen/Qwen3-Embedding-0.6B', revision: UPL_ENCODER_REVISION, pooling: 'last-token',
      normalization: 'l2', templateVersion: 'angelos.upl-text.v1' },
    upl: Array.from({ length: dimension }, (_, index) => (index % 2 ? -1 : 1) / Math.sqrt(dimension)),
    context: { campaignId: ingredients.campaignId, factRevision: ingredients.factRevision,
      eventRevision: ingredients.eventRevision, consentVersion: ingredients.consentVersion,
      asOf: ingredients.asOf, ingredientsHash: uplHash(ingredients) },
    recentIntent: ingredients.events.map(({ kind, topic, occurredAt, expiresAt }) => ({ kind, topic, occurredAt, expiresAt })) };
}
