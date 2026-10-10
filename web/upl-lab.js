const fieldNames = ['interests', 'goals', 'preferredStyles', 'preferredLanguage'];
const element = identifier => document.getElementById(identifier);

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  return JSON.stringify(value);
}

function exactKeys(value, fields) {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value)
    && Object.keys(value).length === fields.length && fields.every(field => Object.hasOwn(value, field)));
}

async function sha256Text(text) {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))), value => value.toString(16).padStart(2, '0')).join('');
}

export async function validateEncodedUpl(result, current, context, dimension) {
  const snapshot = result?.snapshot;
  const bundle = result?.bundle;
  const same = (left, right) => canonicalJson(left) === canonicalJson(right);
  const fail = () => { throw new Error('Encode response is not a current, bound v2 snapshot. No bundle will be copied.'); };
  const encoder = { id: 'qwen3-embedding-0.6b-v1', dimension, modelId: 'Qwen/Qwen3-Embedding-0.6B',
    revision: '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3', pooling: 'last-token', normalization: 'l2', templateVersion: 'angelos.upl-text.v1' };
  if (!snapshot || snapshot.format !== 'angelos.snapshot.v2' || snapshot.profileId !== context.profile.id
    || snapshot.campaignId !== context.campaignId || !/^[a-f0-9]{64}$/.test(snapshot.hash)
    || !exactKeys(bundle, ['format', 'synthetic', 'adult', 'factsSchema', 'facts', 'encoder', 'upl', 'context', 'recentIntent'])
    || bundle.format !== 'angelos.upl-input.v2' || bundle.synthetic !== true || bundle.adult !== true
    || bundle.factsSchema !== 'angelos.facts.v2' || ![128, 256, 512, 1024].includes(dimension) || !same(bundle.encoder, encoder)
    || !Array.isArray(bundle.upl) || bundle.upl.length !== dimension
    || bundle.upl.some(value => typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > 1)
    || Math.abs(Math.hypot(...bundle.upl) - 1) > 0.0001
    || !same(snapshot.bundle, bundle) || !same(bundle.facts, current.facts)
    || !same(snapshot.facts, bundle.facts) || !same(snapshot.encoder, encoder)
    || !same(snapshot.upl, bundle.upl) || !same(snapshot.vector, bundle.upl)
    || snapshot.factSchemaVersion !== bundle.factsSchema || snapshot.consentSchemaVersion !== 'angelos.consent.v1'
    || !exactKeys(bundle.context, ['campaignId', 'factRevision', 'eventRevision', 'consentVersion', 'asOf', 'ingredientsHash'])
    || bundle.context.campaignId !== context.campaignId || !/^[a-f0-9]{64}$/.test(bundle.context.ingredientsHash)
    || !Number.isSafeInteger(bundle.context.asOf) || bundle.context.asOf < 0 || bundle.context.asOf > current.asOf
    || !Number.isSafeInteger(snapshot.createdAt) || snapshot.createdAt < bundle.context.asOf || snapshot.createdAt > current.asOf
    || ['factRevision', 'eventRevision', 'consentVersion'].some(field => snapshot[field] !== current[field] || bundle.context[field] !== current[field])
    || !Array.isArray(bundle.recentIntent) || bundle.recentIntent.length > 64
    || bundle.recentIntent.some(event => !exactKeys(event, ['kind', 'topic', 'occurredAt', 'expiresAt'])
      || !['view', 'click', 'save', 'dismiss'].includes(event.kind)
      || !Number.isSafeInteger(event.occurredAt) || event.occurredAt < 0 || event.occurredAt > bundle.context.asOf
      || !Number.isSafeInteger(event.expiresAt) || event.expiresAt <= current.asOf)) fail();
  const recentIntent = current.events.map(({ kind, topic, occurredAt, expiresAt }) => ({ kind, topic, occurredAt, expiresAt }));
  if (!same(bundle.recentIntent, recentIntent)) fail();
  const frozenIngredients = { ...current, asOf: bundle.context.asOf };
  const unsignedSnapshot = Object.fromEntries(Object.entries(snapshot).filter(([key]) => key !== 'hash'));
  if (bundle.context.ingredientsHash !== await sha256Text(canonicalJson(frozenIngredients))
    || snapshot.hash !== await sha256Text(canonicalJson(unsignedSnapshot))) fail();
}

function ingredientIdentity(ingredients) {
  return JSON.stringify([ingredients.campaignId, ingredients.factRevision, ingredients.eventRevision,
    ingredients.consentVersion, ingredients.facts, ingredients.provenance, ingredients.events]);
}

async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? { cache: 'no-store' }
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const result = await response.json();
  if (!response.ok) throw new Error(response.status === 503
    ? `Local embedding unavailable (503). ${result.error ?? 'The pinned Qwen3-Embedding checkpoint may not be cached.'} No fallback vector, automatic download or paid job was started.`
    : result.error ?? `UPL request failed (${response.status}).`);
  return result;
}

export function initializeUplLab({ getContext, onFactsSaved }) {
  let contextKey;
  let generation = 0;
  let ingredients;
  let encoded;
  let busy = false;
  let dirty = false;
  let expiryTimer;

  function permitted() {
    const { profile, campaignId } = getContext();
    return Boolean(profile?.synthetic && profile.consent?.personalization
      && profile.consent.campaignIds.includes(campaignId) && element('allow-personalization').checked
      && ['personalization', 'telemetry', 'training'].every(scope => element(`allow-${scope}`).checked === Boolean(profile.consent[scope])));
  }

  function controls() {
    const available = permitted() && Boolean(ingredients) && !busy;
    element('upl-fact-fields').disabled = !available;
    element('upl-event-fields').disabled = !available || dirty || !getContext().profile?.consent?.telemetry || !element('allow-telemetry').checked;
    element('upl-encode').disabled = !available || dirty || getContext().legacyDirty;
    element('upl-refresh').disabled = busy || !permitted();
    element('upl-copy-v2').disabled = !available || !encoded || dirty || getContext().legacyDirty;
    element('upl-dimension').disabled = busy;
    element('upl-laboratory').setAttribute('aria-busy', String(busy));
  }

  function invalidate(message) {
    generation += 1;
    busy = false;
    encoded = undefined;
    clearTimeout(expiryTimer);
    element('upl-v2-json').value = '';
    element('upl-v2-fallback').hidden = true;
    element('upl-encoded-dimension').textContent = 'Not encoded';
    element('upl-bundle-hash').textContent = 'No current bundle';
    element('upl-snapshot-hash').textContent = 'No stored v2 snapshot';
    element('upl-encoder-info').textContent = 'No semantic inference has succeeded for this selection.';
    element('upl-public-bundle').textContent = 'No current encoded bundle.';
    element('upl-v2-status').textContent = message;
    controls();
  }

  function renderIngredients() {
    const { profile, campaignId } = getContext();
    element('upl-scope').textContent = `${profile.name} · campaign ${campaignId} · synthetic adult`;
    element('upl-fact-revision').textContent = String(ingredients.factRevision);
    element('upl-event-revision').textContent = String(ingredients.eventRevision);
    element('upl-consent-version').textContent = String(ingredients.consentVersion);
    element('upl-ingredients-json').textContent = JSON.stringify(ingredients, null, 2);
    for (const field of fieldNames) {
      const value = ingredients.facts[field];
      element(`upl-${field}`).value = Array.isArray(value) ? value.join(', ') : value ?? '';
    }
    const events = element('upl-events');
    events.replaceChildren();
    for (const event of ingredients.events) {
      const item = document.createElement('li');
      const title = document.createElement('strong');
      title.textContent = `${event.kind} · ${event.topic}`;
      const detail = document.createElement('span');
      detail.textContent = `Occurred ${new Date(event.occurredAt).toISOString()} · received ${new Date(event.receivedAt).toISOString()} · expires ${new Date(event.expiresAt).toISOString()} · source ${event.source} · campaign ${event.campaignId}`;
      item.append(title, detail);
      events.append(item);
    }
    if (!ingredients.events.length) {
      const item = document.createElement('li');
      item.textContent = 'No permitted recent intent events. Missing observations are not negative feedback.';
      events.append(item);
    }
  }

  async function readIngredients(context) {
    const query = new URLSearchParams({ profileId: context.profile.id, campaignId: context.campaignId });
    const result = await request(`/api/upl/ingredients?${query}`);
    if (result.format !== 'angelos.upl-ingredients.v2' || result.synthetic !== true || result.adult !== true
      || result.factsSchema !== 'angelos.facts.v2' || result.eventsSchema !== 'angelos.events.v1'
      || result.campaignId !== context.campaignId || !result.facts || !Array.isArray(result.events)
      || ![result.factRevision, result.eventRevision, result.consentVersion, result.asOf].every(Number.isSafeInteger)
      || result.events.length > 64 || result.events.some(event => ![event.occurredAt, event.receivedAt, event.expiresAt].every(Number.isSafeInteger))) {
      throw new Error('Ingredients contract mismatch. No encoding or handoff is available.');
    }
    return result;
  }

  async function refresh() {
    invalidate('Loading permitted ingredients. No encoder is running.');
    const token = generation;
    ingredients = undefined;
    busy = true;
    controls();
    try {
      const next = await readIngredients(getContext());
      if (token !== generation) return;
      ingredients = next;
      dirty = false;
      renderIngredients();
      element('upl-v2-status').textContent = 'Ingredients ready. Encode explicitly with cached Qwen3-Embedding weights; editing facts is not weight training.';
    } catch (error) {
      if (token === generation) element('upl-v2-status').textContent = error.message;
    } finally {
      if (token === generation) { busy = false; controls(); }
    }
  }

  function sync() {
    const context = getContext();
    const key = JSON.stringify([context.profile?.id, context.campaignId, context.profile?.factRevision,
      context.profile?.eventRevision, context.profile?.consentVersion, permitted(), context.profile?.consent?.telemetry]);
    if (key === contextKey) { controls(); return; }
    contextKey = key;
    ingredients = undefined;
    dirty = false;
    invalidate('Choose a permitted synthetic adult to inspect UPL ingredients.');
    element('upl-scope').textContent = 'No authorized ingredients loaded';
    element('upl-ingredients-json').textContent = '';
    element('upl-events').replaceChildren();
    for (const field of fieldNames) element(`upl-${field}`).value = '';
    for (const name of ['fact-revision', 'event-revision', 'consent-version']) element(`upl-${name}`).textContent = '—';
    if (permitted()) void refresh();
  }

  async function mutate(action, callback) {
    if (!permitted() || busy || !ingredients) return;
    const context = getContext();
    invalidate(action);
    const token = generation;
    busy = true;
    controls();
    try {
      await callback(context, token);
      if (token !== generation) return;
      await refresh();
    } catch (error) {
      if (token === generation) {
        element('upl-v2-status').textContent = `${error.message} Refresh ingredients before retrying if any earlier fields were saved.`;
        busy = false;
        controls();
      }
    }
  }

  element('upl-facts-form').addEventListener('input', () => {
    dirty = true;
    invalidate('Unsaved audience facts. Save them, then explicitly encode again; previous bundles are no longer current.');
  });
  element('upl-facts-form').addEventListener('submit', event => {
    event.preventDefault();
    const changes = [];
    try {
      for (const field of fieldNames) {
        const text = element(`upl-${field}`).value.trim();
        const value = field === 'preferredLanguage' ? text || null : text.split(',').map(item => item.trim()).filter(Boolean);
        if (Array.isArray(value) && (value.length > 16 || value.some(item => item.length > 80) || new Set(value).size !== value.length)) {
          throw new Error('Use at most 16 unique comma-separated items of at most 80 characters per field.');
        }
        if (JSON.stringify(value) !== JSON.stringify(ingredients?.facts[field] ?? (Array.isArray(value) ? [] : null))) changes.push([field, value]);
      }
    } catch (error) { element('upl-v2-status').textContent = error.message; return; }
    void mutate('Saving declared audience facts…', async (context, token) => {
      for (const [field, value] of changes) {
        if (token !== generation) return;
        await request('/api/upl/fact', { profileId: context.profile.id, campaignId: context.campaignId, field, value });
      }
      if (token === generation && changes.length) await onFactsSaved();
    });
  });
  element('upl-event-form').addEventListener('submit', event => {
    event.preventDefault();
    if (!getContext().profile?.consent?.telemetry || !element('allow-telemetry').checked) return;
    const kind = element('upl-event-kind').value;
    const topic = element('upl-event-topic').value.trim();
    if (!topic) return;
    void mutate('Recording one explicit synthetic intent event…', async context => {
      await request('/api/upl/event', { profileId: context.profile.id, campaignId: context.campaignId, kind, topic });
    });
  });
  element('upl-refresh').addEventListener('click', () => { if (permitted() && !busy) void refresh(); });
  element('upl-dimension').addEventListener('change', () => invalidate('Output width changed. Encode again; no vector is padded, truncated or invented by this page.'));
  element('upl-encode').addEventListener('click', async () => {
    if (!permitted() || busy || dirty || !ingredients || getContext().legacyDirty) return;
    const context = getContext();
    const captured = ingredientIdentity(ingredients);
    const dimension = Number(element('upl-dimension').value);
    invalidate('Running explicit local embedding inference. The first cold load may take about 20 seconds; timing varies by machine, not a GPU latency claim. No automatic download or training.');
    const token = generation;
    busy = true;
    controls();
    try {
      const result = await request('/api/upl/encode', { profileId: context.profile.id, campaignId: context.campaignId, dimension });
      if (token !== generation) return;
      const current = await readIngredients(context);
      if (token !== generation) return;
      if (captured !== ingredientIdentity(current)) throw new Error('Ingredients changed during encoding. Refresh and encode the current revisions again.');
      const snapshot = result.snapshot;
      const bundle = result.bundle;
      await validateEncodedUpl(result, current, context, dimension);
      const json = JSON.stringify(result.bundle, null, 2);
      const hash = await sha256Text(json);
      if (token !== generation) return;
      const expiries = current.events.map(event => event.expiresAt);
      const expires = expiries.length ? Math.min(...expiries) : Infinity;
      if (expires <= Date.now()) throw new Error('Recent event expiry changed. Refresh ingredients before encoding again.');
      encoded = { json, identity: captured, expires };
      element('upl-public-bundle').textContent = json;
      element('upl-encoded-dimension').textContent = `${dimension} semantic dimensions`;
      element('upl-bundle-hash').textContent = hash;
      element('upl-snapshot-hash').textContent = snapshot.hash;
      element('upl-encoder-info').textContent = `${bundle.encoder.id} · revision ${bundle.encoder.revision}. Exact public bundle preserved; the copy hash identifies JSON bytes, not a person.`;
      element('upl-v2-status').textContent = 'Current v2 encode ready. Copying is explicit; embeddings are not anonymous and need a compatible, evaluated adapter.';
      if (Number.isFinite(expires)) expiryTimer = setTimeout(() => invalidate('Event freshness expired. Refresh ingredients and encode again.'), Math.min(expires - Date.now(), 2147483647));
    } catch (error) {
      if (token === generation) element('upl-v2-status').textContent = error.message;
    } finally {
      if (token === generation) { busy = false; controls(); }
    }
  });
  element('upl-copy-v2').addEventListener('click', async () => {
    if (!encoded || busy || dirty || !permitted() || getContext().legacyDirty) return;
    const captured = encoded;
    const token = generation;
    busy = true;
    controls();
    try {
      const current = await readIngredients(getContext());
      if (token !== generation) return;
      if (captured.identity !== ingredientIdentity(current) || Date.now() >= captured.expires) {
        invalidate('Ingredients, consent or event freshness changed. Refresh and encode again before copying.');
        return;
      }
      try {
        await navigator.clipboard.writeText(captured.json);
        if (token === generation) element('upl-v2-status').textContent = 'Current public v2 bundle copied for ComfyUI. No internal snapshot wrapper was copied.';
      } catch {
        if (token !== generation) return;
        element('upl-v2-json').value = captured.json;
        element('upl-v2-fallback').hidden = false;
        element('upl-v2-json').focus();
        element('upl-v2-json').select();
        element('upl-v2-status').textContent = 'Clipboard unavailable. Select and manually copy this current public v2 bundle.';
      }
    } catch (error) {
      if (token === generation) invalidate(`${error.message} Copy disabled until a fresh encode succeeds.`);
    } finally {
      if (token === generation) { busy = false; controls(); }
    }
  });
  for (const name of ['age-band', 'region', 'openness', 'energy']) element(name).addEventListener('input', () => invalidate('Legacy facts changed. Save those preferences and encode v2 again.'));
  for (const name of ['allow-personalization', 'allow-telemetry', 'allow-training']) element(name).addEventListener('change', () => {
    invalidate('Permissions changed. Save consent, refresh ingredients and encode again.');
    sync();
  });
  return { sync };
}
