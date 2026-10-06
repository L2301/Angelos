import { createServer } from 'node:http';
import { readFile, writeFile, mkdir, access, stat, rename } from 'node:fs/promises';
import { resolve, dirname, extname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash, randomUUID } from 'node:crypto';
import { parseSource, graphHash } from './core/graph.js';
import { Store } from './core/store.js';
import { validateStudent } from './browser/student.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const maximumBody = 128 * 1024;
const mimeTypes = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.md': 'text/plain; charset=utf-8', '.json': 'application/json; charset=utf-8' };

function checkedBody(value, keys, required = keys) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Expected a JSON object');
  if (Object.keys(value).some(key => !keys.includes(key)) || required.some(key => !Object.hasOwn(value, key))) throw new Error(`Expected fields: ${keys.join(', ')}`);
  return value;
}

async function body(request) {
  if (request.headers['content-type']?.split(';')[0] !== 'application/json') throw Object.assign(new Error('Content-Type must be application/json'), { status: 415 });
  if (Number(request.headers['content-length']) > maximumBody) throw Object.assign(new Error('Request too large'), { status: 413 });
  let size = 0;
  const chunks = [];
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maximumBody) throw Object.assign(new Error('Request too large'), { status: 413 });
    chunks.push(chunk);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

function send(response, status, value) {
  response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
  response.end(JSON.stringify(value));
}

function creativeId(artifactId, snapshotId, seed) {
  const digest = createHash('sha256').update(JSON.stringify([artifactId, snapshotId, seed])).digest('hex');
  return `creative:${digest}:${seed}`;
}

export async function createLab(options = {}) {
  const dataDirectory = options.dataDirectory ?? resolve(root, 'data');
  const artifactDirectory = options.artifactDirectory ?? resolve(root, 'artifacts');
  await mkdir(dataDirectory, { recursive: true });
  const campaignPath = resolve(dataDirectory, 'campaign.ang');
  if (!await access(campaignPath).then(() => true, () => false)) await writeFile(campaignPath, await readFile(resolve(root, options.sourcePath ?? 'examples/aurora-generative.ang')));
  const store = new Store(resolve(dataDirectory, 'angelos.sqlite'));
  const seedReceipt = resolve(dataDirectory, 'seeded');
  if (!await access(seedReceipt).then(() => true, () => false)) {
    if (store.listProfiles().length === 0) {
      const consent = { personalization: true, telemetry: true, training: true, campaignIds: ['aurora'] };
      store.createProfile({ name: 'Alex · synthetic', facts: { ageBand: '25-34', region: 'NA', openness: 0.72, energy: 0.38 }, consent });
      store.createProfile({ name: 'Sam · synthetic', facts: { ageBand: '45-54', region: 'EU', openness: 0.24, energy: 0.84 }, consent });
      store.createProfile({ name: 'Jules · synthetic', facts: { ageBand: '35-44', region: 'APAC', openness: 0.5, energy: 0.6 }, consent });
    }
    await writeFile(seedReceipt, 'Synthetic profiles initialized once. Deleted profiles are not recreated.\n');
  }

  async function campaign() {
    const source = await readFile(campaignPath, 'utf8');
    const graph = parseSource(source);
    return { source, graph, graphHash: await graphHash(graph) };
  }

  async function student() {
    let pointer;
    try { pointer = JSON.parse(await readFile(resolve(artifactDirectory, 'preview.json'), 'utf8')); }
    catch (error) { if (error.code === 'ENOENT') return null; throw error; }
    checkedBody(pointer, ['file']);
    if (typeof pointer.file !== 'string' || basename(pointer.file) !== pointer.file || !/^[A-Za-z0-9_.-]+\.student\.json$/.test(pointer.file)) throw new Error('Invalid artifact pointer');
    const file = resolve(artifactDirectory, pointer.file);
    if ((await stat(file)).size > 2 * 1024 * 1024) throw new Error('Model package is too large for this runtime');
    const serialized = await readFile(file, 'utf8');
    const artifact = validateStudent(JSON.parse(serialized));
    return { artifact, descriptor: { id: artifact.id, campaignId: artifact.campaignId, hash: createHash('sha256').update(serialized).digest('hex'), graphHash: artifact.graphHash, encoder: artifact.encoder } };
  }

  async function compatibleArtifact(artifactId, claimedHash) {
    const current = await campaign();
    const loaded = await student();
    if (!loaded || loaded.artifact.id !== artifactId || loaded.artifact.campaignId !== current.graph.id || loaded.artifact.graphHash !== current.graphHash || (claimedHash && claimedHash !== current.graphHash)) throw Object.assign(new Error('Artifact does not match the current campaign'), { status: 409 });
    return loaded.descriptor;
  }

  function metrics(campaignId) {
    const result = store.metrics({ campaignId });
    return { ...result, impressions: result.exposures, clicks: result.clickedExposures, reviewSources: result.reviews, reviews: result.reviews.operator + result.reviews.synthetic };
  }

  const server = createServer(async (request, response) => {
    response.setHeader('Cache-Control', 'no-store');
    response.setHeader('X-Content-Type-Options', 'nosniff');
    response.setHeader('Referrer-Policy', 'no-referrer');
    response.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");
    try {
      const port = server.address()?.port;
      const hosts = [`127.0.0.1:${port}`, `localhost:${port}`];
      if (!hosts.includes(request.headers.host) || (request.headers.origin && !hosts.map(host => `http://${host}`).includes(request.headers.origin)) || request.headers['sec-fetch-site'] === 'cross-site') throw Object.assign(new Error('Loopback same-origin requests only'), { status: 403 });
      const url = new URL(request.url, `http://${request.headers.host}`);
      if (request.method === 'GET') {
        if (url.pathname === '/api/health') return send(response, 200, { ok: true, mode: 'local-synthetic-laboratory' });
        if (url.pathname === '/api/examples') {
          const examples = [['tiny-sd', 'Tiny-SD image teacher', 'aurora-generative.ang'], ['flux2-klein-4b', 'FLUX.2 Klein 4B teacher', 'aurora-flux.ang'], ['ltx-video', 'LTX video → frame teacher', 'aurora-ltx.ang'], ['procedural', 'Deterministic regression fixture', 'aurora.ang']];
          return send(response, 200, await Promise.all(examples.map(async ([id, label, file]) => ({ id, label, source: await readFile(resolve(root, 'examples', file), 'utf8') }))));
        }
        if (url.pathname === '/api/state') {
          const current = await campaign();
          const loaded = await student();
          return send(response, 200, { ...current, profiles: store.listProfiles(), artifactAvailable: Boolean(loaded), artifactCompatible: Boolean(loaded && loaded.artifact.graphHash === current.graphHash && loaded.artifact.campaignId === current.graph.id), metrics: metrics(current.graph.id) });
        }
        if (url.pathname === '/api/profiles') return send(response, 200, store.listProfiles());
        if (url.pathname === '/api/metrics') return send(response, 200, metrics((await campaign()).graph.id));
        if (url.pathname === '/api/training-rows') return send(response, 200, { rows: store.maturedRows({ campaignId: (await campaign()).graph.id }), note: 'Exposure-qualified mature synthetic examples; no trainer or promotion is scheduled.' });
        if (url.pathname === '/api/artifact') {
          const loaded = await student();
          if (!loaded) throw Object.assign(new Error('No trained preview artifact; run npm run train'), { status: 404 });
          return send(response, 200, loaded.artifact);
        }
        const publicFiles = new Set(['/web/index.html', '/web/app.js', '/web/style.css', '/src/core/graph.js', '/src/browser/student.js', '/src/browser/diffusion.js', '/docs/PLAN.md', '/docs/CONTRACTS.md', '/docs/DIFFUSION.md', '/docs/MODELS.md']);
        const path = url.pathname === '/' ? '/web/index.html' : url.pathname;
        if (!publicFiles.has(path)) throw Object.assign(new Error('Not found'), { status: 404 });
        response.writeHead(200, { 'Content-Type': mimeTypes[extname(path)] ?? 'application/octet-stream' });
        response.end(await readFile(resolve(root, `.${path}`)));
        return;
      }
      if (request.method !== 'POST') throw Object.assign(new Error('Method not allowed'), { status: 405 });
      const payload = await body(request);
      if (url.pathname === '/api/graph') {
        checkedBody(payload, ['source']);
        const graph = parseSource(payload.source);
        const temporary = `${campaignPath}.${randomUUID()}.tmp`;
        await writeFile(temporary, payload.source);
        await rename(temporary, campaignPath);
        const hash = await graphHash(graph);
        const loaded = await student();
        return send(response, 200, { graph, source: payload.source, graphHash: hash, artifactCompatible: Boolean(loaded && loaded.artifact.graphHash === hash && loaded.artifact.campaignId === graph.id) });
      }
      if (url.pathname === '/api/facts') return send(response, 200, store.updateFact(payload));
      if (url.pathname === '/api/consent') {
        checkedBody(payload, ['profileId', 'campaignId', 'personalization', 'telemetry', 'training']);
        const profile = store.getProfile(payload.profileId);
        if (payload.campaignId !== (await campaign()).graph.id) throw new Error('Unknown campaign');
        return send(response, 200, store.setConsent({ profileId: profile.id, eventId: randomUUID(), expectedVersion: profile.consentVersion, personalization: payload.personalization, telemetry: payload.telemetry, training: payload.training, campaignIds: [payload.campaignId] }));
      }
      if (url.pathname === '/api/snapshot') {
        checkedBody(payload, ['profileId', 'campaignId']);
        return send(response, 200, store.snapshot(payload.profileId, payload.campaignId));
      }
      if (url.pathname === '/api/delete-profile') {
        checkedBody(payload, ['profileId']);
        return send(response, 200, store.deleteProfile(payload.profileId));
      }
      if (url.pathname === '/api/impressions') {
        checkedBody(payload, ['profileId', 'campaignId', 'snapshotId', 'artifactId', 'graphHash', 'eventId', 'exposure', 'seed'], ['profileId', 'campaignId', 'snapshotId', 'artifactId', 'graphHash', 'eventId', 'exposure']);
        const seed = payload.seed ?? 2026;
        if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('Creative seed must be uint32');
        const descriptor = await compatibleArtifact(payload.artifactId, payload.graphHash);
        return send(response, 200, store.recordImpression({ eventId: payload.eventId, profileId: payload.profileId, campaignId: payload.campaignId, snapshotId: payload.snapshotId, artifact: descriptor, context: { creativeId: creativeId(payload.artifactId, payload.snapshotId, seed), placementId: 'local-preview', actionSetId: 'single-candidate-v1', policyVersion: 'operator-exposure-v1', propensityVersion: 'single-candidate-v1', propensity: 1 }, exposure: payload.exposure }));
      }
      if (url.pathname === '/api/click') return send(response, 200, store.recordClick(payload));
      if (url.pathname === '/api/reviews') {
        checkedBody(payload, ['profileId', 'campaignId', 'snapshotId', 'artifactId', 'quality', 'brand', 'eventId', 'seed'], ['profileId', 'campaignId', 'snapshotId', 'artifactId', 'quality', 'brand', 'eventId']);
        const seed = payload.seed ?? 2026;
        if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('Creative seed must be uint32');
        if (!Number.isInteger(payload.brand) || payload.brand < 1 || payload.brand > 5) throw new Error('Brand rating must be 1–5');
        const descriptor = await compatibleArtifact(payload.artifactId);
        return send(response, 200, store.recordReview({ eventId: payload.eventId, profileId: payload.profileId, campaignId: payload.campaignId, snapshotId: payload.snapshotId, artifact: descriptor, creativeId: creativeId(payload.artifactId, payload.snapshotId, seed), qualityRating: payload.quality, brandRating: payload.brand, brandPass: payload.brand >= 4, source: 'operator' }));
      }
      throw Object.assign(new Error('Not found'), { status: 404 });
    } catch (error) {
      const status = error.status ?? (error.code === 'CONFLICT' || error.code === 'STALE_SNAPSHOT' ? 409 : error.code === 'CONSENT_DENIED' ? 403 : 400);
      if (!response.headersSent) send(response, status, { error: error.message });
      else response.end();
    }
  });
  server.on('close', () => store.close());
  return { server, store };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const port = Number(process.env.PORT ?? 4310);
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('Invalid PORT');
  const { server } = await createLab();
  server.listen(port, '127.0.0.1', () => process.stdout.write(`Angelos laboratory: http://127.0.0.1:${port}\nSynthetic profiles only. Ctrl+C stops the server.\n`));
}
