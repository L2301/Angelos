import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import { createLab } from '../src/server.js';

const root = fileURLToPath(new URL('../', import.meta.url));
const repository = 'Efficient-Large-Model/Sana_Sprint_0.6B_1024px_diffusers';
const revision = 'aa76e7f4f4928f378716b6716a2130fba3caf5b1';
const ledgerTables = ['profiles', 'fact_events', 'consent_events', 'snapshots', 'artifacts', 'impressions', 'clicks', 'reviews', 'receipts', 'deleted_profiles'];

async function laboratory(context) {
  const directory = await mkdtemp(join(tmpdir(), 'angelos-hd-http-'));
  const artifactDirectory = join(directory, 'artifacts');
  const dataDirectory = join(directory, 'data');
  await mkdir(artifactDirectory);
  const { server } = await createLab({
    dataDirectory, artifactDirectory, sourcePath: 'examples/aurora.ang',
    textRunner: async () => assert.fail('Reading HD evidence must not start text inference'),
  });
  context.after(async () => {
    if (server.listening) await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
    await rm(directory, { recursive: true, force: true });
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const origin = `http://127.0.0.1:${server.address().port}`;
  return { get: (path, options) => fetch(`${origin}${path}`, options), databasePath: join(dataDirectory, 'angelos.sqlite') };
}

async function recordedManifest() {
  return JSON.parse(await readFile(join(root, 'web/evidence/hd-manifest.json'), 'utf8'));
}

function ledgerSnapshot(database) {
  return Object.fromEntries(ledgerTables.map(table => [table, database.prepare(`SELECT * FROM ${table} ORDER BY rowid`).all()]));
}

test('HD preview reports native generation and crop-only delivered metadata without quality or training claims', async context => {
  const lab = await laboratory(context);
  const response = await lab.get('/api/hd-preview');
  assert.equal(response.status, 200, await response.clone().text());
  assert.match(response.headers.get('content-type'), /^application\/json/);
  assert.equal(response.headers.get('cache-control'), 'no-store');
  const preview = await response.json();
  const recorded = await recordedManifest();
  assert.deepEqual(preview, {
    available: true, image: '/api/hd-image', width: 1920, height: 1080,
    nativeWidth: 1920, nativeHeight: 1088, mode: 'pretrained-baseline-not-campaign-trained', ...recorded,
  });
  assert.equal(preview.available, true);
  assert.equal(preview.image, '/api/hd-image');
  assert.equal(preview.schema, 'angelos.hd-evidence.v1');
  assert.equal(preview.model, repository);
  assert.equal(preview.revision, revision);
  assert.equal(preview.mode, 'pretrained-baseline-not-campaign-trained');
  assert.deepEqual([preview.nativeWidth, preview.nativeHeight], [1920, 1088]);
  assert.deepEqual([preview.width, preview.height], [1920, 1080]);
  assert.deepEqual(preview.nativeSize, [1920, 1088]);
  assert.deepEqual(preview.outputSize, [1920, 1080]);
  assert.deepEqual(preview.crop, [0, 4, 1920, 1084]);
  assert.equal(preview.nativeHeight - preview.crop[3], 4);
  assert.equal(preview.crop[3] - preview.crop[1], preview.height);
  assert.equal(preview.resized, false);
  assert.equal(preview.resolutionBinning, false);
  assert.equal(preview.transformerDtype, 'bfloat16');
  assert.equal(preview.vaeDtype, 'float32');
  assert.equal(preview.diffusersVersion, '0.40.0');
  assert.ok(['mps', 'cuda', 'cpu'].includes(preview.device));
  assert.ok([2, 4].includes(preview.steps));
  assert.ok(Number.isInteger(preview.seed) && preview.seed >= 0 && preview.seed <= 0xffffffff);
  assert.ok(typeof preview.prompt === 'string' && preview.prompt.trim().length > 0);
  for (const field of ['campaignTrained', 'browserWebGPU', 'paidJobs', 'qualityGatePassed']) assert.equal(preview[field], false, field);
  for (const field of ['imageSha256', 'nativeImageSha256']) assert.match(preview[field], /^[a-f0-9]{64}$/, field);
  assert.ok(Array.isArray(preview.limitations) && preview.limitations.length > 0);
});

test('HD image HTTP and static bytes match the manifest SHA-256 and actual 1920x1080 PNG header', async context => {
  const lab = await laboratory(context);
  const preview = await (await lab.get('/api/hd-preview')).json();
  const recorded = await readFile(join(root, 'web/evidence/hd-baseline.png'));
  for (const path of [preview.image, '/web/evidence/hd-baseline.png']) {
    const response = await lab.get(path);
    assert.equal(response.status, 200, path);
    assert.equal(response.headers.get('content-type'), 'image/png');
    assert.equal(response.headers.get('cache-control'), 'no-store');
    assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
    const pixels = Buffer.from(await response.arrayBuffer());
    assert.deepEqual(pixels, recorded);
    assert.equal(createHash('sha256').update(pixels).digest('hex'), preview.imageSha256);
    assert.ok(pixels.length > 33);
    assert.deepEqual(pixels.subarray(0, 8), Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
    assert.equal(pixels.readUInt32BE(8), 13);
    assert.equal(pixels.toString('ascii', 12, 16), 'IHDR');
    assert.equal(pixels.readUInt32BE(16), 1920);
    assert.equal(pixels.readUInt32BE(20), 1080);
    assert.equal(pixels.readUInt32BE(16), preview.width);
    assert.equal(pixels.readUInt32BE(20), preview.height);
    assert.notEqual(pixels.readUInt32BE(20), preview.nativeHeight);
  }
  const manifestResponse = await lab.get('/web/evidence/hd-manifest.json');
  assert.equal(manifestResponse.status, 200);
  assert.deepEqual(await manifestResponse.json(), await recordedManifest());
});

test('HD evidence reads and cross-origin refusals leave profiles, campaign and the entire ledger unchanged', async context => {
  const lab = await laboratory(context);
  const database = new DatabaseSync(lab.databasePath, { readOnly: true });
  try {
    const before = await (await lab.get('/api/state')).json();
    const ledgerBefore = ledgerSnapshot(database);
    const versionBefore = database.prepare('PRAGMA data_version').get();
    assert.equal(before.profiles.length, 3);
    for (const path of ['/api/hd-preview', '/api/hd-image', '/api/hd-preview', '/web/evidence/hd-manifest.json', '/web/evidence/hd-baseline.png']) {
      const response = await lab.get(path);
      assert.equal(response.status, 200, path);
      await response.arrayBuffer();
    }
    for (const path of ['/api/hd-preview', '/api/hd-image']) {
      const response = await lab.get(path, { headers: { Origin: 'https://untrusted.invalid' } });
      assert.equal(response.status, 403);
      assert.match((await response.json()).error, /same-origin/);
    }
    const after = await (await lab.get('/api/state')).json();
    assert.deepEqual(after.profiles, before.profiles);
    assert.deepEqual({ ...after.metrics, asOf: 0 }, { ...before.metrics, asOf: 0 });
    assert.equal(after.graphHash, before.graphHash);
    assert.equal(after.source, before.source);
    assert.deepEqual(ledgerSnapshot(database), ledgerBefore);
    assert.deepEqual(database.prepare('PRAGMA data_version').get(), versionBefore);
  } finally {
    database.close();
  }
});

for (const [path, contentType] of [['/docs/COMFYUI.md', 'text/plain; charset=utf-8']]) {
  test(`HD laboratory explicitly serves ${path} when present`, async context => {
    let expected;
    try {
      expected = await readFile(join(root, path.slice(1)));
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      context.skip(`${path} has not been supplied by its owner`);
      return;
    }
    const lab = await laboratory(context);
    const response = await lab.get(path);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('content-type'), contentType);
    assert.deepEqual(Buffer.from(await response.arrayBuffer()), expected);
    for (const privatePath of ['/ml/angelos_ml/hd.py', '/scripts/hd.js', '/data/angelos.sqlite']) {
      const refused = await lab.get(privatePath);
      assert.equal(refused.status, 404, privatePath);
      await refused.arrayBuffer();
    }
  });
}
