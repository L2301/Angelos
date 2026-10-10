import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { buildSite } from '../scripts/build-site.js';

test('static export is portable and never packages laboratory code, profiles or weights', async context => {
  const directory = await mkdtemp(join(tmpdir(), 'angelos-site-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const output = join(directory, 'site');
  const manifest = await buildSite(output, '/Angelos/');
  assert.equal(manifest.liveInference, false);
  assert.equal(manifest.profileData, false);
  assert.ok(manifest.files.length > 10);
  assert.ok(manifest.files.every(file => !/app\.js|server\.js|student\.json|sqlite|safetensors|\.env/.test(file.path)));
  assert.ok(manifest.files.every(file => /^[a-f0-9]{64}$/.test(file.sha256)));
  const html = await readFile(join(output, 'index.html'), 'utf8');
  assert.match(html, /\/Angelos\/web\/story.css/);
  assert.match(html, /\/Angelos\/lab\//);
  assert.doesNotMatch(html, /(?:src|href)="\/(?:web|docs|lab)/);
  const local = await readFile(join(output, 'lab/index.html'), 'utf8');
  assert.match(local, /not a hosted service/);
  assert.match(local, /http:\/\/127.0.0.1:4310\/lab/);
  assert.deepEqual((await readdir(output)).sort(), ['docs', 'index.html', 'integrations', 'lab', 'site-manifest.json', 'web']);
  assert.match(await readFile(join(output, 'integrations/comfyui_angelos/README.md'), 'utf8'), /ComfyUI/);
  await assert.rejects(buildSite(output), error => error.code === 'EEXIST');
  await assert.rejects(buildSite(join(directory, 'invalid'), '/../../'), /Base must/);
});
