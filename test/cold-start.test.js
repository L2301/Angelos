import test from 'node:test';
import assert from 'node:assert/strict';
import { cp, mkdtemp, rm, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

test('a source-only checkout starts without Python, weights, artifacts or node_modules', async context => {
  const directory = await mkdtemp(join(tmpdir(), 'angelos-cold-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  for (const path of ['src', 'web', 'examples', 'package.json']) await cp(resolve(path), join(directory, path), { recursive: true });
  for (const path of ['.venv', 'artifacts', 'node_modules', 'data']) await assert.rejects(access(join(directory, path)));
  const { createLab } = await import(pathToFileURL(join(directory, 'src/server.js')));
  const { server } = await createLab();
  await new Promise(resolveReady => server.listen(0, '127.0.0.1', resolveReady));
  context.after(() => new Promise(resolveClosed => server.close(resolveClosed)));
  const origin = `http://127.0.0.1:${server.address().port}`;
  assert.match(await (await fetch(origin)).text(), /Louis Anderson/);
  assert.match(await (await fetch(`${origin}/lab`)).text(), /id="text-form"/);
  const state = await (await fetch(`${origin}/api/state`)).json();
  assert.equal(state.artifactAvailable, false);
  assert.equal(state.graph.id, 'aurora');
  assert.equal((await fetch(`${origin}/api/artifact`)).status, 404);
  const text = await fetch(`${origin}/api/text-preview`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: 'qwen3-0.6b', prompt: 'Synthetic test', system: 'Be brief.', max_tokens: 8, temperature: 0, seed: 0 }),
  });
  assert.equal(text.status, 503);
  assert.match((await text.json()).error, /Python environment is missing/);
  assert.equal((await fetch(`${origin}/api/state`)).status, 200);
  await assert.rejects(access(join(directory, 'artifacts')));
});
