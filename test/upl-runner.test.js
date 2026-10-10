import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { createUplRunner } from '../src/upl-runner.js';
import { Store } from '../src/core/store.js';
import { uplCanonical } from '../src/core/upl.js';
import { semanticFixture } from './upl-fixture.js';

function fixture(context) {
  const store = new Store(':memory:');
  context.after(() => store.close());
  const profile = store.createProfile({ name: 'Synthetic fixture', facts: { ageBand: '35-44', interests: ['hiking'] },
    consent: { personalization: true, telemetry: true, training: true, campaignIds: ['aurora'] } });
  const ingredients = store.uplIngredients(profile.id, 'aurora');
  const calls = [];
  const workers = [];
  const spawnProcess = (...argumentsList) => {
    calls.push(argumentsList);
    const worker = new EventEmitter();
    worker.stdin = new PassThrough();
    worker.stdout = new PassThrough();
    worker.stderr = new PassThrough();
    worker.input = '';
    worker.signals = [];
    worker.stdin.on('data', chunk => { worker.input += chunk.toString(); });
    worker.kill = signal => { worker.signals.push(signal); queueMicrotask(() => worker.emit('close', null)); };
    workers.push(worker);
    return worker;
  };
  return { ingredients, calls, workers, spawnProcess };
}

test('semantic worker uses fixed offline invocation and binds result to original input bytes', async context => {
  const injected = fixture(context);
  const run = createUplRunner(injected);
  await assert.rejects(run({ ingredients: injected.ingredients, dimension: 4 }), error => error.status === 400);
  assert.equal(injected.calls.length, 0);
  const pending = run({ ingredients: injected.ingredients, dimension: 512 });
  const [program, argumentsList, options] = injected.calls[0];
  assert.ok(program.endsWith('/.venv/bin/python'));
  assert.deepEqual(argumentsList, ['-m', 'angelos_ml.upl', '--stdin', '--device', 'cpu']);
  assert.equal(options.shell, false);
  assert.equal(options.env.HF_HUB_OFFLINE, '1');
  assert.equal(options.env.TRANSFORMERS_OFFLINE, '1');
  assert.equal(options.env.HF_HUB_DISABLE_TELEMETRY, '1');
  assert.equal(JSON.parse(injected.workers[0].input).ingredientsJson, uplCanonical(injected.ingredients));
  injected.workers[0].stdout.write(JSON.stringify(semanticFixture(injected.ingredients)));
  injected.workers[0].emit('close', 0);
  assert.equal((await pending).upl.length, 512);
});

test('semantic worker rejects output mismatch, unavailable weights and output overflow', async context => {
  for (const output of ['not JSON', JSON.stringify({}), JSON.stringify(semanticFixture(fixture(context).ingredients, 128))]) {
    const injected = fixture(context);
    const pending = createUplRunner(injected)({ ingredients: injected.ingredients, dimension: 512 });
    injected.workers[0].stdout.write(output);
    injected.workers[0].emit('close', 0);
    await assert.rejects(pending, error => error.status === 502);
  }
  const injected = fixture(context);
  const run = createUplRunner(injected);
  const missing = run({ ingredients: injected.ingredients });
  injected.workers[0].stderr.write('Checkpoint unavailable offline');
  injected.workers[0].emit('close', 1);
  await assert.rejects(missing, /no model was downloaded/);
  const excessive = run({ ingredients: injected.ingredients });
  injected.workers[1].stdout.write('x'.repeat(256 * 1024 + 1));
  await assert.rejects(excessive, error => error.status === 502);
  assert.deepEqual(injected.workers[1].signals, ['SIGTERM']);
});

test('semantic worker limits concurrency, cancels, times out and releases its slot', async context => {
  const injected = fixture(context);
  const run = createUplRunner({ ...injected, timeoutMs: 20 });
  const controller = new AbortController();
  const pending = run({ ingredients: injected.ingredients }, { signal: controller.signal });
  await assert.rejects(run({ ingredients: injected.ingredients }), error => error.status === 429);
  controller.abort();
  await assert.rejects(pending, error => error.status === 499);
  await assert.rejects(run({ ingredients: injected.ingredients }, { signal: controller.signal }), error => error.status === 499);
  const keepAlive = setTimeout(() => {}, 1000);
  context.after(() => clearTimeout(keepAlive));
  await assert.rejects(run({ ingredients: injected.ingredients }), error => error.status === 504);
  assert.deepEqual(injected.workers[1].signals, ['SIGTERM']);
});
