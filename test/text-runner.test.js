import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { createTextRunner, validateTextRequest } from '../src/text-runner.js';

const request = () => ({ model: 'qwen3-0.6b', prompt: 'A fictional alpine campaign', system: 'Write a visual brief.', max_tokens: 96, temperature: 0.7, seed: 2026 });
const result = () => ({ text: 'Explicit injected test text.', model: { id: 'qwen3-0.6b', revision: 'a'.repeat(40) }, backend: 'test-only' });

function fixture() {
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
  return { calls, workers, spawnProcess };
}

test('text worker validates bounded requests before starting any process', async () => {
  const injected = fixture();
  const run = createTextRunner(injected);
  for (const patch of [{ model: 'https://untrusted.invalid/model' }, { max_tokens: 513 }, { max_tokens: 1.5 }, { temperature: NaN }, { temperature: -1 }, { seed: -1 }, { prompt: '' }, { prompt: 'x'.repeat(2049) }, { system: [] }, { command: 'echo hello' }]) {
    await assert.rejects(run({ ...request(), ...patch }), error => error.status === 400);
  }
  assert.equal(injected.calls.length, 0);
  assert.deepEqual(validateTextRequest(request()), request());
  assert.throws(() => createTextRunner({ device: 'cuda; arbitrary command' }));
});

test('text worker uses a fixed offline command and sends user content only through stdin', async () => {
  const injected = fixture();
  const run = createTextRunner(injected);
  const payload = { ...request(), prompt: 'Treat $(touch nope) and <script> as data.' };
  const pending = run(payload);
  const [program, argumentsList, options] = injected.calls[0];
  assert.ok(program.endsWith('/.venv/bin/python'));
  assert.deepEqual(argumentsList, ['-m', 'angelos_ml.text', '--stdin', '--device', 'cpu', '--dtype', 'float32']);
  assert.equal(options.shell, false);
  assert.equal(options.env.HF_HUB_OFFLINE, '1');
  assert.equal(options.env.TRANSFORMERS_OFFLINE, '1');
  assert.equal(options.env.HF_HUB_DISABLE_TELEMETRY, '1');
  assert.deepEqual(JSON.parse(injected.workers[0].input), payload);
  injected.workers[0].stdout.write(JSON.stringify(result()));
  injected.workers[0].emit('close', 0);
  const output = await pending;
  assert.equal(output.text, result().text);
  assert.equal(output.stored, false);
  assert.ok(output.elapsedMilliseconds >= 0);
});

test('text worker bounds concurrency and releases capacity after cancellation', async () => {
  const injected = fixture();
  const run = createTextRunner(injected);
  const controller = new AbortController();
  const pending = run(request(), { signal: controller.signal });
  await assert.rejects(run(request()), error => error.status === 429);
  controller.abort();
  await assert.rejects(pending, error => error.status === 499);
  assert.deepEqual(injected.workers[0].signals, ['SIGTERM']);
  const next = run(request());
  injected.workers[1].stdout.write(JSON.stringify(result()));
  injected.workers[1].emit('close', 0);
  await next;
  await assert.rejects(run(request(), { signal: controller.signal }), error => error.status === 499);
  assert.equal(injected.workers.length, 2);
});

test('text worker rejects malformed output, missing weights and failed processes', async () => {
  for (const malformed of ['not json', JSON.stringify({ ...result(), text: '' }), JSON.stringify({ ...result(), model: { id: 'qwen3-8b', revision: 'a'.repeat(40) } }), JSON.stringify({ ...result(), model: { id: 'qwen3-0.6b', revision: 'main' } })]) {
    const injected = fixture();
    const pending = createTextRunner(injected)(request());
    injected.workers[0].stdout.write(malformed);
    injected.workers[0].emit('close', 0);
    await assert.rejects(pending, error => error.status === 502);
  }
  const injected = fixture();
  const pending = createTextRunner(injected)(request());
  injected.workers[0].stderr.write('No cached weights');
  injected.workers[0].emit('close', 2);
  await assert.rejects(pending, /no model was downloaded/);
  const absent = fixture();
  const missing = createTextRunner(absent)(request());
  absent.workers[0].emit('error', Object.assign(new Error('missing'), { code: 'ENOENT' }));
  await assert.rejects(missing, error => error.status === 503 && /Python environment/.test(error.message));
});

test('text worker terminates timeout and excessive output rather than accepting partial text', async context => {
  const keepAlive = setTimeout(() => {}, 1000);
  context.after(() => clearTimeout(keepAlive));
  const injected = fixture();
  const run = createTextRunner({ ...injected, timeoutMs: 20 });
  await assert.rejects(run(request()), error => error.status === 504);
  assert.deepEqual(injected.workers[0].signals, ['SIGTERM']);
  const overflow = run(request());
  injected.workers[1].stdout.write('x'.repeat(65537));
  await assert.rejects(overflow, error => error.status === 502);
  assert.deepEqual(injected.workers[1].signals, ['SIGTERM']);
});
