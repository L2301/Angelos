import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const models = new Set(['qwen3-0.6b', 'qwen3-8b']);
const failure = (message, status = 400) => Object.assign(new Error(message), { status });

export function validateTextRequest(value) {
  const fields = ['model', 'prompt', 'system', 'max_tokens', 'temperature', 'seed'];
  if (!value || typeof value !== 'object' || Array.isArray(value) || Object.keys(value).length !== fields.length || fields.some(field => !Object.hasOwn(value, field))) throw failure('Expected model, prompt, system, max_tokens, temperature and seed');
  if (!models.has(value.model)) throw failure('Choose an allowlisted Qwen3 text model');
  for (const field of ['prompt', 'system']) {
    if (typeof value[field] !== 'string' || !value[field].trim() || value[field].length > 2048) throw failure(`${field} must be nonempty text, at most 2048 characters`);
  }
  if (!Number.isInteger(value.max_tokens) || value.max_tokens < 1 || value.max_tokens > 512) throw failure('max_tokens must be an integer from 1 to 512');
  if (typeof value.temperature !== 'number' || !Number.isFinite(value.temperature) || value.temperature < 0 || value.temperature > 2) throw failure('temperature must be from 0 to 2');
  if (!Number.isSafeInteger(value.seed) || value.seed < 0) throw failure('seed must be a nonnegative safe integer');
  return Object.fromEntries(fields.map(field => [field, value[field]]));
}

export function createTextRunner(options = {}) {
  const spawnProcess = options.spawnProcess ?? spawn;
  const device = options.device ?? process.env.ANGELOS_TEXT_DEVICE ?? 'cpu';
  const dtype = options.dtype ?? process.env.ANGELOS_TEXT_DTYPE ?? 'float32';
  if (!['cpu', 'mps', 'cuda'].includes(device) || !['float32', 'float16', 'bfloat16'].includes(dtype)) throw new Error('Invalid text worker device or dtype');
  let active = false;

  return async (payload, { signal } = {}) => {
    const input = validateTextRequest(payload);
    if (signal?.aborted) throw failure('Text generation cancelled', 499);
    if (active) throw failure('A text generation is already running. Wait or cancel it first.', 429);
    active = true;
    const started = performance.now();
    try {
      return await new Promise((resolveResult, reject) => {
        let output = '';
        let diagnostics = '';
        let outputSize = 0;
        let stopped;
        let forceKill;
        const worker = spawnProcess(resolve(root, '.venv/bin/python'), ['-m', 'angelos_ml.text', '--stdin', '--device', device, '--dtype', dtype], {
          cwd: root,
          shell: false,
          env: { ...process.env, PYTHONPATH: resolve(root, 'ml'), HF_HOME: process.env.HF_HOME ?? resolve(root, 'artifacts/huggingface'), HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', HF_HUB_DISABLE_TELEMETRY: '1', TOKENIZERS_PARALLELISM: 'false' },
          stdio: ['pipe', 'pipe', 'pipe'],
        });
        const stop = error => {
          if (stopped) return;
          stopped = error;
          worker.kill('SIGTERM');
          forceKill = setTimeout(() => worker.kill('SIGKILL'), 1500);
          forceKill.unref?.();
        };
        const abort = () => stop(failure('Text generation cancelled', 499));
        const timer = setTimeout(() => stop(failure('Text generation timed out. Use fewer tokens or a smaller model.', 504)), options.timeoutMs ?? 120000);
        timer.unref?.();
        const cleanup = () => {
          clearTimeout(timer);
          clearTimeout(forceKill);
          signal?.removeEventListener('abort', abort);
        };
        signal?.addEventListener('abort', abort, { once: true });
        if (signal?.aborted) abort();
        worker.stdout.setEncoding('utf8');
        worker.stderr.setEncoding('utf8');
        worker.stdout.on('data', chunk => {
          outputSize += Buffer.byteLength(chunk);
          if (outputSize > 65536) return stop(failure('Text worker exceeded its output limit', 502));
          output += chunk;
        });
        worker.stderr.on('data', chunk => { diagnostics = (diagnostics + chunk).slice(-6000); });
        worker.stdin.on('error', () => {});
        worker.on('error', error => {
          cleanup();
          reject(failure(error.code === 'ENOENT' ? 'Python environment is missing. Follow the text-model setup instructions.' : 'Unable to start the local text worker.', 503));
        });
        worker.on('close', code => {
          cleanup();
          if (stopped) return reject(stopped);
          if (code !== 0) return reject(failure(`Local text generation failed. Install dependencies and explicitly cache ${input.model}; no model was downloaded. ${diagnostics.trim().slice(-2000)}`, 503));
          try {
            const result = JSON.parse(output);
            if (typeof result.text !== 'string' || !result.text.trim() || result.text.length > 16384 || result.model?.id !== input.model || typeof result.model.revision !== 'string' || !/^[a-f0-9]{40}$/.test(result.model.revision)) throw new Error('Invalid text result');
            resolveResult({ ...result, elapsedMilliseconds: performance.now() - started, execution: 'local-pretrained-text', stored: false });
          } catch {
            reject(failure('The text worker returned an invalid result; no output is accepted.', 502));
          }
        });
        worker.stdin.end(JSON.stringify(input));
      });
    } finally {
      active = false;
    }
  };
}
