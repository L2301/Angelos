import { spawn } from 'node:child_process';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { UPL_DIMENSIONS, uplCanonical, validateSemanticBundle } from './core/upl.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const failure = (message, status) => Object.assign(new Error(message), { status });

export function createUplRunner(options = {}) {
  const spawnProcess = options.spawnProcess ?? spawn;
  const device = options.device ?? process.env.ANGELOS_UPL_DEVICE ?? 'cpu';
  if (!['cpu', 'mps', 'cuda'].includes(device)) throw new Error('Invalid UPL worker device');
  let active = false;
  return async ({ ingredients, dimension = 512 }, { signal } = {}) => {
    if (!UPL_DIMENSIONS.includes(dimension) || ingredients?.format !== 'angelos.upl-ingredients.v2') throw failure('Invalid UPL encoding request', 400);
    const serialized = JSON.stringify({ ingredients, ingredientsJson: uplCanonical(ingredients), dimension });
    if (Buffer.byteLength(serialized) > 128 * 1024) throw failure('UPL ingredients exceed the input limit', 400);
    if (signal?.aborted) throw failure('UPL encoding cancelled', 499);
    if (active) throw failure('A UPL encoding is already running; wait or cancel it first', 429);
    active = true;
    try {
      return await new Promise((accept, reject) => {
        let output = '';
        let diagnostics = '';
        let outputBytes = 0;
        let stopped;
        let forceKill;
        const worker = spawnProcess(resolve(root, '.venv/bin/python'), ['-m', 'angelos_ml.upl', '--stdin', '--device', device], {
          cwd: root, shell: false, stdio: ['pipe', 'pipe', 'pipe'],
          env: { ...process.env, PYTHONPATH: resolve(root, 'ml'), HF_HOME: process.env.HF_HOME ?? resolve(root, 'artifacts/huggingface'),
            HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', HF_HUB_DISABLE_TELEMETRY: '1', TOKENIZERS_PARALLELISM: 'false' },
        });
        const stop = error => {
          if (stopped) return;
          stopped = error;
          worker.kill('SIGTERM');
          forceKill = setTimeout(() => worker.kill('SIGKILL'), 1500);
          forceKill.unref?.();
        };
        const abort = () => stop(failure('UPL encoding cancelled', 499));
        const timer = setTimeout(() => stop(failure('UPL encoding timed out; no partial embedding was accepted', 504)), options.timeoutMs ?? 180000);
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
          outputBytes += Buffer.byteLength(chunk);
          if (outputBytes > 256 * 1024) return stop(failure('UPL worker exceeded its output limit', 502));
          output += chunk;
        });
        worker.stderr.on('data', chunk => { diagnostics = (diagnostics + chunk).slice(-4000); });
        worker.stdin.on('error', () => {});
        worker.on('error', () => {
          cleanup();
          reject(failure('Unable to start the local UPL worker. Install the Python environment described in docs/UPL.md.', 503));
        });
        worker.on('close', code => {
          cleanup();
          if (stopped) return reject(stopped);
          if (code !== 0) return reject(failure(`Local UPL encoding failed. Explicitly cache the pinned Qwen3-Embedding-0.6B checkpoint; no model was downloaded. ${diagnostics.trim().slice(-1500)}`, 503));
          try {
            const result = validateSemanticBundle(JSON.parse(output), ingredients);
            if (result.encoder.dimension !== dimension) throw new Error('Dimension mismatch');
            accept(result);
          } catch {
            reject(failure('Invalid or mismatched UPL worker output; no embedding was accepted', 502));
          }
        });
        worker.stdin.end(serialized);
      });
    } finally {
      active = false;
    }
  };
}
