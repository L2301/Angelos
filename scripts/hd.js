import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { constants } from 'node:os';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const download = args[0] === 'download';
const subprocess = spawn(resolve(root, '.venv/bin/python'), ['-m', 'angelos_ml.hd', ...args], {
  cwd: root,
  env: {
    ...process.env,
    PYTHONPATH: resolve(root, 'ml'),
    HF_HOME: process.env.HF_HOME ?? resolve(root, 'artifacts/huggingface'),
    HF_HUB_OFFLINE: download ? '0' : '1',
    TRANSFORMERS_OFFLINE: '1',
    HF_HUB_DISABLE_TELEMETRY: '1',
  },
  stdio: 'inherit',
});
subprocess.on('error', error => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
subprocess.on('exit', (code, signal) => {
  if (signal) process.exitCode = 128 + (constants.signals[signal] ?? 0);
  else process.exitCode = code ?? 1;
});
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => subprocess.kill(signal));
