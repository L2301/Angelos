import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { parseSource, graphHash, canonicalJson } from '../src/core/graph.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = process.argv.slice(2);
const source = args[0] && !args[0].startsWith('--') ? args.shift() : 'examples/aurora-generative.ang';
const graph = parseSource(await readFile(resolve(root, source), 'utf8'));
await mkdir(resolve(root, 'artifacts'), { recursive: true });
const graphPath = resolve(root, 'artifacts', `${graph.id}.${Date.now()}.dataset.graph.json`);
await writeFile(graphPath, canonicalJson({ ...graph, nodes: [...graph.nodes].sort((left, right) => left.id < right.id ? -1 : left.id > right.id ? 1 : 0) }));
const subprocess = spawn(resolve(root, '.venv/bin/python'), ['-m', 'angelos_ml.dataset', '--graph', graphPath, '--graph-canonical', graphPath, '--graph-hash', await graphHash(graph), '--output', resolve(root, `artifacts/${graph.id}.dataset.npz`), '--device', process.env.ANGELOS_DEVICE ?? 'cpu', ...args], {
  cwd: root,
  env: { ...process.env, PYTHONPATH: resolve(root, 'ml'), HF_HOME: process.env.HF_HOME ?? resolve(root, 'artifacts/huggingface') },
  stdio: 'inherit',
});
subprocess.on('error', error => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
subprocess.on('exit', code => { process.exitCode = code ?? 1; });
