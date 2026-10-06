import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parseSource, graphHash, canonicalJson } from '../src/core/graph.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sourcePath = resolve(root, process.argv[2] ?? 'examples/aurora.ang');
const graph = parseSource(await readFile(sourcePath, 'utf8'));
await mkdir(resolve(root, 'artifacts'), { recursive: true });
const buildId = `${graph.id}.${Date.now()}`;
const graphPath = resolve(root, 'artifacts', `${buildId}.graph.json`);
await writeFile(graphPath, canonicalJson({ ...graph, nodes: [...graph.nodes].sort((left, right) => left.id < right.id ? -1 : left.id > right.id ? 1 : 0) }));
const subprocess = spawn(resolve(root, '.venv/bin/python'), ['-m', 'angelos_ml.compile', '--graph', graphPath, '--graph-canonical', graphPath, '--graph-hash', await graphHash(graph), '--output', resolve(root, 'artifacts', `${buildId}.teacher.pt2`), '--samples', '256'], {
  cwd: root, env: { ...process.env, PYTHONPATH: resolve(root, 'ml') }, stdio: 'inherit',
});
subprocess.on('error', error => { process.stderr.write(`${error.message}\n`); process.exitCode = 1; });
subprocess.on('exit', code => { process.exitCode = code ?? 1; });
