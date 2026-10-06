import { mkdir, readFile, writeFile, access, rename } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { parseSource, graphHash, canonicalJson } from '../src/core/graph.js';
import { validateStudent } from '../src/browser/student.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const fixture = process.argv.includes('--fixture-mlp');
const args = process.argv.slice(2).filter(argument => argument !== '--fixture-mlp');
const savedSourceExists = await access(resolve(root, 'data/campaign.ang')).then(() => true, () => false);
const sourcePath = resolve(root, args[0] ?? (savedSourceExists ? 'data/campaign.ang' : 'examples/aurora-generative.ang'));
const steps = args[1] ?? '2000';
if (!/^\d+$/.test(steps) || Number(steps) < 1 || Number(steps) > 100_000) throw new Error('Steps must be an integer from 1 to 100000');
const graph = parseSource(await readFile(sourcePath, 'utf8'));
const datasetPath = resolve(root, args[2] ?? `artifacts/${graph.id}.dataset.npz`);
if (!fixture && !await access(datasetPath).then(() => true, () => false)) throw new Error(`Missing full-graph teacher dataset: ${datasetPath}. Run npm run dataset first; training never silently substitutes procedural images.`);
await mkdir(resolve(root, 'artifacts'), { recursive: true });
const graphPath = resolve(root, 'artifacts', `${graph.id}.graph.json`);
await writeFile(graphPath, canonicalJson({ ...graph, nodes: [...graph.nodes].sort((left, right) => left.id < right.id ? -1 : left.id > right.id ? 1 : 0) }));
const outputName = `${graph.id}.${Date.now()}.student.json`;
const outputPath = resolve(root, 'artifacts', outputName);
const moduleName = fixture ? 'angelos_ml.train' : 'angelos_ml.train_diffusion';
const datasetArguments = fixture ? [] : ['--dataset', datasetPath];
const subprocess = spawn(resolve(root, '.venv/bin/python'), ['-m', moduleName, '--graph', graphPath, '--graph-canonical', graphPath, '--graph-hash', await graphHash(graph), '--output', outputPath, '--steps', steps, '--device', process.env.ANGELOS_DEVICE ?? 'cpu', ...datasetArguments], {
  cwd: root,
  env: { ...process.env, PYTHONPATH: resolve(root, 'ml') },
  stdio: 'inherit',
});
subprocess.on('error', error => { process.stderr.write(`${error.message}. Install the ML dependencies first; see README.\n`); process.exitCode = 1; });
subprocess.on('exit', async code => {
  process.exitCode = code ?? 1;
  if (code !== 0) return;
  try {
    const student = validateStudent(JSON.parse(await readFile(outputPath, 'utf8')));
    if (!student.metrics.improvedOverUntrained) throw new Error('Candidate did not improve over untrained baseline; preview pointer unchanged');
    const pointer = resolve(root, 'artifacts/preview.json');
    const temporary = `${pointer}.${Date.now()}.tmp`;
    await writeFile(temporary, JSON.stringify({ file: outputName }, null, 2) + '\n');
    await rename(temporary, pointer);
    process.stdout.write(`\nPreview candidate selected: ${outputName}\nThis is not a production promotion. Reload the laboratory.\n`);
  } catch (error) { process.stderr.write(`${error.message}\n`); process.exitCode = 1; }
});
