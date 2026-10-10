import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const documents = ['MODEL-COVERAGE', 'SHOWCASE-AUDIT', 'SHOWCASE-EXECUTION', 'DIFFUSION-EXECUTION', 'TEXT', 'MODELS', 'RESEARCH-IMAGE', 'RESEARCH-VIDEO', 'RESEARCH-AUDIO', 'RESEARCH-TEXT-3D', 'RESEARCH-OPERATIONS', 'RESEARCH-RL', 'RESEARCH-DIFFUSION', 'RESEARCH-DISTILLATION', 'RESEARCH-UPL', 'UPL', 'UPL-EXECUTION', 'RL', 'HD', 'HD-ALIGNMENT', 'HD-EXECUTION', 'COMFYUI', 'PLAN', 'CONTRACTS', 'DIFFUSION', 'BACKLOG'];

export async function buildSite(outputDirectory, base = '/') {
  if (!/^\/(?:[A-Za-z0-9_-]+\/)*$/.test(base)) throw new Error('Base must be a slash-delimited path, for example / or /Angelos/');
  const output = resolve(outputDirectory);
  if (output === root) throw new Error('Choose a new output directory, not the repository root');
  await mkdir(dirname(output), { recursive: true });
  await mkdir(output);
  const files = [
    ['web/story.html', 'index.html'], ['web/local-lab.html', 'lab/index.html'],
    ['integrations/comfyui_angelos/README.md', 'integrations/comfyui_angelos/README.md'],
    ...['story.css', 'story.js'].map(file => [`web/${file}`, `web/${file}`]),
    ...['teacher.svg', 'student.svg', 'delivered.svg', 'manifest.json', 'hd-baseline.png', 'hd-manifest.json'].map(file => [`web/evidence/${file}`, `web/evidence/${file}`]),
    ...documents.map(name => [`docs/${name}.md`, `docs/${name}.md`]),
  ];
  const manifest = { mode: 'static-showcase-only', base, createdAt: new Date().toISOString(), liveInference: false, profileData: false, files: [] };
  for (const [source, destination] of files) {
    const target = resolve(output, destination);
    await mkdir(dirname(target), { recursive: true });
    if (destination.endsWith('.html')) {
      const html = (await readFile(resolve(root, source), 'utf8')).replace(/(href=["'])\/lab(?=[#"'])/g, '$1/lab/').replace(/((?:href|src)=["'])\/(?!\/)/g, `$1${base}`);
      await writeFile(target, html);
    } else {
      await copyFile(resolve(root, source), target);
    }
    const bytes = await readFile(target);
    manifest.files.push({ path: destination, sha256: createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length });
  }
  await writeFile(resolve(output, 'site-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  return manifest;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const manifest = await buildSite(resolve(root, process.argv[2] ?? 'dist'), process.argv[3] ?? '/');
    process.stdout.write(`Built ${manifest.files.length} static files. No lab API, profiles or model weights included. Base: ${manifest.base}\n`);
  } catch (error) {
    process.stderr.write(`${error.code === 'EEXIST' ? 'Output already exists; choose a fresh directory to avoid mixing builds.' : error.message}\n`);
    process.exitCode = 1;
  }
}
