import { readFile, writeFile, stat } from 'node:fs/promises';
import { importComfyWorkflow, compileComfyProgram } from '../src/core/comfy-program.js';

try {
  const [command, inputPath, flag, outputPath, ...extra] = process.argv.slice(2);
  if (!['import', 'compile'].includes(command) || !inputPath || flag !== '--output' || !outputPath || extra.length) throw new Error('Usage: npm run comfy:program -- import|compile INPUT --output NEW_OUTPUT');
  if ((await stat(inputPath)).size > 262144) throw new Error('Input exceeds 256 KiB');
  const source = await readFile(inputPath, 'utf8');
  const result = command === 'import' ? importComfyWorkflow(source) : JSON.stringify(compileComfyProgram(source), null, 2) + '\n';
  await writeFile(outputPath, result, { flag: 'wx' });
  process.stdout.write(`Wrote ${outputPath}. Translation only: no models loaded, workflow execution or training.\n`);
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exitCode = 1;
}
