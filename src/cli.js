import { readFile, writeFile } from 'node:fs/promises';
import { parseSource, printSource, graphHash, canonicalJson, validateGraph } from './core/graph.js';

const [command, inputPath, outputPath] = process.argv.slice(2);
try {
  if (!inputPath || !['compile', 'check', 'decompile'].includes(command)) throw new Error('Usage: node src/cli.js compile|check|decompile INPUT [OUTPUT]');
  const input = await readFile(inputPath, 'utf8');
  const graph = inputPath.endsWith('.json') ? JSON.parse(input) : parseSource(input);
  validateGraph(graph);
  let output;
  if (command === 'decompile') output = printSource(graph);
  else if (command === 'check') output = JSON.stringify({ valid: true, graphHash: await graphHash(graph), nodes: graph.nodes.length }, null, 2) + '\n';
  else output = canonicalJson(graph) + '\n';
  if (outputPath) await writeFile(outputPath, output);
  else process.stdout.write(output);
} catch (error) {
  process.stderr.write(`${error.message}\n`);
  process.exitCode = 1;
}
