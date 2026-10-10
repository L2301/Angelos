import { readFile, writeFile, access } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Store } from '../src/core/store.js';
import { parseSource } from '../src/core/graph.js';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = process.argv[2];
if (!output || process.argv.length !== 3) throw new Error('Usage: npm run feedback:export -- NEW_OUTPUT.json');
await access(resolve(root, 'data/angelos.sqlite'));
const graph = parseSource(await readFile(resolve(root, 'data/campaign.ang'), 'utf8'));
const store = new Store(resolve(root, 'data/angelos.sqlite'));
try {
  const feedback = store.learningFeedback({ campaignId: graph.id });
  await writeFile(resolve(output), JSON.stringify(feedback, null, 2) + '\n', { flag: 'wx' });
  process.stdout.write(`Exported ${feedback.clicks.length} mature click labels and ${feedback.reviews.length} eligible reviews. No training or promotion ran.\n`);
} finally { store.close(); }
