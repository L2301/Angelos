import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { parseSource, graphHash } from '../src/core/graph.js';
import { buildCommands } from '../web/build-plan.js';

test('rebuild commands bind downloads to the graph and share a fresh dataset path', async () => {
  for (const [example,models] of [['aurora', []], ['aurora-generative', ['tiny-sd']], ['aurora-flux', ['flux2-klein-4b']], ['aurora-ltx', ['ltx-video']], ['aurora-text', ['qwen3-0.6b', 'tiny-sd']]]) {
    const graph = parseSource(readFileSync(new URL(`../examples/${example}.ang`, import.meta.url), 'utf8'));
    const hash = await graphHash(graph);
    const plan = buildCommands(graph, hash, 'run-123', 'mps');
    assert.deepEqual([...plan.models].sort(), [...models].sort());
    assert.ok(plan.dataset.includes(hash.slice(0, 12)));
    assert.ok(plan.commands.includes(`--output ${plan.dataset}`));
    assert.ok(plan.commands.endsWith(`2000 ${plan.dataset}`));
    assert.notEqual(buildCommands(graph, hash, 'run-124').dataset, plan.dataset);
    assert.throws(() => buildCommands(graph, hash, 'shell; command'));
    assert.throws(() => buildCommands(graph, hash, 'valid', 'untrusted'));
  }
});
