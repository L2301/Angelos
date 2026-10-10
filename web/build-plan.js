import { validateGraph } from '../src/core/graph.js';

export function buildCommands(graph, hash, runId, device = 'cpu') {
  validateGraph(graph);
  if (!/^[a-f0-9]{64}$/.test(hash) || !/^[a-zA-Z0-9_-]{1,80}$/.test(runId)) throw new Error('Invalid build identity');
  if (!['cpu', 'mps', 'cuda'].includes(device)) throw new Error('Invalid build device');
  const models = [...new Set(graph.nodes.filter(node => node.op.startsWith('generate_')).map(node => node.params.model))];
  const dataset = `artifacts/${graph.id}.${hash.slice(0, 12)}.${runId}.dataset.npz`;
  const prefix = `ANGELOS_DEVICE=${device}`;
  return {
    models, dataset, device,
    commands: [
      ...models.map(model => `npm run models -- download ${model}`),
      `${prefix} npm run dataset -- data/campaign.ang --output ${dataset}`,
      `${prefix} npm run train -- data/campaign.ang 2000 ${dataset}`,
    ].join('\n'),
  };
}
