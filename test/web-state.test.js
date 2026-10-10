import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createContext, runInContext } from 'node:vm';
import { webcrypto } from 'node:crypto';
import { parseSource, printSource, graphHash, validateGraph, OPERATIONS, createReference } from '../src/core/graph.js';
import { validateStudent } from '../src/browser/student.js';

const appSource = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const initialization = /^initialize\s*\(\s*\)(?=\s*\.)/gm;
assert.equal([...appSource.matchAll(initialization)].length, 1, 'The VM harness must gate exactly one automatic initialization call');
const executable = appSource.replace(/^import .+;\s*$/gm, '').replace(initialization, 'globalThis.initialization = startInitialization()');
const sources = Object.fromEntries(['aurora-generative', 'aurora-flux'].map(name => [name, readFileSync(new URL(`../examples/${name}.ang`, import.meta.url), 'utf8')]));

function deferred() {
  let resolve;
  const promise = new Promise(complete => { resolve = complete; });
  return { promise, resolve };
}

async function settle(predicate, message) {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    if (predicate()) return;
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.fail(message);
}

async function fixtureArtifact(id, source) {
  const graph = parseSource(source);
  return validateStudent({
    format: 'angelos.student.v1', id, campaignId: graph.id, graphHash: await graphHash(graph),
    encoder: { ...graph.encoder }, inputOrder: ['x', 'y', 'upl.0', 'upl.1', 'upl.2', 'upl.3'],
    layers: [
      { input: 6, output: 1, activation: 'relu', weights: [0, 0, 0, 0, 0, 0], bias: [0] },
      { input: 1, output: 1, activation: 'relu', weights: [0], bias: [0] },
      { input: 1, output: 3, activation: 'sigmoid', weights: [0, 0, 0], bias: [0, 0, 0] },
    ],
    brand: null, metrics: { baselineMse: 0.2, heldoutMse: 0.1, brandMaxError: null, brandTolerance: 0.06 },
    training: { seed: 1, steps: 1, device: 'cpu', fixtureOnly: true },
  });
}

async function laboratory(initialSource = sources['aurora-generative']) {
  const startup = deferred();
  const draws = [];
  const calls = [];
  const requests = [];
  const renderers = [];
  const elements = new Map();
  const profiles = [
    { id: 'allowed', name: 'Synthetic allowed', facts: { ageBand: '25-34', region: 'NA', openness: 0.2, energy: 0.3 }, consent: { personalization: true, telemetry: true, campaignIds: ['aurora'] } },
    { id: 'denied', name: 'Synthetic denied', facts: { ageBand: '65+', region: 'OTHER', openness: 0.9, energy: 0.8 }, consent: { personalization: false, telemetry: false, campaignIds: ['aurora'] } },
  ];
  const server = { source: initialSource, artifact: await fixtureArtifact('state-fixture-tiny', sources['aurora-generative']), claimCompatible: undefined };
  const hooks = { beforeState: undefined, beforeGraph: undefined, beforeArtifact: undefined, beforeRender: undefined };

  function node(id = '') {
    return {
      id, value: '', textContent: '', disabled: false, hidden: false, width: 192, height: 192,
      style: {}, dataset: {}, children: [], classList: { add() {}, remove() {}, toggle() {} },
      setAttribute() {}, addEventListener() {},
      closest(selector) {
        assert.equal(selector, '.result-state');
        return element('result-state');
      },
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) {
        this.children = children;
        if (id === 'profile' && !children.some(child => child.value === this.value)) this.value = children[0]?.value ?? '';
      },
      getContext() {
        return { fillRect() {}, fillText() {}, putImageData: image => { draws.push({ canvas: id, image }); } };
      },
    };
  }

  function element(id) {
    if (!elements.has(id)) elements.set(id, node(id));
    return elements.get(id);
  }

  async function graphState(source = server.source) {
    const graph = parseSource(source);
    const identity = await graphHash(graph);
    return {
      source, graph, graphHash: identity,
      artifactAvailable: Boolean(server.artifact),
      artifactCompatible: server.claimCompatible ?? Boolean(server.artifact && server.artifact.graphHash === identity && server.artifact.campaignId === graph.id),
    };
  }

  async function fetchState(path, options) {
    const payload = options?.body ? JSON.parse(options.body) : undefined;
    requests.push({ path, payload });
    let result;
    let status = 200;
    if (path === '/api/state') {
      await hooks.beforeState?.();
      result = { ...await graphState(), profiles };
    }
    else if (path === '/api/artifact') {
      await hooks.beforeArtifact?.();
      status = server.artifact ? 200 : 404;
      result = server.artifact ?? { error: 'No trained artifact' };
    } else if (path === '/api/graph') {
      await hooks.beforeGraph?.(payload.source);
      server.source = payload.source;
      result = await graphState(payload.source);
    } else if (path === '/api/snapshot') {
      const profile = profiles.find(candidate => candidate.id === payload.profileId);
      assert.equal(profile.consent.personalization, true, 'Denied profiles must never request a new snapshot');
      result = { id: `snapshot-${profile.id}`, profileId: profile.id, campaignId: payload.campaignId, vector: [0.2, 0, profile.facts.openness, profile.facts.energy] };
    } else assert.fail(`Unexpected network request in offline state test: ${path}`);
    return { ok: status === 200, status, json: async () => structuredClone(result) };
  }

  async function createFakeRenderer(packageArtifact) {
    const fake = {
      artifactId: packageArtifact.id, disposed: false,
      dispose() { this.disposed = true; },
      async render(upl, width, height, options) {
        const call = { artifactId: packageArtifact.id, upl: [...upl], width, height, ...options };
        calls.push(call);
        await hooks.beforeRender?.(call);
        return { pixels: new Float32Array(width * height * 4), milliseconds: 1 };
      },
    };
    renderers.push(fake);
    return fake;
  }

  const context = createContext({
    parseSource, printSource, graphHash, validateGraph, OPERATIONS, createReference, validateStudent,
    createGpuRenderer: createFakeRenderer, fetch: fetchState, structuredClone, crypto: webcrypto,
    initializeUplLab: () => ({ sync() {} }),
    startInitialization: () => startup.promise.then(() => context.app.initialize()),
    document: { getElementById: element, createElement: () => node(), createElementNS: () => node(), querySelectorAll: () => [], addEventListener() {} },
    window: { addEventListener() {}, removeEventListener() {} },
    setTimeout: () => 1, clearTimeout() {},
    ImageData: class {
      constructor(data, width, height) {
        assert.equal(data.length, width * height * 4);
        this.data = data;
      }
    },
  });
  runInContext(`${executable}\nglobalThis.app = {
    initialize, applySource, refreshPreview, selectProfile, renderLatest, canRenderStudent,
    read: () => ({ state, artifact, renderer, snapshot, revision, pendingRender, rendering, acceptedRevision }),
  };`, context, { filename: 'web/app.js' });
  const app = context.app;
  const idle = () => settle(() => !app.read().rendering, 'Web state render loop did not settle');
  const start = () => { startup.resolve(); return context.initialization; };
  return { app, server, hooks, draws, calls, requests, renderers, element, idle, start };
}

test('offline web state: gated startup retains the production success chain', async () => {
  const lab = await laboratory();
  assert.equal(lab.requests.length, 0, 'Constructing the harness must not start initialization');
  await lab.start();
  await lab.idle();
  assert.equal(lab.requests.filter(request => request.path === '/api/state').length, 1);
  assert.equal(lab.app.canRenderStudent(), true);
  assert.equal(lab.element('result-state-title').textContent, 'Research preview ready');
  assert.equal(lab.element('show-placement').disabled, false);
});

test('offline web state: startup errors reach the production persistent and transient notices', async () => {
  const lab = await laboratory();
  lab.hooks.beforeState = () => { throw new Error('Fixture: local state unavailable'); };
  await assert.doesNotReject(lab.start());
  assert.equal(lab.element('guide-error').hidden, false);
  assert.match(lab.element('guide-error').textContent, /Fixture: local state unavailable/);
  assert.match(lab.element('guide-error').textContent, /Reload after checking the local server/);
  assert.equal(lab.element('result-state-title').textContent, 'The local lab is unavailable');
  assert.match(lab.element('result-state-detail').textContent, /No successful generation is assumed/);
  assert.equal(lab.element('notice').hidden, false);
  assert.equal(lab.element('notice').textContent, 'Initialization failed: Fixture: local state unavailable');
  assert.equal(lab.renderers.length, 0);
  assert.equal(lab.draws.length, 0);
});

test('offline web state: queued variations cannot render or commit after selecting a nonconsenting profile', async context => {
  for (const composite of [false, true]) {
    await context.test(`invalidate while ${composite ? 'delivered' : 'raw'} render is pending`, async () => {
      const lab = await laboratory();
      await lab.app.initialize();
      await lab.idle();
      const gate = deferred();
      let blocked = false;
      lab.hooks.beforeRender = async call => {
        if (call.composite === composite) {
          blocked = true;
          await gate.promise;
        }
      };
      lab.draws.length = 0;
      lab.calls.length = 0;
      lab.app.refreshPreview();
      await settle(() => blocked, 'Expected an in-flight render before changing profiles');
      lab.app.refreshPreview();
      assert.equal(lab.app.read().pendingRender, true);
      lab.element('profile').value = 'denied';
      try {
        await lab.app.selectProfile();
        assert.equal(lab.app.read().snapshot, undefined);
        assert.equal(lab.app.read().pendingRender, false);
        assert.equal(lab.app.canRenderStudent(), false);
      } finally { gate.resolve(); }
      await lab.idle();
      assert.equal(lab.calls.length, composite ? 2 : 1, 'No queued render or subsequent render stage may run after denial');
      assert.equal(lab.draws.length, 0, 'A stale renderer result must not be committed to either canvas');
      assert.equal(lab.element('show-placement').disabled, true);
      assert.equal(lab.element('click-ad').disabled, true);
      assert.notEqual(lab.app.read().acceptedRevision, lab.app.read().revision);
    });
  }
});

test('offline web state: graph compatibility belongs to the freshly loaded local artifact, not a server flag or cached renderer', async () => {
  const lab = await laboratory();
  await lab.app.initialize();
  await lab.idle();
  const previousRenderer = lab.app.read().renderer;
  lab.server.claimCompatible = true;
  await lab.app.applySource(sources['aurora-flux']);
  await lab.idle();
  assert.equal(previousRenderer.disposed, true);
  assert.equal(lab.app.read().state.artifactCompatible, false, 'A server compatibility flag cannot authorize wrong-hash local weights');
  assert.equal(lab.app.read().renderer, undefined);
  assert.equal(lab.element('show-placement').disabled, true);
  lab.server.artifact = await fixtureArtifact('state-fixture-flux', sources['aurora-flux']);
  lab.server.artifact.campaignId = 'different_campaign';
  await lab.app.applySource(sources['aurora-flux']);
  assert.equal(lab.app.read().state.artifactCompatible, false, 'A matching hash cannot authorize the wrong campaign');
  lab.server.artifact.campaignId = 'aurora';
  await lab.app.applySource(sources['aurora-flux']);
  await lab.idle();
  assert.equal(lab.app.read().artifact.id, 'state-fixture-flux');
  assert.equal(lab.app.read().renderer.artifactId, 'state-fixture-flux');
  assert.equal(lab.app.read().state.artifactCompatible, true);
  assert.equal(lab.app.canRenderStudent(), true);
  assert.equal(lab.calls.at(-1).artifactId, 'state-fixture-flux');
  assert.deepEqual(lab.renderers.map(renderer => renderer.artifactId), ['state-fixture-tiny', 'state-fixture-flux']);
});

test('offline web state: a matching preset creates the renderer skipped during incompatible initialization', async () => {
  const lab = await laboratory(sources['aurora-flux']);
  await lab.app.initialize();
  assert.equal(lab.app.read().renderer, undefined);
  assert.equal(lab.renderers.length, 0);
  assert.equal(lab.app.read().state.artifactCompatible, false);
  await lab.app.applySource(sources['aurora-generative']);
  await lab.idle();
  assert.equal(lab.renderers.length, 1);
  assert.equal(lab.app.read().renderer.artifactId, 'state-fixture-tiny');
  assert.equal(lab.app.canRenderStudent(), true);
  assert.equal(lab.app.read().acceptedRevision, lab.app.read().revision);
  assert.equal(lab.element('show-placement').disabled, false);
  assert.deepEqual(lab.draws.map(draw => draw.canvas), ['student', 'delivered']);
});

test('offline web state: rapid graph applications serialize through artifact synchronization and finish at the server graph', async () => {
  const lab = await laboratory();
  await lab.app.initialize();
  await lab.idle();
  const gate = deferred();
  const artifactGate = deferred();
  let synchronizing = false;
  lab.hooks.beforeGraph = async source => {
    if (source === sources['aurora-flux']) await gate.promise;
  };
  lab.hooks.beforeArtifact = async () => {
    if (lab.server.source === sources['aurora-flux']) {
      synchronizing = true;
      await artifactGate.promise;
    }
  };
  const first = lab.app.applySource(sources['aurora-flux']);
  const second = lab.app.applySource(sources['aurora-generative']);
  try {
    await settle(() => lab.requests.some(request => request.path === '/api/graph'), 'First graph update never started');
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(lab.requests.filter(request => request.path === '/api/graph').length, 1, 'A second graph write must not race the first response');
    gate.resolve();
    await settle(() => synchronizing, 'First graph update never synchronized its artifact');
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(lab.requests.filter(request => request.path === '/api/graph').length, 1, 'The next graph write must wait for the prior artifact load');
  } finally {
    gate.resolve();
    artifactGate.resolve();
  }
  await Promise.all([first, second]);
  await lab.idle();
  assert.deepEqual(lab.requests.filter(request => request.path === '/api/graph').map(request => request.payload.source), [sources['aurora-flux'], sources['aurora-generative']]);
  assert.equal(lab.server.source, sources['aurora-generative']);
  assert.equal(lab.app.read().state.source, lab.server.source);
  assert.equal(lab.element('source').value, lab.server.source);
  assert.equal(lab.app.read().state.graphHash, await graphHash(parseSource(lab.server.source)));
  assert.equal(lab.app.read().artifact.graphHash, lab.app.read().state.graphHash);
  assert.equal(lab.app.canRenderStudent(), true);
});
