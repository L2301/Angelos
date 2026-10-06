import { parseSource, printSource, graphHash, validateGraph, OPERATIONS, createReference } from '/src/core/graph.js';
import { createGpuRenderer, validateStudent } from '/src/browser/student.js';

const element = identifier => document.getElementById(identifier);
let state;
let artifact;
let renderer;
let snapshot;
let impression;
let revision = 0;
let rendering = false;
let pendingRender = false;
let acceptedRevision = -1;
let noticeTimer;
let selectedNode;
let exposureTimer;
let selectionRevision = 0;
let creativeSeed = 2026;
let graphUpdates = Promise.resolve();
const positions = new Map();

function notice(message, error = false) {
  clearTimeout(noticeTimer);
  element('notice').textContent = message;
  element('notice').classList.toggle('error', error);
  element('notice').hidden = false;
  noticeTimer = setTimeout(() => { element('notice').hidden = true; }, 6500);
}

async function request(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
  const result = await response.json();
  if (!response.ok) throw Object.assign(new Error(result.error ?? `Request failed: ${response.status}`), { status: response.status });
  return result;
}

function task(callback) {
  return async event => { try { await callback(event); } catch (error) { notice(error.message, true); } };
}

function currentProfile() {
  return state.profiles.find(profile => profile.id === element('profile').value);
}

function previewVector() {
  return [['18-24', '25-34', '35-44', '45-54', '55-64', '65+'].indexOf(element('age-band').value) / 5, ['NA', 'EU', 'APAC', 'OTHER'].indexOf(element('region').value) / 3, Number(element('openness').value), Number(element('energy').value)];
}

function drawPixels(canvas, pixels) {
  const bytes = Uint8ClampedArray.from(pixels, value => Math.round(Math.max(0, Math.min(1, value)) * 255));
  canvas.getContext('2d').putImageData(new ImageData(bytes, canvas.width, canvas.height), 0, 0);
}

function clearCanvas(identifier, message) {
  const canvas = element(identifier);
  const context = canvas.getContext('2d');
  context.fillStyle = '#e8eddf';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = '#8a987d';
  context.font = '10px system-ui';
  context.textAlign = 'center';
  context.fillText(message, canvas.width / 2, canvas.height / 2);
}

function invalidateExposure() {
  clearTimeout(exposureTimer);
  impression = undefined;
  element('click-ad').disabled = true;
  element('show-placement').disabled = true;
  element('exposure-status').textContent = 'No exposed impression for this creative.';
}

function refreshPreview() {
  revision += 1;
  pendingRender = false;
  invalidateExposure();
  if (!currentProfile()) return;
  if (!currentProfile().consent.personalization || !currentProfile().consent.campaignIds.includes(state.graph.id)) {
    for (const identifier of ['teacher', 'student', 'delivered']) clearCanvas(identifier, 'Personalization not permitted');
    return;
  }
  if (!snapshot) {
    for (const identifier of ['teacher', 'student', 'delivered']) clearCanvas(identifier, 'Complete saved ingredients required');
    return;
  }
  const upl = previewVector();
  element('openness-value').textContent = upl[2].toFixed(2);
  element('energy-value').textContent = upl[3].toFixed(2);
  element('latent-values').textContent = `[ ${upl.map(value => value.toFixed(2)).join(', ')} ]`;
  element('latent-bars').replaceChildren(...upl.map(value => {
    const bar = document.createElement('span');
    bar.style.height = `${2 + value * 32}px`;
    return bar;
  }));
  const canvas = element('teacher');
  const generative = state.graph.nodes.some(node => node.op.startsWith('generate_'));
  element('teacher-backend').textContent = generative ? 'Offline open-weight teacher' : 'CPU diagnostic reference';
  element('teacher-caption').textContent = generative ? 'Run the complete teacher offline with npm run dataset. Changing this UPL runs the distilled student, not FLUX or Tiny-SD in the browser.' : 'Deterministic graph reference. No pretrained model in this graph.';
  if (generative) {
    clearCanvas('teacher', 'Open-weight graph · offline teacher');
  } else {
    const reference = createReference(state.graph);
    const pixels = new Float32Array(canvas.width * canvas.height * 4);
    for (let row = 0; row < canvas.height; row += 1) {
      for (let column = 0; column < canvas.width; column += 1) {
        const offset = (row * canvas.width + column) * 4;
        pixels.set([...reference(upl, (column + 0.5) / canvas.width, (row + 0.5) / canvas.height), 1], offset);
      }
    }
    drawPixels(canvas, pixels);
  }
  element('creative-seed').textContent = String(creativeSeed);
  pendingRender = true;
  void renderLatest();
}

function canRenderStudent() {
  const profile = currentProfile();
  return Boolean(renderer && artifact && snapshot && profile && snapshot.profileId === profile.id && snapshot.campaignId === state.graph.id && artifact.graphHash === state.graphHash && artifact.campaignId === state.graph.id && state.artifactCompatible && profile.consent.personalization && profile.consent.campaignIds.includes(state.graph.id));
}

async function renderLatest() {
  if (rendering || !canRenderStudent()) return;
  rendering = true;
  let activeRenderer = renderer;
  try {
    while (pendingRender) {
      pendingRender = false;
      if (!canRenderStudent()) break;
      const renderingRevision = revision;
      const upl = previewVector();
      activeRenderer = renderer;
      const seed = creativeSeed;
      const raw = await activeRenderer.render(upl, 192, 192, { composite: false, seed });
      if (renderingRevision !== revision || activeRenderer !== renderer || !canRenderStudent()) continue;
      const delivered = await activeRenderer.render(upl, 192, 192, { composite: true, seed });
      if (renderingRevision !== revision || activeRenderer !== renderer || !canRenderStudent()) continue;
      drawPixels(element('student'), raw.pixels);
      drawPixels(element('delivered'), delivered.pixels);
      element('render-time').textContent = `${delivered.milliseconds.toFixed(1)} ms`;
      acceptedRevision = renderingRevision;
      const profile = currentProfile();
      element('show-placement').disabled = !(snapshot && JSON.stringify(snapshot.vector) === JSON.stringify(upl) && profile?.consent?.telemetry && profile?.consent?.personalization);
    }
  } catch (error) {
    if (activeRenderer === renderer && canRenderStudent()) {
      pendingRender = false;
      element('gpu-status').textContent = 'WebGPU unavailable · no hidden fallback';
      clearCanvas('student', 'GPU render failed');
      clearCanvas('delivered', 'Not served');
      invalidateExposure();
      notice(error.message, true);
    }
  } finally {
    rendering = false;
    if (pendingRender && canRenderStudent()) void renderLatest();
  }
}

function svg(tag, attributes = {}, text) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderGraph() {
  const graph = element('graph');
  const { ordered } = validateGraph(state.graph);
  const depths = new Map();
  const rows = new Map();
  for (const node of ordered) {
    const depth = Math.max(-1, ...Object.values(node.inputs).map(source => depths.get(source))) + 1;
    depths.set(node.id, depth);
    const row = rows.get(depth) ?? 0;
    rows.set(depth, row + 1);
    if (!positions.has(node.id)) positions.set(node.id, { x: 24 + depth * 165, y: 38 + row * 105 });
  }
  graph.setAttribute('width', Math.max(640, ...[...positions.values()].map(position => position.x + 170)));
  graph.setAttribute('height', Math.max(260, ...[...positions.values()].map(position => position.y + 100)));
  graph.replaceChildren();
  for (const node of ordered) {
    const target = positions.get(node.id);
    for (const sourceId of Object.values(node.inputs)) {
      const source = positions.get(sourceId);
      graph.append(svg('path', { class: 'graph-edge', d: `M${source.x + 135},${source.y + 36} C${source.x + 153},${source.y + 36} ${target.x - 25},${target.y + 36} ${target.x},${target.y + 36}` }));
    }
  }
  for (const node of ordered) {
    const position = positions.get(node.id);
    const group = svg('g', { class: `graph-node${selectedNode === node.id ? ' selected' : ''}`, transform: `translate(${position.x} ${position.y})`, tabindex: '0', role: 'button', 'aria-label': `Inspect ${node.id}: ${node.op}` });
    group.append(svg('rect', { width: 135, height: 73, rx: 7 }), svg('text', { x: 12, y: 21, class: 'node-kind' }, OPERATIONS[node.op].output.toUpperCase()), svg('text', { x: 12, y: 42 }, node.id), svg('text', { x: 12, y: 59, class: 'node-op' }, `${node.op}()`), svg('circle', { cx: 135, cy: 36, r: 3 }));
    if (Object.keys(node.inputs).length) group.append(svg('circle', { cx: 0, cy: 36, r: 3 }));
    group.addEventListener('click', () => inspectNode(node.id));
    group.addEventListener('keydown', event => { if (event.key === 'Enter') inspectNode(node.id); });
    group.addEventListener('pointerdown', event => {
      const initialX = event.clientX;
      const initialY = event.clientY;
      const initial = { ...positions.get(node.id) };
      function move(next) {
        positions.set(node.id, { x: Math.max(5, initial.x + next.clientX - initialX), y: Math.max(5, initial.y + next.clientY - initialY) });
        renderGraph();
      }
      function stop() { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', stop); }
      window.addEventListener('pointermove', move);
      window.addEventListener('pointerup', stop, { once: true });
    });
    graph.append(group);
  }
  element('graph-count').textContent = `${ordered.length} typed nodes · checked DAG`;
  element('operation-count').textContent = `${Object.keys(OPERATIONS).length} supported operations`;
  element('graph-hash').textContent = `SHA-256 ${state.graphHash.slice(0, 14)}…`;
}

function inspectNode(nodeId) {
  selectedNode = nodeId;
  const node = state.graph.nodes.find(candidate => candidate.id === nodeId);
  const inspector = element('inspector');
  inspector.replaceChildren();
  const title = document.createElement('div');
  title.className = 'eyebrow';
  title.textContent = `${node.id} / ${node.op} → ${OPERATIONS[node.op].output}`;
  inspector.append(title);
  const description = document.createElement('p');
  description.textContent = Object.entries(node.inputs).map(([port, source]) => `${port} ← ${source}`).join('   ·   ') || 'Source node. No upstream inputs.';
  inspector.append(description);
  const form = document.createElement('form');
  const fields = [];
  for (const [name, value] of Object.entries(node.params)) {
    const label = document.createElement('label');
    label.textContent = name;
    const input = document.createElement('input');
    input.value = JSON.stringify(value);
    input.setAttribute('aria-label', `${node.id}.${name}`);
    label.append(input);
    form.append(label);
    fields.push({ name, input });
  }
  if (fields.length) {
    const button = document.createElement('button');
    button.className = 'button small';
    button.textContent = 'Apply parameters';
    form.append(button);
    form.addEventListener('submit', task(async event => {
      event.preventDefault();
      const graph = structuredClone(state.graph);
      const changed = graph.nodes.find(candidate => candidate.id === nodeId);
      for (const { name, input } of fields) changed.params[name] = JSON.parse(input.value);
      await applySource(printSource(graph));
    }));
    inspector.append(form);
  }
  renderGraph();
}

function applySource(source) {
  parseSource(source);
  const update = graphUpdates.then(() => performApplySource(source));
  graphUpdates = update.catch(() => {});
  return update;
}

async function performApplySource(source) {
  pendingRender = false;
  revision += 1;
  invalidateExposure();
  const result = await request('/api/graph', { source });
  state = { ...state, ...result, artifactCompatible: false };
  element('source').value = state.source;
  revision += 1;
  invalidateExposure();
  clearCanvas('student', 'Rebuild weights for this graph');
  clearCanvas('delivered', 'Artifact compatibility required');
  element('gpu-status').textContent = 'Checking current campaign weights';
  renderGraph();
  if (selectedNode && state.graph.nodes.some(node => node.id === selectedNode)) inspectNode(selectedNode);
  await synchronizeArtifact();
  await selectProfile();
  notice('Graph checked and saved. A changed graph needs new campaign weights.');
}

async function selectProfile() {
  const selection = ++selectionRevision;
  const profile = currentProfile();
  snapshot = undefined;
  pendingRender = false;
  revision += 1;
  invalidateExposure();
  for (const identifier of ['teacher', 'student', 'delivered']) clearCanvas(identifier, 'Loading selected profile');
  for (const identifier of ['age-band', 'region', 'openness', 'energy', 'save-facts', 'save-consent', 'delete-profile', 'save-review']) element(identifier).disabled = !profile;
  if (!profile) {
    element('profile-json').textContent = 'No synthetic profiles remain. Deletion is not undone on restart.';
    for (const identifier of ['teacher', 'student', 'delivered']) clearCanvas(identifier, 'No profile selected');
    element('latent-values').textContent = '[ —, —, —, — ]';
    element('latent-bars').replaceChildren();
    return;
  }
  element('age-band').value = profile.facts.ageBand ?? '25-34';
  element('region').value = profile.facts.region ?? 'NA';
  element('openness').value = profile.facts.openness ?? 0.5;
  element('energy').value = profile.facts.energy ?? 0.5;
  element('profile-json').textContent = JSON.stringify(profile, null, 2);
  for (const scope of ['personalization', 'telemetry', 'training']) element(`allow-${scope}`).checked = Boolean(profile.consent?.[scope]);
  if (profile.consent?.personalization) {
    try {
      const result = await request('/api/snapshot', { profileId: profile.id, campaignId: state.graph.id });
      if (selection !== selectionRevision) return;
      snapshot = result;
    } catch (error) {
      if (selection !== selectionRevision) return;
      for (const identifier of ['teacher', 'student', 'delivered']) clearCanvas(identifier, 'Could not encode profile');
      notice(error.message, true);
      return;
    }
  }
  refreshPreview();
}

async function refreshLedger() {
  const metrics = await request('/api/metrics');
  element('ledger-json').textContent = JSON.stringify(metrics, null, 2);
  const entries = [['Exposed impressions', metrics.impressions ?? 0], ['Attributed clicks', metrics.clicks ?? 0], ['Observed CTR', `${((metrics.ctr ?? 0) * 100).toFixed(1)}%`], ['Human reviews', metrics.reviews ?? 0]];
  element('ledger-metrics').replaceChildren(...entries.map(([label, value]) => {
    const card = document.createElement('div');
    card.textContent = label;
    const number = document.createElement('strong');
    number.textContent = value;
    card.append(number);
    return card;
  }));
}

function installHandlers() {
  for (const button of document.querySelectorAll('[data-section]')) button.addEventListener('click', task(async () => {
    for (const section of document.querySelectorAll('.section')) section.hidden = section.id !== button.dataset.section;
    for (const navigation of document.querySelectorAll('.nav')) navigation.classList.toggle('active', navigation === button);
    invalidateExposure();
    if (button.dataset.section === 'studio') refreshPreview();
    if (button.dataset.section === 'learning') await refreshLedger();
  }));
  element('show-canvas').onclick = () => { element('canvas-area').hidden = false; element('source-area').hidden = true; element('show-canvas').classList.add('selected'); element('show-source').classList.remove('selected'); };
  element('show-source').onclick = () => { element('canvas-area').hidden = true; element('source-area').hidden = false; element('show-source').classList.add('selected'); element('show-canvas').classList.remove('selected'); };
  element('apply-source').onclick = task(() => applySource(element('source').value));
  element('load-example').onclick = task(async () => {
    const examples = await request('/api/examples');
    const example = examples.find(candidate => candidate.id === element('teacher-example').value);
    if (!example) throw new Error('Unknown graph preset');
    await applySource(example.source);
    notice('Graph loaded only. Model downloads and teacher generation require explicit CLI commands.');
  });
  element('profile').onchange = task(selectProfile);
  element('new-seed').onclick = () => {
    creativeSeed = crypto.getRandomValues(new Uint32Array(1))[0];
    refreshPreview();
  };
  for (const identifier of ['openness', 'energy', 'age-band', 'region']) element(identifier).addEventListener('input', refreshPreview);
  element('save-facts').onclick = task(async () => {
    const profile = currentProfile();
    if (!profile) throw new Error('No profile selected');
    const facts = { ageBand: element('age-band').value, region: element('region').value, openness: Number(element('openness').value), energy: Number(element('energy').value) };
    let expectedRevision = profile.factRevision;
    for (const [field, value] of Object.entries(facts)) {
      if (profile.facts[field] === value) continue;
      const result = await request('/api/facts', { profileId: profile.id, field, value, expectedRevision, eventId: crypto.randomUUID() });
      expectedRevision = result.factRevision;
    }
    state.profiles = await request('/api/profiles');
    await selectProfile();
    notice('Ingredient facts saved. A new UPL snapshot is independent of the old one.');
  });
  element('save-consent').onclick = task(async () => {
    const profile = currentProfile();
    if (!profile) throw new Error('No profile selected');
    const consent = Object.fromEntries(['personalization', 'telemetry', 'training'].map(scope => [scope, element(`allow-${scope}`).checked]));
    await request('/api/consent', { profileId: profile.id, campaignId: state.graph.id, ...consent });
    state.profiles = await request('/api/profiles');
    await selectProfile();
    notice('Consent updated. Revoked permissions take effect immediately.');
  });
  element('delete-profile').onclick = task(async () => {
    const profile = currentProfile();
    if (!profile || !confirm(`Delete synthetic profile ${profile.name} and its dependent local records?`)) return;
    await request('/api/delete-profile', { profileId: profile.id });
    state.profiles = await request('/api/profiles');
    populateProfiles();
    await selectProfile();
    notice('Profile and dependent local records deleted.');
  });
  element('show-placement').onclick = task(async () => {
    if (!snapshot || acceptedRevision !== revision || !state.artifactCompatible) throw new Error('Save ingredients and wait for a compatible GPU render first');
    const capturedRevision = revision;
    const capturedSnapshot = snapshot;
    element('show-placement').disabled = true;
    element('exposure-status').textContent = 'Waiting for one continuous second of ≥50% visible creative…';
    let visibleSince;
    const qualify = async () => {
      if (capturedRevision !== revision) return;
      const bounds = element('delivered').getBoundingClientRect();
      const visibleWidth = Math.max(0, Math.min(bounds.right, innerWidth) - Math.max(bounds.left, 0));
      const visibleHeight = Math.max(0, Math.min(bounds.bottom, innerHeight) - Math.max(bounds.top, 0));
      const visible = bounds.width > 0 && bounds.height > 0 && document.visibilityState === 'visible' && visibleWidth * visibleHeight >= bounds.width * bounds.height * 0.5;
      if (!visible) visibleSince = undefined;
      else visibleSince ??= performance.now();
      if (visibleSince !== undefined && performance.now() - visibleSince >= 1000) {
        try {
          const recorded = await request('/api/impressions', { profileId: currentProfile().id, campaignId: state.graph.id, snapshotId: capturedSnapshot.id, artifactId: artifact.id, graphHash: artifact.graphHash, seed: creativeSeed, eventId: crypto.randomUUID(), exposure: { visibleFraction: Math.min(1, visibleWidth * visibleHeight / (bounds.width * bounds.height)), visibleMilliseconds: Math.floor(performance.now() - visibleSince), documentVisible: document.visibilityState === 'visible' } });
          if (capturedRevision !== revision) return;
          impression = recorded;
          element('click-ad').disabled = false;
          element('exposure-status').textContent = `Exposed impression ${recorded.id.slice(0, 12)}… · profile and model pinned.`;
        } catch (error) { notice(error.message, true); }
      } else exposureTimer = setTimeout(qualify, 100);
    };
    await qualify();
  });
  element('click-ad').onclick = task(async () => {
    if (!impression) throw new Error('No qualified exposure');
    await request('/api/click', { impressionId: impression.id, campaignId: state.graph.id, eventId: crypto.randomUUID() });
    element('click-ad').disabled = true;
    element('exposure-status').textContent = 'Click attributed to this exposed impression. No model update has been triggered.';
    notice('Click recorded once. Preview and speculative work are never negative examples.');
  });
  element('save-review').onclick = task(async () => {
    if (!artifact || !state.artifactCompatible || acceptedRevision !== revision || !snapshot || JSON.stringify(snapshot.vector) !== JSON.stringify(previewVector())) throw new Error('Build and render a compatible student using a saved profile first');
    await request('/api/reviews', { profileId: currentProfile().id, campaignId: state.graph.id, snapshotId: snapshot.id, artifactId: artifact.id, seed: creativeSeed, quality: Number(element('quality-score').value), brand: Number(element('brand-score').value), eventId: crypto.randomUUID() });
    notice('Review saved with model identity. Critic training is a separate future stage.');
  });
  element('training-help').onclick = () => element('training-dialog').showModal();
  element('close-training').onclick = () => element('training-dialog').close();
  element('refresh-ledger').onclick = task(refreshLedger);
  document.addEventListener('visibilitychange', () => { if (document.hidden) invalidateExposure(); });
  window.addEventListener('beforeunload', () => { renderer?.dispose(); clearTimeout(exposureTimer); });
}

function populateProfiles() {
  const previous = element('profile').value;
  element('profile').replaceChildren(...state.profiles.map(profile => {
    const option = document.createElement('option');
    option.value = profile.id;
    option.textContent = profile.name;
    return option;
  }));
  if (state.profiles.some(profile => profile.id === previous)) element('profile').value = previous;
}

async function synchronizeArtifact() {
  revision += 1;
  pendingRender = false;
  invalidateExposure();
  renderer?.dispose();
  renderer = undefined;
  artifact = undefined;
  state.artifactCompatible = false;
  try {
    artifact = validateStudent(await request('/api/artifact'));
  } catch (error) {
    if (error.status !== 404) throw error;
    state.artifactAvailable = false;
    element('gpu-status').textContent = 'No trained artifact · build a student';
    return;
  }
  state.artifactAvailable = true;
  state.artifactCompatible = artifact.graphHash === await graphHash(state.graph) && artifact.campaignId === state.graph.id;
  element('mse').textContent = artifact.metrics.heldoutMse.toFixed(5);
  const diffusion = artifact.format === 'angelos.diffusion.v1';
  element('mse-label').textContent = diffusion ? 'HELD-OUT NOISE MSE' : 'HELD-OUT PIXEL MSE';
  element('student-caption').textContent = diffusion ? `${artifact.schedule.timesteps.length}-step conditional diffusion · native ${artifact.architecture.resolution}×${artifact.architecture.resolution}, enlarged for display. Research quality, not a high-resolution ad model.` : 'Legacy coordinate MLP fixture. Not a diffusion model.';
  element('conditioning-status').textContent = diffusion && artifact.metrics.improvedWithCorrectUpl === false ? 'Conditioning audit did not improve with the correct UPL versus mismatched UPLs. This candidate does not demonstrate useful personalization.' : 'Numerical UPL sensitivity is not proof of relevant personalization or improved click rates.';
  element('brand-error').textContent = artifact.metrics.brandMaxError?.toFixed(3) ?? 'No mark';
  const parameters = artifact.layers.reduce((count, layer) => count + layer.weights.length + layer.bias.length, 0);
  element('student-size').textContent = `${(parameters * 4 / 1024).toFixed(1)} KiB FP32`;
  element('brand-mode').textContent = artifact.brand?.mode === 'composite' ? 'Exact GPU mark' : 'Neural-only';
  if (!state.artifactCompatible) { element('gpu-status').textContent = 'Artifact is stale · rebuild for current graph'; return; }
  try {
    renderer = await createGpuRenderer(artifact);
    element('gpu-status').textContent = 'WebGPU student · active';
  } catch (error) {
    element('gpu-status').textContent = 'WebGPU unavailable · reference only';
    clearCanvas('student', 'WebGPU not available');
    clearCanvas('delivered', 'Not served');
    notice(`WebGPU initialization failed: ${error.message}`, true);
  }
}

async function initialize() {
  state = await request('/api/state');
  populateProfiles();
  element('source').value = state.source;
  renderGraph();
  clearCanvas('student', 'Train a student to see learned pixels');
  clearCanvas('delivered', 'No model loaded');
  await synchronizeArtifact();
  installHandlers();
  await selectProfile();
}

initialize().catch(error => notice(`Initialization failed: ${error.message}`, true));
