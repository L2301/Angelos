import { parseSource, printSource, graphHash, validateGraph, OPERATIONS, createReference } from '/src/core/graph.js';
import { createGpuRenderer, validateStudent } from '/src/browser/student.js';
import { buildCommands } from '/web/build-plan.js';
import { initializeUplLab } from '/web/upl-lab.js';

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
let previousSource;
let buildRunId;
let handoffSnapshot;
let handoffJson;
let uplCopyPending = false;
let uplLab;
const positions = new Map();

function refreshGuide() {
  if (!state) return;
  updateGuide({ state, artifact, snapshot, profile: currentProfile(), vector: previewVector(), ready: canRenderStudent(), rendered: acceptedRevision === revision, revision });
  refreshUplHandoff();
  uplLab?.sync();
}

function comfyUplPayload() {
  const profile = currentProfile();
  if (!profile?.consent.personalization || !profile.consent.campaignIds.includes(state.graph.id) || !element('allow-personalization').checked) {
    throw new Error('Personalization permission is required before copying a UPL.');
  }
  if (profile.synthetic !== true || !snapshot || snapshot.profileId !== profile.id || snapshot.campaignId !== state.graph.id
    || snapshot.factRevision !== profile.factRevision || snapshot.consentVersion !== profile.consentVersion) {
    throw new Error('Select a saved, consented fictional adult snapshot before copying.');
  }
  if (JSON.stringify(snapshot.vector) !== JSON.stringify(previewVector())) throw new Error('Save these preferences before copying. Unsaved changes cannot be exported.');
  if (snapshot.factSchemaVersion !== 'angelos.facts.v1' || snapshot.encoder?.id !== 'structured-demo-v1' || snapshot.encoder.dimension !== 4 || !snapshot.facts) {
    throw new Error('This saved snapshot does not use the supported ComfyUI UPL schema.');
  }
  const { ageBand, region, openness, energy } = snapshot.facts;
  const facts = { ageBand, region, openness, energy };
  const ageIndex = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'].indexOf(ageBand);
  const regionIndex = ['NA', 'EU', 'APAC', 'OTHER'].indexOf(region);
  const expected = [ageIndex / 5, regionIndex / 3, openness, energy];
  if (ageIndex < 0 || regionIndex < 0 || !Array.isArray(snapshot.vector) || snapshot.vector.length !== 4
    || expected.some(value => typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1)
    || expected.some((value, index) => value !== snapshot.vector[index])
    || Object.entries(facts).some(([field, value]) => value !== profile.facts[field])) {
    throw new Error('The frozen facts and UPL do not match exactly. Select the saved person again.');
  }
  return { format: 'angelos.upl-input.v1', synthetic: true, adult: true, factsSchema: 'angelos.facts.v1', facts,
    encoder: { id: 'structured-demo-v1', dimension: 4 }, upl: [...snapshot.vector] };
}

function refreshUplHandoff() {
  let nextJson;
  let message = 'Ready to copy this saved snapshot into Angelos Parse UPL.';
  try { nextJson = JSON.stringify(comfyUplPayload(), null, 2); }
  catch (error) { message = error.message; }
  if (handoffJson !== nextJson || handoffSnapshot !== snapshot || !nextJson) {
    element('upl-copy-json').value = '';
    element('upl-copy-fallback').hidden = true;
    element('upl-copy-status').textContent = message;
  }
  handoffJson = nextJson;
  handoffSnapshot = snapshot;
  element('copy-upl').disabled = !nextJson || uplCopyPending;
}

async function copyUpl() {
  refreshUplHandoff();
  if (!handoffJson || uplCopyPending) return;
  const copiedJson = handoffJson;
  const copiedSnapshot = snapshot;
  uplCopyPending = true;
  element('copy-upl').disabled = true;
  element('upl-copy-status').textContent = 'Copying the saved snapshot…';
  try {
    await navigator.clipboard.writeText(copiedJson);
    refreshUplHandoff();
    if (handoffJson === copiedJson && snapshot === copiedSnapshot) element('upl-copy-status').textContent = 'Saved UPL copied. Paste it into Angelos Parse UPL in ComfyUI.';
  } catch {
    refreshUplHandoff();
    if (handoffJson === copiedJson && snapshot === copiedSnapshot) {
      element('upl-copy-json').value = copiedJson;
      element('upl-copy-fallback').hidden = false;
      element('upl-copy-status').textContent = 'Clipboard unavailable. Select the JSON below and copy it manually.';
      element('upl-copy-json').focus();
      element('upl-copy-json').select();
    }
  } finally {
    uplCopyPending = false;
    refreshUplHandoff();
  }
}

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
  refreshGuide();
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
  element('teacher-caption').textContent = generative ? 'The full image teacher has not run in this browser. Run the saved recipe offline with the local build commands. Preference changes run only a compatible student, not the teacher.' : 'A deterministic drawing made from the saved recipe. No pretrained image model runs in this reference.';
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
  if (!canRenderStudent()) {
    clearCanvas('student', artifact ? 'Student unavailable for this recipe' : 'No trained model loaded');
    clearCanvas('delivered', 'No current student result');
    refreshGuide();
    return;
  }
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
    refreshGuide();
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
    group.append(svg('title', {}, `${node.id}: ${explainNode(node).description}`), svg('rect', { width: 135, height: 73, rx: 7 }), svg('text', { x: 12, y: 21, class: 'node-kind' }, OPERATIONS[node.op].output.toUpperCase()), svg('text', { x: 12, y: 42 }, explainNode(node).title), svg('text', { x: 12, y: 59, class: 'node-op' }, `${node.op}()`), svg('circle', { cx: 135, cy: 36, r: 3 }));
    if (Object.keys(node.inputs).length) group.append(svg('circle', { cx: 0, cy: 36, r: 3 }));
    group.addEventListener('click', () => inspectNode(node.id));
    group.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); inspectNode(node.id); } });
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
  describeRecipe(state.graph, ordered);
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
  const explanation = document.createElement('p');
  explanation.textContent = `${explainNode(node).title}. ${explainNode(node).description}`;
  inspector.append(explanation);
  const description = document.createElement('p');
  description.textContent = Object.entries(node.inputs).map(([port, source]) => `${port} ← ${source}`).join('   ·   ') || 'Source node. No upstream inputs.';
  inspector.append(description);
  if (node.op === 'prompt' || node.op === 'generate_text') {
    const explanation = document.createElement('p');
    explanation.textContent = node.op === 'prompt' ? 'Deterministic formatting: selects a template and substitutes the numeric condition. No language model runs here.' : 'Learned generation: a pinned local Qwen3 model produces text for downstream nodes. Separate from prompt formatting; executed by the offline teacher.';
    inspector.append(explanation);
  }
  const form = document.createElement('form');
  const fields = [];
  for (const [name, value] of Object.entries(node.params)) {
    const label = document.createElement('label');
    label.textContent = name;
    const input = document.createElement('input');
    input.value = typeof value === 'string' ? value : JSON.stringify(value);
    input.setAttribute('aria-label', `${node.id}.${name}`);
    label.append(input);
    form.append(label);
    fields.push({ name, input, text: typeof value === 'string' });
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
      for (const { name, input, text } of fields) changed.params[name] = text ? input.value : JSON.parse(input.value);
      await applySource(printSource(graph));
    }));
    const parameters = document.createElement('details');
    const summary = document.createElement('summary');
    summary.textContent = 'Edit step parameters · text values need no quotes';
    parameters.append(summary, form);
    inspector.append(parameters);
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
  if (result.source !== state.source) previousSource = state.source;
  state = { ...state, ...result, artifactCompatible: false };
  element('restore-source').disabled = !previousSource;
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
      refreshGuide();
      notice(error.message, true);
      return;
    }
  }
  refreshPreview();
}

async function refreshLedger() {
  const metrics = await request('/api/metrics');
  element('ledger-json').textContent = JSON.stringify(metrics, null, 2);
  const entries = [['Local exposures', metrics.impressions ?? 0], ['Recorded test clicks', metrics.clicks ?? 0], ['Local test click rate', `${((metrics.ctr ?? 0) * 100).toFixed(1)}%`], ['Human reviews', metrics.reviews ?? 0]];
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
    for (const navigation of document.querySelectorAll('.nav')) {
      navigation.classList.toggle('active', navigation === button);
      if (navigation === button) navigation.setAttribute('aria-current', 'page');
      else navigation.removeAttribute('aria-current');
    }
    invalidateExposure();
    if (button.dataset.section === 'studio') refreshPreview();
    if (button.dataset.section === 'learning') await refreshLedger();
  }));
  element('show-canvas').onclick = () => { element('canvas-area').hidden = false; element('source-area').hidden = true; element('show-canvas').classList.add('selected'); element('show-source').classList.remove('selected'); element('show-canvas').setAttribute('aria-pressed', 'true'); element('show-source').setAttribute('aria-pressed', 'false'); };
  element('show-source').onclick = () => { element('canvas-area').hidden = true; element('source-area').hidden = false; element('show-source').classList.add('selected'); element('show-canvas').classList.remove('selected'); element('show-source').setAttribute('aria-pressed', 'true'); element('show-canvas').setAttribute('aria-pressed', 'false'); };
  element('apply-source').onclick = task(() => applySource(element('source').value));
  element('restore-source').onclick = task(async () => {
    if (!previousSource) return;
    await applySource(previousSource);
    notice('Previous saved source restored. Matching artifacts can be used again.');
  });
  element('download-source').onclick = () => {
    const url = URL.createObjectURL(new Blob([state.source], { type: 'text/plain' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${state.graph.id}.${state.graphHash.slice(0, 12)}.ang`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
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
    notice('Preferences saved. This version has its own preference snapshot.');
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
    element('delivered').scrollIntoView({ block: 'center' });
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
    markReviewSaved(revision);
    refreshGuide();
    notice('Review saved with model identity. Candidate training is separate; no model update was triggered.');
  });
  element('training-help').onclick = () => {
    buildRunId = `${Date.now()}-${crypto.randomUUID().slice(0, 8)}`;
    updateBuildCommands();
    element('training-dialog').showModal();
  };
  element('build-device').onchange = updateBuildCommands;
  element('close-training').onclick = () => element('training-dialog').close();
  element('refresh-ledger').onclick = task(refreshLedger);
  installGuide();
  document.addEventListener('visibilitychange', () => { if (document.hidden) invalidateExposure(); });
  window.addEventListener('beforeunload', () => { renderer?.dispose(); clearTimeout(exposureTimer); });
}

function updateBuildCommands() {
  const plan = buildCommands(state.graph, state.graphHash, buildRunId, element('build-device').value);
  element('training-command').textContent = plan.commands;
  element('build-identity').textContent = `Saved graph ${state.graphHash.slice(0, 12)} · fresh dataset ${plan.dataset}`;
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
  element('mse').textContent = '—';
  element('brand-error').textContent = '—';
  element('render-time').textContent = '—';
  element('student-size').textContent = 'Unbuilt';
  element('brand-mode').textContent = 'Awaiting validation';
  try {
    artifact = validateStudent(await request('/api/artifact'));
  } catch (error) {
    state.artifactAvailable = false;
    element('gpu-status').textContent = error.status === 404 ? 'No trained artifact · build a student' : 'Artifact invalid · graph tools remain usable';
    if (error.status !== 404) notice(`Could not load the preview artifact: ${error.message}`, true);
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
  installHandlers();
  uplLab = initializeUplLab({
    getContext: () => ({ profile: currentProfile(), campaignId: state.graph.id,
      legacyDirty: Boolean(currentProfile() && JSON.stringify(previewVector()) !== JSON.stringify([
        ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'].indexOf(currentProfile().facts.ageBand) / 5,
        ['NA', 'EU', 'APAC', 'OTHER'].indexOf(currentProfile().facts.region) / 3,
        currentProfile().facts.openness, currentProfile().facts.energy])) }),
    onFactsSaved: async () => { state.profiles = await request('/api/profiles'); await selectProfile(); },
  });
  try {
    await synchronizeArtifact();
  } catch (error) {
    renderer?.dispose();
    renderer = undefined;
    artifact = undefined;
    state.artifactCompatible = false;
    element('gpu-status').textContent = 'Artifact unavailable · graph tools remain usable';
    notice(`Could not load the preview artifact: ${error.message}`, true);
  }
  await selectProfile();
}

let reviewedRevision;

const operations = {
  upl: ['Preferences (UPL)', 'Pass four declared preference numbers into the recipe. No personality is inferred.'],
  feature: ['Read a preference', 'Select one preference number to control a later step.'],
  constant: ['Fixed value', 'Supply a fixed number to the next step.'],
  prompt: ['Format the brief', 'Choose and format instructions using a preference. This is a template, not a language model.'],
  generate_text: ['Write with a model', 'A local pretrained text model writes new words for the next step. It runs in the offline teacher.'],
  generate_image: ['Generate an image', 'A pretrained image model follows the brief. This teacher step runs offline, not when you move a slider.'],
  generate_video: ['Generate a video', 'A pretrained video model follows the brief offline. This is not live browser video generation.'],
  frame: ['Choose a frame', 'Take one image from the generated video.'],
  gradient: ['Paint a background', 'Make a deterministic color gradient. No pretrained image model runs in this step.'],
  disk: ['Add a light shape', 'Draw a soft circular shape, adjusted by a preference. This is a procedural operation.'],
  exposure: ['Adjust brightness', 'Change image brightness using the energy preference.'],
  vignette: ['Shade the edges', 'Darken the edges to focus attention toward the center.'],
  brand: ['Add the brand mark', 'Compose an exact brand mark onto the image. This is separate from what a student learns.'],
};

function explainNode(node) {
  if (node.op === 'feature') {
    const field = ['age range', 'broad region', 'openness', 'energy'][node.params.index];
    return { title: `Read ${field}`, description: `Select the ${field} number from the fictional person's preferences.` };
  }
  const [title, description] = operations[node.op] ?? [node.op, 'Inspect this saved operation and its parameters.'];
  return { title, description };
}

function describeRecipe(graph, ordered) {
  const generative = graph.nodes.some(node => node.op.startsWith('generate_'));
  element('recipe-name').textContent = `${graph.id} / ${generative ? 'offline generative teacher' : 'procedural reference · no pretrained model'}`;
  element('recipe-summary').textContent = generative
    ? 'This recipe includes offline model generation. Loading it does not execute it.'
    : 'This saved recipe draws colors and shapes. It is a procedural test, not a pretrained image generator.';
  element('recipe-steps').replaceChildren(...ordered.map(node => {
    const item = document.createElement('li');
    const explanation = explainNode(node);
    item.textContent = `${explanation.title}: ${explanation.description} (${node.id})`;
    return item;
  }));
}

function markReviewSaved(revision) {
  reviewedRevision = revision;
}

function updateGuide({ state, artifact, snapshot, profile, vector, ready, rendered, revision }) {
  const saved = Boolean(snapshot && JSON.stringify(snapshot.vector) === JSON.stringify(vector));
  const permitted = Boolean(profile?.consent.personalization && profile.consent.campaignIds.includes(state.graph.id));
  element('profile-save-status').textContent = !profile ? 'No fictional people remain. Nothing is selected.'
    : !permitted ? 'Personalization is not permitted for this recipe. Check People & permissions.'
      : !snapshot ? 'No usable saved preferences. Select a person again or check their permissions.'
        : saved ? 'Using saved preferences. You can change them to explore another version.'
          : 'Unsaved preferences: preview only. Save these preferences before grading or recording events.';
  let title;
  let detail;
  if (!profile) {
    title = 'No person selected';
    detail = 'A preview needs a fictional person and permitted preferences. No previous person’s result is being shown.';
  } else if (!permitted) {
    title = 'Personalization is not permitted';
    detail = 'Open People & permissions to review this person’s settings. Rendering and event recording are unavailable.';
  } else if (!snapshot) {
    title = 'Preferences are unavailable';
    detail = 'The selected person could not be prepared. Select them again or check permissions; no output is ready to review.';
  } else if (!artifact) {
    title = 'No usable trained model is loaded';
    detail = 'You can still explore and edit the recipe. Previewing the student needs a compatible artifact (saved trained weights). See local build instructions below; nothing trains automatically.';
  } else if (!state.artifactCompatible) {
    title = 'The saved model does not match this recipe';
    detail = 'Changing a recipe does not retrain the student. Restore the matching recipe or explicitly build matching weights. Previous output is not a result for this recipe.';
  } else if (!ready || element('gpu-status').textContent.includes('unavailable')) {
    title = 'The browser GPU is unavailable';
    detail = 'No student image is being served. Use a browser with a working WebGPU adapter. The recipe reference may still work, but it is not a CPU substitute for the trained student.';
  } else if (!rendered) {
    title = 'Preparing the current preview';
    detail = 'Wait for the matching student to finish rendering. Earlier pixels, if visible, are not yet the current result and cannot be graded or exposed.';
  } else {
    title = saved ? 'Research preview ready' : 'Preview ready · preferences not saved';
    detail = artifact.format === 'angelos.diffusion.v1'
      ? `The student rendered at native ${artifact.architecture.resolution} × ${artifact.architecture.resolution}, enlarged for display. Compare its raw pixels with the separate brand treatment. This is not the HD foundation example.`
      : 'This is the legacy coordinate-model fixture, not a pretrained image generator or a diffusion model. Numerical rendering is not evidence of ad quality.';
  }
  element('result-state-title').textContent = title;
  element('result-state-detail').textContent = detail;
  element('result-state-title').closest('.result-state').dataset.ready = String(Boolean(rendered && ready));
  element('new-seed').disabled = !(rendered && ready);
  element('review-status').textContent = reviewedRevision === revision ? 'Review saved for this model and preference snapshot. No model update was triggered.'
    : !saved ? 'Save usable preferences before grading this version.'
      : !rendered || !ready ? 'Wait for a compatible rendered result before grading. A reference or HD example alone is not reviewable here.'
        : 'Ready to grade this student result. Your scores will be saved with its model and preference snapshot.';
}

function installGuide() {
  element('copy-upl').addEventListener('click', copyUpl);
  element('select-upl-json').addEventListener('click', () => {
    refreshUplHandoff();
    if (!handoffJson || element('upl-copy-fallback').hidden) return;
    element('upl-copy-json').focus();
    element('upl-copy-json').select();
  });
  element('allow-personalization').addEventListener('change', refreshUplHandoff);
  for (const button of document.querySelectorAll('[data-open-section]')) button.addEventListener('click', () => {
    document.querySelector(`.nav[data-section="${button.dataset.openSection}"]`).click();
    element(button.dataset.openSection).scrollIntoView({ block: 'start' });
  });
  for (const link of document.querySelectorAll('a[href="#build-tools"]')) link.addEventListener('click', () => {
    element('build-tools').open = true;
  });
  element('text-graph-preset').addEventListener('click', () => {
    element('show-canvas').click();
  }, { capture: true });
  element('load-hd-preview').addEventListener('click', loadHdPreview);
}

async function loadHdPreview() {
  const button = element('load-hd-preview');
  const status = element('hd-preview-status');
  const picture = element('hd-preview-image');
  button.disabled = true;
  picture.hidden = true;
  status.textContent = 'Checking for a verified, previously generated local image. No generation is being started.';
  try {
    const response = await fetch('/api/hd-preview');
    if (!response.ok) throw new Error('The read-only HD preview is not available on this server. The student workflow is unaffected.');
    const preview = await response.json();
    if (!preview.available) throw new Error('No verified HD example is installed. Nothing has been downloaded or generated.');
    if (preview.mode !== 'pretrained-baseline-not-campaign-trained' || preview.image !== '/api/hd-image'
      || ![preview.width, preview.height, preview.nativeWidth, preview.nativeHeight].every(value => Number.isInteger(value) && value > 0)) {
      throw new Error('The HD example has an unrecognized provenance contract. No image is displayed.');
    }
    picture.src = preview.image;
    await picture.decode();
    picture.width = preview.width;
    picture.height = preview.height;
    picture.hidden = false;
    status.textContent = `Saved pretrained baseline · ${preview.nativeWidth} × ${preview.nativeHeight} native, cropped to ${preview.width} × ${preview.height}. Not campaign-trained, not the current graph’s output, and not learned from your grades.`;
    button.textContent = 'Refresh saved example';
  } catch (error) {
    status.textContent = error.message;
    picture.removeAttribute('src');
    button.textContent = 'Check for saved HD example';
  } finally {
    button.disabled = false;
  }
}

initialize().then(refreshGuide).catch(error => {
  element('guide-error').hidden = false;
  element('guide-error').textContent = `The local lab could not load: ${error.message}. Reload after checking the local server. No model result is available.`;
  element('result-state-title').textContent = 'The local lab is unavailable';
  element('result-state-detail').textContent = 'The saved recipe and model could not be checked. No successful generation is assumed.';
  notice(`Initialization failed: ${error.message}`, true);
});
