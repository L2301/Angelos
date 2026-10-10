import { importComfyWorkflow, compileComfyProgram } from '/src/core/comfy-program.js';

const element = id => document.getElementById(id);
let currentSource = '';
let inputRevision = 0;

function invalidate() {
  inputRevision += 1;
  currentSource = '';
  element('comfy-source').value = '';
  element('comfy-result').hidden = true;
  element('comfy-status').textContent = 'Input changed. Check & convert again; the previous translation is no longer available.';
}

element('comfy-json').addEventListener('input', invalidate);
element('comfy-file').addEventListener('change', async () => {
  invalidate();
  const revision = inputRevision;
  const file = element('comfy-file').files[0];
  if (!file) return;
  element('comfy-json').value = '';
  try {
    if (file.size > 262144) throw new Error('Choose a JSON file no larger than 256 KiB.');
    const source = await file.text();
    if (revision !== inputRevision) return;
    element('comfy-json').value = source;
    element('comfy-status').textContent = 'File read locally. Choose Check & convert to validate it.';
  } catch (error) { if (revision === inputRevision) element('comfy-status').textContent = error.message; }
});

element('comfy-example').addEventListener('click', async () => {
  invalidate();
  const revision = inputRevision;
  try {
    const response = await fetch('/examples/aurora-comfy-api.json');
    if (!response.ok) throw new Error('Example unavailable. Paste your API export instead.');
    const source = await response.text();
    if (revision !== inputRevision) return;
    element('comfy-json').value = source;
    element('comfy-status').textContent = 'Example loaded, not executed. YOUR_LOCAL_CHECKPOINT is a placeholder, not installed weights. Choose Check & convert.';
  } catch (error) { if (revision === inputRevision) element('comfy-status').textContent = error.message; }
});

element('comfy-convert').addEventListener('click', () => {
  invalidate();
  try {
    currentSource = importComfyWorkflow(element('comfy-json').value);
    const graph = compileComfyProgram(currentSource);
    element('comfy-source').value = currentSource;
    element('comfy-result').hidden = false;
    element('comfy-status').textContent = `${Object.keys(graph).length} nodes converted and type-checked. No generation, training, upload or change to the built-in recipe.`;
  } catch (error) { element('comfy-status').textContent = `Not converted: ${error.message}`; }
});

function download(name, contents, type) {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

element('comfy-download').addEventListener('click', () => {
  if (currentSource) download('campaign.comfy.ang', currentSource, 'text/plain');
});
element('comfy-roundtrip').addEventListener('click', () => {
  if (currentSource) download('campaign-api.json', JSON.stringify(compileComfyProgram(currentSource), null, 2) + '\n', 'application/json');
});
