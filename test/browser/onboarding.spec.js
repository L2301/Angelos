import { test, expect } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  const state = await (await page.request.get('/api/state')).json();
  const profile = state.profiles[0];
  await page.request.post('/api/consent', { data: { profileId: profile.id, campaignId: state.graph.id, personalization: true, telemetry: true, training: true } });
});

test('ComfyUI leads, the built-in graph is visible, and internals are optional', async ({ page }) => {
  const errors = [];
  const writes = [];
  const hdRequests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => {
    if (request.method() === 'POST' && !request.url().endsWith('/api/snapshot')) writes.push(request.url());
    if (request.url().includes('/api/hd-')) hdRequests.push(request.url());
  });
  await page.goto('/lab');
  await expect(page.locator('#lab-explanation')).toContainText('Saving a rating does not update the model');
  await expect(page.locator('#lab-explanation li')).toHaveCount(5);
  await expect(page.getByRole('link', { name: 'Build your ad in ComfyUI' })).toHaveAttribute('href', '/docs/COMFYUI.md');
  await expect(page.locator('#build-recipe')).toContainText('CLIP text and typed scalar inputs');
  await expect(page.getByRole('link', { name: 'Set up the ComfyUI plugin & UPL nodes' })).toHaveAttribute('href', '/docs/COMFYUI.md');
  await expect(page.locator('#canvas-area')).toBeVisible();
  await expect(page.locator('#graph .graph-node')).not.toHaveCount(0);
  await expect(page.locator('#source-area')).toBeHidden();
  await expect(page.locator('#text-form')).toBeHidden();
  await expect(page.locator('#recipe-tools')).not.toHaveAttribute('open');
  expect(await page.locator('#build-recipe').evaluate(recipe => recipe.compareDocumentPosition(document.getElementById('choose-person')) & Node.DOCUMENT_POSITION_FOLLOWING)).toBeTruthy();
  await page.getByRole('button', { name: 'Inspect person: upl' }).click();
  await expect(page.locator('#inspector')).toContainText('No personality is inferred');
  await expect(page.locator('#see-result')).toContainText('not proof the model learned it');
  await expect(page.locator('#next-version')).toContainText('No automatic replacement');
  await page.evaluate(() => { document.activeElement?.blur(); window.scrollTo(0, 0); });
  await page.screenshot({ path: 'test-results/onboarding-desktop.png', fullPage: true });
  for (const width of [320, 390, 760, 1024, 1920]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: 'test-results/onboarding-mobile.png', fullPage: true });
  expect(writes).toEqual([]);
  expect(hdRequests).toEqual([]);
  expect(errors).toEqual([]);
});

test('missing model remains explained without hiding the recipe or claiming HD generation', async ({ page }) => {
  await page.route('**/api/artifact', route => route.fulfill({ status: 404, json: { error: 'No trained artifact' } }));
  await page.route('**/api/hd-preview', route => route.fulfill({ json: { available: false } }));
  await page.goto('/lab');
  await expect(page.locator('#result-state-title')).toHaveText('No usable trained model is loaded');
  await expect(page.locator('#result-state-detail')).toContainText('nothing trains automatically');
  await expect(page.locator('#show-placement')).toBeDisabled();
  await expect(page.locator('#new-seed')).toBeDisabled();
  await page.locator('#save-review').click();
  await expect(page.locator('#notice')).toContainText('saved profile');
  await page.locator('#load-hd-preview').click();
  await expect(page.locator('#hd-preview-status')).toContainText('No verified HD example is installed');
  await expect(page.locator('#hd-preview-image')).toBeHidden();
  await page.locator('#result-help').click();
  await expect(page.locator('#training-help')).toBeVisible();
  await page.locator('#training-help').click();
  await expect(page.locator('#training-dialog')).toBeVisible();
  await expect(page.locator('#training-dialog')).toContainText('Nothing runs from this dialog');
  await page.locator('#close-training').click();
  await expect(page.locator('#result-state-title')).toHaveText('No usable trained model is loaded');
});

test('missing WebGPU has a persistent honest state and preferences stay usable', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true }));
  await page.goto('/lab');
  await expect(page.locator('#result-state-title')).toHaveText('The browser GPU is unavailable');
  await expect(page.locator('#result-state-detail')).toContainText('not a CPU substitute');
  await expect(page.locator('#show-placement')).toBeDisabled();
  await page.locator('#energy').fill('0.82');
  await expect(page.locator('#profile-save-status')).toContainText('Unsaved preferences');
  await expect(page.locator('#result-state-title')).toHaveText('The browser GPU is unavailable');
  await expect(page.locator('#canvas-area')).toBeVisible();
});

test('HD example reads only on request and stays separate from the student', async ({ page }) => {
  const reads = [];
  await page.route('**/api/hd-preview', route => {
    reads.push(route.request().method());
    return route.fulfill({ json: { available: true, image: '/api/hd-image', width: 1920, height: 1080, nativeWidth: 1920, nativeHeight: 1088, mode: 'pretrained-baseline-not-campaign-trained' } });
  });
  await page.route('**/api/hd-image', route => route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080"><rect width="1920" height="1080" fill="#31553a"/><text x="40" y="80" fill="white">UI test fixture, not model evidence</text></svg>' }));
  await page.goto('/lab');
  expect(reads).toEqual([]);
  await page.locator('#load-hd-preview').click();
  await expect(page.locator('#hd-preview-image')).toBeVisible();
  await expect(page.locator('#hd-preview-status')).toContainText('1920 × 1088 native, cropped to 1920 × 1080');
  await expect(page.locator('#hd-preview-status')).toContainText('Not campaign-trained');
  expect(reads).toEqual(['GET']);
  await expect(page.locator('#student')).toBeVisible();
  await expect(page.locator('#student-caption')).toContainText('native 32×32');
});

test('local server failure leaves an actionable explanation, not a blank success state', async ({ page }) => {
  await page.route('**/api/state', route => route.fulfill({ status: 503, json: { error: 'Test server unavailable' } }));
  await page.goto('/lab');
  await expect(page.locator('#guide-error')).toContainText('Reload after checking the local server');
  await expect(page.locator('#result-state-title')).toHaveText('The local lab is unavailable');
  await expect(page.locator('#show-placement')).toBeDisabled();
  await expect(page.getByRole('link', { name: 'Build your ad in ComfyUI' })).toBeVisible();
});

test('copy UPL uses the exact frozen parser contract without IDs, hashes or network writes', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
    writeText: async value => { window.copiedUplJson = value; },
  } }));
  const writes = [];
  page.on('request', request => { if (request.method() === 'POST') writes.push(request.url()); });
  const snapshotResponse = page.waitForResponse('**/api/snapshot');
  await page.goto('/lab');
  const snapshot = await (await snapshotResponse).json();
  await expect(page.locator('#copy-upl')).toBeEnabled();
  const beforeCopy = [...writes];
  await page.locator('#copy-upl').click();
  await expect(page.locator('#upl-copy-status')).toContainText('Saved UPL copied');
  const payload = JSON.parse(await page.evaluate(() => window.copiedUplJson));
  expect(payload).toEqual({ format: 'angelos.upl-input.v1', synthetic: true, adult: true, factsSchema: 'angelos.facts.v1',
    facts: snapshot.facts, encoder: { id: 'structured-demo-v1', dimension: 4 }, upl: snapshot.vector });
  expect(Object.keys(payload).sort()).toEqual(['adult', 'encoder', 'facts', 'factsSchema', 'format', 'synthetic', 'upl']);
  expect(writes).toEqual(beforeCopy);
  expect(writes).toHaveLength(1);
  expect(writes[0]).toMatch(/\/api\/snapshot$/);
  await expect(page.locator('#upl-copy-fallback')).toBeHidden();
});

test('clipboard denial provides selectable JSON, which edits and revoked consent invalidate', async ({ page }) => {
  await page.addInitScript(() => {
    window.copyAttempts = 0;
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
      writeText: async () => { window.copyAttempts += 1; throw new DOMException('Denied by test', 'NotAllowedError'); },
    } });
  });
  await page.goto('/lab');
  await expect(page.locator('#copy-upl')).toBeEnabled();
  const energy = await page.locator('#energy').inputValue();
  const profileId = await page.locator('#profile').inputValue();
  const state = await (await page.request.get('/api/state')).json();
  try {
    await page.locator('#copy-upl').click();
    await expect(page.locator('#upl-copy-fallback')).toBeVisible();
    await expect(page.locator('#upl-copy-status')).toContainText('Clipboard unavailable');
    await page.locator('#select-upl-json').click();
    expect(await page.locator('#upl-copy-json').evaluate(input => input.readOnly && input.selectionStart === 0 && input.selectionEnd === input.value.length && input.value.length > 0)).toBe(true);
    await page.locator('#energy').fill(energy === '0.91' ? '0.92' : '0.91');
    await expect(page.locator('#copy-upl')).toBeDisabled();
    await expect(page.locator('#upl-copy-status')).toContainText('Unsaved changes cannot be exported');
    await expect(page.locator('#upl-copy-fallback')).toBeHidden();
    await expect(page.locator('#upl-copy-json')).toHaveValue('');
    await page.locator('#copy-upl').dispatchEvent('click');
    expect(await page.evaluate(() => window.copyAttempts)).toBe(1);
    await page.locator('#energy').fill(energy);
    await expect(page.locator('#copy-upl')).toBeEnabled();
    await page.getByRole('button', { name: 'People & permissions', exact: true }).click();
    await page.locator('#allow-personalization').uncheck();
    await expect(page.locator('#copy-upl')).toBeDisabled();
    await page.locator('#save-consent').click();
    await expect(page.locator('#notice')).toContainText('Consent updated');
    await page.getByRole('button', { name: 'Build an ad', exact: true }).click();
    await expect(page.locator('#copy-upl')).toBeDisabled();
    await expect(page.locator('#upl-copy-status')).toContainText('Personalization permission is required');
    await page.locator('#copy-upl').dispatchEvent('click');
    expect(await page.evaluate(() => window.copyAttempts)).toBe(1);
  } finally {
    await page.request.post('/api/consent', { data: { profileId, campaignId: state.graph.id, personalization: true, telemetry: true, training: true } });
  }
});

test('mismatched frozen facts are refused rather than rebuilding a UPL from live inputs', async ({ page }) => {
  await page.route('**/api/snapshot', async route => {
    const response = await route.fetch();
    const snapshot = await response.json();
    snapshot.facts.energy = snapshot.facts.energy === 0 ? 1 : 0;
    await route.fulfill({ response, json: snapshot });
  });
  await page.goto('/lab');
  await expect(page.locator('#upl-copy-status')).toContainText('frozen facts and UPL do not match exactly');
  await expect(page.locator('#copy-upl')).toBeDisabled();
  await page.locator('#copy-upl').dispatchEvent('click');
  await expect(page.locator('#upl-copy-fallback')).toBeHidden();
});

test('late clipboard denial cannot reveal a snapshot invalidated by unsaved preferences', async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
    writeText: () => new Promise((resolve, reject) => { window.rejectUplCopy = reject; }),
  } }));
  await page.goto('/lab');
  await expect(page.locator('#copy-upl')).toBeEnabled();
  await page.locator('#copy-upl').click();
  await expect(page.locator('#upl-copy-status')).toContainText('Copying the saved snapshot');
  const energy = await page.locator('#energy').inputValue();
  await page.locator('#energy').fill(energy === '0.91' ? '0.92' : '0.91');
  await page.evaluate(() => window.rejectUplCopy(new Error('Delayed clipboard denial')));
  await expect(page.locator('#upl-copy-status')).toContainText('Unsaved changes cannot be exported');
  await expect(page.locator('#copy-upl')).toBeDisabled();
  await expect(page.locator('#upl-copy-json')).toHaveValue('');
  await expect(page.locator('#upl-copy-fallback')).toBeHidden();
});
