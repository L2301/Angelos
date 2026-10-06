import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';

test('laboratory loads typed graph, rejects invalid source, and remains usable on narrow screens', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#graph-count')).toContainText('typed nodes');
  await expect(page.locator('#profile option')).toHaveCount(3);
  await expect(page.locator('#gpu-status')).toContainText('WebGPU student · active');
  await expect(page.locator('#render-time')).not.toHaveText('—');
  await page.screenshot({ path: 'test-results/laboratory-desktop.png', fullPage: true });
  await page.getByRole('button', { name: 'Source', exact: true }).click();
  await page.locator('#source').fill('campaign broken\noutput missing');
  await page.getByRole('button', { name: 'Check & apply' }).click();
  await expect(page.locator('#notice')).toBeVisible();
  await page.setViewportSize({ width: 430, height: 932 });
  await expect(page.getByRole('heading', { name: 'One graph. One creative model.' })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  expect(errors).toEqual([]);
});

test('real browser GPU matches the trained PyTorch fixture and CPU reference', async ({ page }) => {
  const pointer = JSON.parse(await readFile(resolve('artifacts/preview.json'), 'utf8'));
  const fixturePath = resolve('artifacts', pointer.file.replace(/\.json$/, '.parity.json'));
  const fixture = JSON.parse(await readFile(fixturePath, 'utf8'));
  test.skip(fixture.nativeResolution === 32, 'Diffusion numerical parity is exercised by diffusion.spec.js');
  await page.goto('/');
  const result = await page.evaluate(async fixture => {
    const { createGpuRenderer, renderCpu } = await import('/src/browser/student.js');
    const { createReference } = await import('/src/core/graph.js');
    const artifact = await (await fetch('/api/artifact')).json();
    const state = await (await fetch('/api/state')).json();
    const reference = createReference(state.graph);
    const renderer = await createGpuRenderer(artifact);
    let gpuError = 0;
    let cpuError = 0;
    let teacherError = 0;
    let brandError = 0;
    const timings = [];
    try {
      for (const raster of fixture.rasterCases) {
        for (const composite of [false, true]) {
          const expected = composite ? raster.studentRgba : raster.rawStudentRgba;
          const cpu = renderCpu(artifact, raster.upl, raster.width, raster.height, { composite });
          const gpu = await renderer.render(raster.upl, raster.width, raster.height, { composite });
          timings.push(gpu.milliseconds);
          for (let index = 0; index < expected.length; index += 1) {
            gpuError = Math.max(gpuError, Math.abs(gpu.pixels[index] - expected[index]));
            cpuError = Math.max(cpuError, Math.abs(cpu[index] - expected[index]));
            if (composite && Math.abs(raster.studentRgba[index] - raster.rawStudentRgba[index]) > 0.001) brandError = Math.max(brandError, Math.abs(gpu.pixels[index] - expected[index]));
          }
        }
        for (let row = 0; row < raster.height; row += 1) {
          for (let column = 0; column < raster.width; column += 1) {
            const color = reference(raster.upl, (column + 0.5) / raster.width, (row + 0.5) / raster.height);
            for (let channel = 0; channel < 3; channel += 1) teacherError = Math.max(teacherError, Math.abs(color[channel] - raster.teacherRgba[(row * raster.width + column) * 4 + channel]));
          }
        }
      }
    } finally { renderer.dispose(); }
    return { gpuError, cpuError, teacherError, brandError, timings, artifactId: artifact.id };
  }, fixture);
  console.log('Actual browser numerical parity:', JSON.stringify(result));
  expect(result.gpuError).toBeLessThan(0.001);
  expect(result.cpuError).toBeLessThan(0.001);
  expect(result.teacherError).toBeLessThan(0.0001);
  expect(result.brandError).toBeLessThan(0.0001);
});

test('saved UPL drives GPU creative, qualified exposure, click and multidimensional human review', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#gpu-status')).toContainText('WebGPU student · active', { timeout: 30000 });
  await page.locator('#delivered').scrollIntoViewIfNeeded();
  await expect(page.locator('#show-placement')).toBeEnabled({ timeout: 30000 });
  await page.locator('#show-placement').click();
  await expect(page.locator('#click-ad')).toBeEnabled({ timeout: 10000 });
  await page.locator('#click-ad').click();
  await expect(page.locator('#exposure-status')).toContainText('Click attributed');
  await page.locator('#save-review').click();
  await expect(page.locator('#notice')).toContainText('Review saved');
  await page.getByRole('button', { name: 'Learning ledger' }).click();
  await expect(page.locator('#ledger-json')).toContainText('"clicks": 1');
  await expect(page.locator('#ledger-json')).toContainText('"reviews": 1');
});

test('hidden creative cannot qualify and unsaved UPL changes cannot be reviewed', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#gpu-status')).toContainText('WebGPU student · active');
  await page.locator('#delivered').scrollIntoViewIfNeeded();
  await expect(page.locator('#show-placement')).toBeEnabled();
  const before = await (await page.request.get('/api/metrics')).json();
  await page.locator('#show-placement').click();
  await page.getByRole('button', { name: 'Personal latents' }).click();
  await page.waitForTimeout(1300);
  const after = await (await page.request.get('/api/metrics')).json();
  expect(after.impressions).toBe(before.impressions);
  await page.getByRole('button', { name: 'Campaign studio' }).click();
  await page.locator('#energy').fill('0.91');
  await page.locator('#energy').dispatchEvent('input');
  await page.locator('#save-review').click();
  await expect(page.locator('#notice')).toContainText('saved profile');
});

test('measure bounded 192-pixel inference on the actual adapter', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#render-time')).not.toHaveText('—');
  const result = await page.evaluate(async () => {
    const { createGpuRenderer } = await import('/src/browser/student.js');
    const adapter = await navigator.gpu.requestAdapter();
    const adapterInfo = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device, description: adapter.info.description, isFallbackAdapter: adapter.info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null };
    const bytes = await (await fetch('/api/artifact')).text();
    const artifact = JSON.parse(bytes);
    const started = performance.now();
    const renderer = await createGpuRenderer(artifact);
    const initializationMilliseconds = performance.now() - started;
    const durations = [];
    try {
      for (let index = 0; index < 23; index += 1) {
        const output = await renderer.render([0.2, 0, 0.72, 0.38], 192, 192);
        if (index >= 3) durations.push(output.milliseconds);
      }
    } finally { renderer.dispose(); }
    durations.sort((left, right) => left - right);
    return { adapterInfo, initializationMilliseconds, jsonBytes: new TextEncoder().encode(bytes).length, width: 192, height: 192, samples: durations.length, p50Milliseconds: durations[Math.floor(durations.length * 0.5)], p95Milliseconds: durations[Math.floor(durations.length * 0.95)], note: 'Includes GPU completion/readback; excludes display and network cold start; no CI timing assertion.' };
  });
  console.log('Measured browser benchmark:', JSON.stringify(result));
  expect(result.samples).toBe(20);
  expect(result.adapterInfo.isFallbackAdapter).not.toBe(true);
  expect(result.p50Milliseconds).toBeGreaterThan(0);
});

test('new diffusion variations invalidate exposure and generative presets never run a hidden teacher', async ({ page }) => {
  const artifact = await (await page.request.get('/api/artifact')).json();
  test.skip(artifact.format !== 'angelos.diffusion.v1', 'Variation behavior requires the learned diffusion student');
  const state = await (await page.request.get('/api/state')).json();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#show-placement')).toBeEnabled();
  const previousSeed = await page.locator('#creative-seed').textContent();
  const previousImage = await page.locator('#delivered').evaluate(canvas => canvas.toDataURL());
  await page.locator('#new-seed').click();
  await expect(page.locator('#creative-seed')).not.toHaveText(previousSeed);
  await expect(page.locator('#show-placement')).toBeEnabled();
  expect(await page.locator('#delivered').evaluate(canvas => canvas.toDataURL())).not.toBe(previousImage);
  await expect(page.locator('#click-ad')).toBeDisabled();
  for (const model of ['flux2-klein-4b', 'ltx-video']) {
    await page.locator('#teacher-example').selectOption(model);
    const applied = page.waitForResponse(response => response.url().endsWith('/api/graph') && response.request().method() === 'POST');
    await page.locator('#load-example').click();
    await applied;
    await expect(page.locator('#notice')).toContainText('Graph loaded only');
    await expect(page.locator('#teacher-backend')).toHaveText('Offline open-weight teacher');
    await expect(page.locator('#show-placement')).toBeDisabled();
    const current = await (await page.request.get('/api/state')).json();
    expect(current.graph.nodes.some(node => node.params.model === model)).toBe(true);
    expect(current.artifactCompatible).toBe(false);
  }
  await page.getByRole('button', { name: 'Source', exact: true }).click();
  await page.locator('#source').fill(state.source);
  await page.getByRole('button', { name: 'Check & apply' }).click();
  await expect(page.locator('#notice')).toContainText('Graph checked and saved');
  await expect(page.locator('#show-placement')).toBeEnabled();
  expect(errors).toEqual([]);
});

test('revocation stops rendering and a failed profile switch clears previous creatives', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#gpu-status')).toContainText('WebGPU student · active');
  await expect(page.locator('#render-time')).not.toHaveText('—');
  const previousImage = await page.locator('#delivered').evaluate(canvas => canvas.toDataURL());
  const state = await (await page.request.get('/api/state')).json();
  const selected = await page.locator('#profile').inputValue();
  const nextProfile = state.profiles.find(profile => profile.id !== selected);
  await page.route('**/api/snapshot', route => route.fulfill({ status: 400, contentType: 'application/json', body: JSON.stringify({ error: 'Test: missing authorized profile ingredients' }) }));
  await page.locator('#profile').selectOption(nextProfile.id);
  await expect(page.locator('#notice')).toContainText('missing authorized');
  expect(await page.locator('#delivered').evaluate(canvas => canvas.toDataURL())).not.toBe(previousImage);
  await expect(page.locator('#show-placement')).toBeDisabled();
  await page.unroute('**/api/snapshot');
  await page.locator('#profile').selectOption(selected);
  await expect(page.locator('#show-placement')).toBeEnabled();
  await page.getByRole('button', { name: 'Personal latents' }).click();
  await page.locator('#allow-personalization').uncheck();
  await page.locator('#save-consent').click();
  await expect(page.locator('#notice')).toContainText('Consent updated');
  await page.getByRole('button', { name: 'Campaign studio' }).click();
  const deniedImage = await page.locator('#delivered').evaluate(canvas => canvas.toDataURL());
  await page.locator('#energy').fill('0.1');
  await page.locator('#energy').dispatchEvent('input');
  await page.waitForTimeout(150);
  expect(await page.locator('#delivered').evaluate(canvas => canvas.toDataURL())).toBe(deniedImage);
  await expect(page.locator('#show-placement')).toBeDisabled();
});
