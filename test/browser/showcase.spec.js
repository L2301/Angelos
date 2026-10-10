import { test, expect } from '@playwright/test';

test('showcase explains the system without API calls and adapts to small screens', async ({ page }) => {
  const errors = [];
  const apiRequests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.goto('/');
  await page.keyboard.press('Tab');
  await expect(page.locator('.skip-link')).toBeFocused();
  await expect(page.locator('.skip-link')).toHaveCSS('opacity', '1');
  await page.keyboard.press('Tab');
  await expect(page).toHaveTitle(/GPU Accelerated Ads — Louis Anderson/);
  await expect(page.locator('#hero-title')).toContainText('that campaign.');
  await expect(page.locator('#thesis .plain-steps li')).toHaveCount(5);
  await expect(page.locator('#thesis')).toContainText('supported ComfyUI API export');
  await expect(page.locator('#thesis')).toContainText('This training step is separate from converting the graph to code');
  await page.locator('[data-step="2"]').click();
  await expect(page.locator('#stage-2')).toBeVisible();
  await expect(page.locator('#stage-0')).toBeHidden();
  await page.locator('[data-step="4"]').click();
  await expect(page.locator('#stage-4')).toContainText('Implemented / candidate model training');
  await expect(page.locator('#stage-4')).toContainText('both candidates failed their gates');
  await expect(page.locator('.system-diagram')).not.toContainText('PLANNED');
  await expect(page.locator('.system-diagram')).toContainText('MODEL UPDATE');
  await expect(page.locator('.feedback-update-wire')).toHaveAttribute('d', 'M302 150V112H455V86');
  await expect(page.locator('.feedback-update-wire')).toHaveCSS('stroke-dasharray', 'none');
  await expect(page.locator('.diagram-feedback-note')).toContainText('manual, local click + critic training');
  await page.locator('.system-diagram').screenshot({ path: 'test-results/feedback-loop-diagram.png' });
  await page.locator('button[data-persona="bold"]').click();
  await expect(page.locator('#campaign-preview')).toHaveAttribute('data-persona', 'bold');
  await expect(page.locator('#campaign-headline')).toContainText('Go where');
  await page.locator('.hd-showcase').scrollIntoViewIfNeeded();
  await expect(page.locator('.aurora-ad-image')).toBeVisible();
  await expect.poll(() => page.locator('.aurora-ad-image').evaluate(image => [image.naturalWidth, image.naturalHeight])).toEqual([1920, 1080]);
  await expect(page.locator('.aurora-ad-headline')).toHaveText('Make roomfor wonder.');
  await expect(page.locator('#hd-showcase-boundary')).toContainText('not output from the campaign-trained student');
  await expect(page.locator('.prototype-archive')).toHaveCount(0);
  await expect(page.locator('#evidence')).not.toContainText('Earlier 32px browser prototype');
  await expect(page.locator('#evidence img')).toHaveCount(1);
  await page.locator('.hd-showcase').screenshot({ path: 'test-results/hd-showcase-desktop.png' });
  await page.screenshot({ path: 'test-results/showcase-desktop.png', fullPage: true });
  for (const width of [320, 390, 760, 1024, 1920]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('.hd-showcase').screenshot({ path: 'test-results/hd-showcase-mobile.png' });
  await page.locator('.menu-toggle').click();
  await expect(page.locator('.menu-toggle')).toHaveAttribute('aria-expanded', 'true');
  await page.keyboard.press('Escape');
  await expect(page.locator('.menu-toggle')).toBeFocused();
  await expect(page.locator('.menu-toggle')).toHaveAttribute('aria-expanded', 'false');
  await page.screenshot({ path: 'test-results/showcase-mobile.png', fullPage: true });
  expect(errors).toEqual([]);
  expect(apiRequests).toEqual([]);
});

test('core explanation remains readable without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  try {
    const page = await context.newPage();
    await page.goto('http://127.0.0.1:4311/');
    await expect(page.locator('#thesis')).toContainText('What this actually does');
    for (let index = 0; index < 5; index += 1) await expect(page.locator(`#stage-${index}`)).toBeVisible();
    await expect(page.locator('.persona-controls')).toBeHidden();
    await page.locator('.hd-showcase').scrollIntoViewIfNeeded();
    await expect(page.locator('.aurora-ad-headline')).toBeVisible();
    await expect.poll(() => page.locator('.aurora-ad-image').evaluate(image => image.naturalWidth)).toBe(1920);
    await expect(page.locator('#hd-showcase-boundary')).toContainText('Pretrained baseline');
    await expect(page.locator('.prototype-archive')).toHaveCount(0);
    await expect(page.getByRole('link', { name: 'Explore the laboratory' })).toHaveAttribute('href', '/lab');
  } finally { await context.close(); }
});

test('missing student is recoverable; text is explicit and separate from prompt', async ({ page }) => {
  const requests = [];
  await page.route('**/api/artifact', route => route.fulfill({ status: 404, json: { error: 'No trained artifact. Build a student first.' } }));
  await page.route('**/api/text-preview', async route => {
    requests.push(route.request().postDataJSON());
    await route.fulfill({ json: { text: '<fixture> explicit UI test, not model evidence', model: { id: 'qwen3-0.6b', revision: 'c1899de289a04d12100db370d81485cdf75e47ca' }, elapsedMilliseconds: 42, finish_reason: 'stop' } });
  });
  const state = await (await page.request.get('/api/state')).json();
  try {
    await page.goto('/lab');
    await expect(page.locator('#graph-count')).toContainText('typed nodes');
    await expect(page.locator('#show-placement')).toBeDisabled();
    expect(requests).toHaveLength(0);
    await page.locator('#text-tools > summary').click();
    await page.locator('#generate-text').click();
    await expect(page.locator('#text-result')).toContainText('<fixture>');
    expect(await page.locator('#text-result').locator('fixture').count()).toBe(0);
    expect(requests).toHaveLength(1);
    expect(requests[0].model).toBe('qwen3-0.6b');
    await page.locator('#text-graph-preset').click();
    await expect(page.locator('#notice')).toContainText('Graph loaded only');
    const updated = await (await page.request.get('/api/state')).json();
    expect(updated.graph.nodes.some(node => node.op === 'prompt')).toBe(true);
    expect(updated.graph.nodes.some(node => node.op === 'generate_text')).toBe(true);
    await page.locator('#build-tools > summary').click();
    await page.locator('#training-help').click();
    await expect(page.locator('#training-command')).toContainText('download qwen3-0.6b');
    await expect(page.locator('#training-command')).toContainText('download tiny-sd');
    await page.locator('#close-training').click();
    await page.getByRole('button', { name: 'Source', exact: true }).click();
    await expect(page.locator('#restore-source')).toBeEnabled();
    await page.locator('#restore-source').click();
    await expect(page.locator('#source')).toHaveValue(state.source);
    await page.setViewportSize({ width: 390, height: 844 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  } finally { await page.request.post('/api/graph', { data: { source: state.source } }); }
});
