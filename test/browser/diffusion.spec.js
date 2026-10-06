import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';

async function learnedFixture() {
  const pointer = JSON.parse(await readFile(resolve('artifacts/preview.json'), 'utf8'));
  const artifact = JSON.parse(await readFile(resolve('artifacts', pointer.file), 'utf8'));
  if (artifact.format !== 'angelos.diffusion.v1') return null;
  try {
    const fixture = JSON.parse(await readFile(resolve('artifacts', pointer.file.replace(/\.json$/, '.parity.json')), 'utf8'));
    return { artifact, fixture };
  } catch (error) {
    if (error.code === 'ENOENT') return null;
    throw error;
  }
}

test('actual Chrome WebGPU DDIM16 matches every learned PyTorch raster and explicit CPU reference', async ({ page }) => {
  test.setTimeout(120000);
  const learned = await learnedFixture();
  test.skip(!learned, 'No learned diffusion artifact and ML parity fixture are available; never replace with a zero-weight GPU success.');
  const { artifact, fixture } = learned;
  expect(artifact.training.steps).toBeGreaterThan(0);
  expect(fixture.artifactId).toBe(artifact.id);
  expect(fixture.graphHash).toBe(artifact.graphHash);
  expect(fixture.rasterCases.length).toBeGreaterThan(0);
  await page.goto('/api/health');
  const result = await page.evaluate(async ({ artifact, fixture }) => {
    const { createGpuRenderer, renderCpu } = await import('/src/browser/student.js');
    const adapter = await navigator.gpu.requestAdapter();
    const adapterInfo = { vendor: adapter.info.vendor, architecture: adapter.info.architecture, description: adapter.info.description, isFallbackAdapter: adapter.info.isFallbackAdapter ?? adapter.isFallbackAdapter ?? null };
    const renderer = await createGpuRenderer(artifact);
    const errors = { gpu: 0, cpu: 0, gpuCpu: 0, brand: 0, bytes: 0 };
    const timings = [];
    const seeds = [];
    let replayError = 0;
    let differentSeedError = 0;
    let scaledError = 0;
    let singleFlight = false;
    let disposed = false;
    function compare(actual, expected, name) {
      if (actual.length !== expected.length || !actual.every(Number.isFinite)) throw new Error(`Invalid ${name} raster`);
      for (let index = 0; index < actual.length; index += 1) errors[name] = Math.max(errors[name], Math.abs(actual[index] - expected[index]));
    }
    try {
      for (const raster of fixture.rasterCases) {
        const seed = raster.seed ?? 2026;
        seeds.push(seed);
        for (const composite of [false, true]) {
          const expected = composite ? raster.studentRgba : raster.rawStudentRgba;
          const cpu = renderCpu(artifact, raster.upl, raster.width, raster.height, { composite, seed });
          const gpu = await renderer.render(raster.upl, raster.width, raster.height, { composite, seed });
          if (gpu.seed !== seed || gpu.nativeResolution !== 32 || gpu.steps !== 16) throw new Error('Incorrect creative metadata');
          timings.push(gpu.milliseconds);
          compare(gpu.pixels, expected, 'gpu');
          compare(cpu, expected, 'cpu');
          compare(gpu.pixels, cpu, 'gpuCpu');
          const expectedBytes = composite ? raster.studentBytes : raster.rawStudentBytes;
          if (expectedBytes) {
            for (let index = 0; index < expectedBytes.length; index += 1) errors.bytes = Math.max(errors.bytes, Math.abs(Math.floor(Math.min(1, Math.max(0, gpu.pixels[index])) * 255 + 0.5) - expectedBytes[index]));
          }
          if (composite) {
            for (let index = 0; index < expected.length; index += 1) {
              if (Math.abs(raster.studentRgba[index] - raster.rawStudentRgba[index]) > 0.001) errors.brand = Math.max(errors.brand, Math.abs(gpu.pixels[index] - expected[index]));
            }
          }
        }
      }
      const raster = fixture.rasterCases[0];
      const seed = raster.seed ?? 2026;
      const pending = renderer.render(raster.upl, 32, 32, { seed, composite: false });
      try { await renderer.render(raster.upl, 32, 32); } catch (error) { singleFlight = /in flight/.test(error.message); }
      const first = await pending;
      const repeated = await renderer.render(raster.upl, 32, 32, { seed, composite: false });
      const different = await renderer.render(raster.upl, 32, 32, { seed: (seed + 1) >>> 0, composite: false });
      const scaled = await renderer.render(raster.upl, 67, 49, { seed, composite: false });
      for (let index = 0; index < first.pixels.length; index += 1) {
        replayError = Math.max(replayError, Math.abs(first.pixels[index] - repeated.pixels[index]));
        differentSeedError = Math.max(differentSeedError, Math.abs(first.pixels[index] - different.pixels[index]));
      }
      for (let row = 0; row < 49; row += 1) {
        for (let column = 0; column < 67; column += 1) {
          for (let channel = 0; channel < 4; channel += 1) {
            const source = (Math.floor(row * 32 / 49) * 32 + Math.floor(column * 32 / 67)) * 4 + channel;
            scaledError = Math.max(scaledError, Math.abs(scaled.pixels[(row * 67 + column) * 4 + channel] - first.pixels[source]));
          }
        }
      }
      const composite = await renderer.render(raster.upl, 67, 49, { seed });
      const brand = artifact.brand;
      for (let row = 0; row < 49; row += 1) {
        for (let column = 0; column < 67; column += 1) {
          const horizontal = Math.fround((column + 0.5) / 67);
          const vertical = Math.fround((row + 0.5) / 49);
          const plateColumn = brand ? Math.floor(Math.fround(Math.fround(horizontal - Math.fround(brand.x)) / Math.fround(brand.cell))) : -1;
          const plateRow = brand ? Math.floor(Math.fround(Math.fround(vertical - Math.fround(brand.y)) / Math.fround(brand.cell))) : -1;
          const onPlate = brand?.mode === 'composite' && horizontal >= Math.fround(brand.x) && horizontal < Math.fround(brand.x + 7 * brand.cell) && vertical >= Math.fround(brand.y) && vertical < Math.fround(brand.y + 9 * brand.cell);
          const ink = ['01110', '10001', '10001', '11111', '10001', '10001', '10001'][plateRow - 1]?.[plateColumn - 1] === '1';
          for (let channel = 0; channel < 3; channel += 1) {
            const index = (row * 67 + column) * 4 + channel;
            const expected = onPlate ? Math.fround((ink ? brand.color : brand.background)[channel]) : scaled.pixels[index];
            errors.brand = Math.max(errors.brand, Math.abs(composite.pixels[index] - expected));
          }
        }
      }
    } finally { renderer.dispose(); }
    try { await renderer.render(fixture.rasterCases[0].upl, 32, 32); } catch (error) { disposed = /disposed/.test(error.message); }
    return { adapterInfo, artifactId: artifact.id, errors, timings, seeds, replayError, differentSeedError, scaledError, singleFlight, disposed };
  }, learned);
  console.log('Actual Chrome diffusion parity:', JSON.stringify(result));
  expect(result.adapterInfo.isFallbackAdapter).not.toBe(true);
  const tolerance = Math.min(fixture.floatAbsoluteTolerance ?? 0.001, 0.001);
  expect(result.errors.gpu).toBeLessThanOrEqual(tolerance);
  expect(result.errors.cpu).toBeLessThanOrEqual(tolerance);
  expect(result.errors.gpuCpu).toBeLessThanOrEqual(tolerance);
  expect(result.errors.brand).toBeLessThan(0.000001);
  expect(result.errors.bytes).toBeLessThanOrEqual(1);
  expect(result.replayError).toBe(0);
  expect(result.differentSeedError).toBeGreaterThan(0);
  expect(result.scaledError).toBe(0);
  expect(result.singleFlight).toBe(true);
  expect(result.disposed).toBe(true);
  expect(result.timings.every(milliseconds => milliseconds > 0)).toBe(true);
});

test('actual Chrome compositor matches PyTorch float32 boundary masks without changing learned denoising', async ({ page }) => {
  const learned = await learnedFixture();
  test.skip(!learned, 'A learned diffusion artifact is required; no zero-weight browser fixture.');
  const expectedMasks = [
    [10, 10, '766c00f1d107d62324e476efb35778018a33f920e91beafce13c2099d70fbe35'],
    [20, 20, 'b3f0c852401131a2a48a78ebdf2f8a04f745a228038cb37ec5cded910b3eee9b'],
    [32, 32, '81240134d7047b0eb5b7877bebcd0290ba75c23747f8a7714ae546641c8de64e'],
    [67, 49, '84a43f33f9dc19eb3144a4a3078c366f0cd150b39ac9ee7397ab6a831c3ab2e2'],
    [192, 192, '7ebb5372bb5507e4df0dd3af5efce46ac7abf920600abb844dbc08cc9f3cf9df'],
  ];
  await page.goto('/api/health');
  const results = await page.evaluate(async ({ artifact, raster, dimensions }) => {
    const { createDiffusionGpuRenderer } = await import('/src/browser/diffusion.js');
    artifact.brand = { mode: 'composite', x: 0.1, y: 0.1, cell: 0.05, color: [0.2, 0.4, 0.9], background: [0.9, 0.1, 0.3] };
    artifact.metrics.brandMaxError ??= 0;
    const foreground = new Float32Array([...artifact.brand.color, 1]);
    const background = new Float32Array([...artifact.brand.background, 1]);
    const renderer = await createDiffusionGpuRenderer(artifact);
    const results = [];
    try {
      for (const [width, height] of dimensions) {
        const raw = await renderer.render(raster.upl, width, height, { seed: raster.seed ?? 2026, composite: false });
        const delivered = await renderer.render(raster.upl, width, height, { seed: raster.seed ?? 2026 });
        let mask = '';
        let outsideError = 0;
        for (let offset = 0; offset < delivered.pixels.length; offset += 4) {
          const pixel = delivered.pixels.subarray(offset, offset + 4);
          if (pixel.every((value, channel) => value === foreground[channel])) mask += 'i';
          else if (pixel.every((value, channel) => value === background[channel])) mask += 'b';
          else {
            mask += '.';
            for (let channel = 0; channel < 4; channel += 1) outsideError = Math.max(outsideError, Math.abs(pixel[channel] - raw.pixels[offset + channel]));
          }
        }
        results.push({ width, height, mask, outsideError });
      }
    } finally { renderer.dispose(); }
    return results;
  }, { artifact: learned.artifact, raster: learned.fixture.rasterCases[0], dimensions: expectedMasks.map(([width, height]) => [width, height]) });
  for (let index = 0; index < results.length; index += 1) {
    const result = results[index];
    expect(createHash('sha256').update(result.mask).digest('hex'), `${result.width}x${result.height} PyTorch mask`).toBe(expectedMasks[index][2]);
    expect(result.outsideError).toBe(0);
  }
});
