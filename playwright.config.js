import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './test/browser',
  timeout: 60000,
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:4311', viewport: { width: 1440, height: 1100 }, browserName: 'chromium', channel: 'chrome', launchOptions: { args: ['--enable-unsafe-webgpu'] }, trace: 'retain-on-failure' },
  webServer: { command: 'node scripts/test-server.js', url: 'http://127.0.0.1:4311/api/health', reuseExistingServer: false, timeout: 15000 },
});
