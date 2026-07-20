import { defineConfig, devices } from "@playwright/test";

// End-to-end tests for the dashboard, run against a real gateway serving the
// built bundle (booted by `webServer` below). Component behavior is covered by
// Vitest; this exercises the multi-page flows a browser actually walks.
export default defineConfig({
  testDir: "./e2e",
  // The flows mutate one shared gateway DB, so they run in order, not parallel.
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  // No retries: the serial flows share one gateway DB that serve.sh resets only
  // at server start, and Playwright does not restart the webServer between
  // retries. A retry would re-run the block against state left by the first
  // attempt (the provider/alias already exist) and fail deterministically.
  retries: 0,
  reporter: process.env.CI ? "list" : "line",
  use: {
    baseURL: "http://127.0.0.1:8000",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "bash e2e/serve.sh",
    url: "http://127.0.0.1:8000/health",
    // Opt-in only: by default always start a fresh gateway (serve.sh resets the
    // DB), so a stray server already on :8000 can't silently skip the reset and
    // leave the serial flows running against dirty state. Set
    // PLAYWRIGHT_REUSE_SERVER=1 for fast local iteration against a running one.
    reuseExistingServer: !!process.env.PLAYWRIGHT_REUSE_SERVER,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
