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
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? "list" : "line",
  use: {
    baseURL: "http://127.0.0.1:8000",
    trace: "on-first-retry",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "bash e2e/serve.sh",
    url: "http://127.0.0.1:8000/health",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
