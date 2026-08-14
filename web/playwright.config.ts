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
  // "list" everywhere, not "line" locally: the gateway's own request log is piped
  // into this stream (see `webServer.stdout`), and the line reporter rewrites its
  // single status line with carriage returns, so interleaving the two shreds the
  // failure detail exactly when it is needed.
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:8000",
    // Not "on-first-retry": retries are off by design (see above), so that mode
    // never fires and a CI failure arrives with no trace to open. Retaining on
    // failure keeps the artifact for the run that needs it and discards it for
    // every run that passes.
    trace: "retain-on-failure",
  },
  // Three ordered projects over one gateway, rather than one project relying on
  // the alphabetical order of filenames. `onboarding` needs the empty database
  // serve.sh leaves behind (it asserts the first-run screens), so anything that
  // writes usage has to come after it; the parity specs in turn all read one
  // seeded fixture, so they hang off the seed rather than each re-creating it.
  projects: [
    {
      name: "onboarding",
      testMatch: /dashboard\.spec\.ts/,
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "seed",
      testMatch: /parity\.setup\.ts/,
      dependencies: ["onboarding"],
      use: { ...devices["Desktop Chrome"] },
    },
    {
      name: "parity",
      // Everything that is not the onboarding spec, rather than a `parity.*`
      // pattern: a project's testMatch is the only thing that collects a file, so
      // a spec named outside every pattern here runs in no project at all and is
      // dropped from the run silently, with no warning and a green exit.
      testIgnore: /dashboard\.spec\.ts/,
      dependencies: ["seed"],
      use: { ...devices["Desktop Chrome"] },
    },
  ],
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
