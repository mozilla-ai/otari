import { defineConfig, devices } from "@playwright/test"

import { HYBRID_BASE_URL } from "./e2e/hybrid"

// End-to-end tests for the dashboard, run against a real gateway serving the
// built bundle (booted by `webServer` below). Component behavior is covered by
// Vitest; this exercises the multi-page flows a browser actually walks.

// The visual-regression matrix, ported from otari-ai/frontend's screenshot
// suite so the two produce comparable captures and neither has to rediscover
// what makes them stable. Widths are theirs, and so is the mobile descriptor:
// `isMobile`/`hasTouch`/`deviceScaleFactor` layered on Desktop Chrome renders
// the page as a real phone does (touch guards, viewport meta interpretation,
// `:hover` rules) rather than as a desktop browser resized down.
// `devices["iPhone N"]` is deliberately not used: those default to webkit, and
// one engine across all six projects is what keeps baselines comparable.
const SCREENSHOT_VIEWPORTS = {
  "desktop-large": {
    ...devices["Desktop Chrome"],
    viewport: { width: 1920, height: 1080 },
  },
  "desktop-small": {
    ...devices["Desktop Chrome"],
    viewport: { width: 1280, height: 800 },
  },
  mobile: {
    ...devices["Desktop Chrome"],
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
    deviceScaleFactor: 3,
    userAgent:
      "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
  },
} as const

const SCREENSHOT_THEMES = ["light", "dark"] as const

// The second gateway this suite boots, in hybrid mode (e2e/serve-hybrid.sh). The
// port is e2e/otari.hybrid.yml's, and the host is 127.0.0.1 rather than
// localhost so the page's own `window.location.origin` matches this string,
// which the hybrid spec asserts against.
// One project per cell. The theme reaches the app through localStorage (see
// e2e/screenshots/fixtures.ts, which reads it back off the project name);
// `colorScheme` here is the OS-level preference underneath it, set to match so
// the two never disagree and native controls are painted the same way.
//
// `locale`, `timezoneId` and `reducedMotion` are pinned per project rather than
// globally: identical date and number formatting is something a capture needs
// and the behavioral specs have never depended on, so they keep running under
// whatever the machine has.
const screenshotProjects = Object.entries(SCREENSHOT_VIEWPORTS).flatMap(
  ([viewportName, viewportUse]) =>
    SCREENSHOT_THEMES.map((theme) => ({
      // Prefixed, unlike otari-ai's bare `<viewport>-<theme>`, because this
      // config also holds the behavioral projects and a bare `mobile-dark`
      // would not say which suite it belongs to.
      name: `screenshots-${viewportName}-${theme}`,
      testDir: "./e2e/screenshots",
      // The seed, and deliberately not the parity project on top of it. What a
      // page renders depends on the rows in the database, so some dependency is
      // required, but each parity flow creates and removes what it acts on, so
      // the state they leave behind is the seed's. Depending on them instead
      // buys nothing and costs everything: a single flaky behavioral test takes
      // all 108 captures down with it, which is exactly what happened before
      // this was narrowed. otari-ai needs no dependency at all, because its
      // screenshots run against mocked endpoints.
      dependencies: ["seed"],
      // A capture reads pages and writes nothing, so unlike the behavioral
      // projects (which share one gateway database and must not re-run) a
      // single retry is safe, and absorbs CI contention.
      retries: process.env.CI ? 1 : 0,
      use: {
        ...viewportUse,
        colorScheme: theme,
        locale: "en-US",
        timezoneId: "UTC",
        reducedMotion: "reduce" as const,
      },
    })),
)

export default defineConfig({
  testDir: "./e2e",
  // otari-ai's layout: snapshots sit beside the spec that takes them, scoped
  // per project. `{platform}` is deliberately absent, so there is one canonical
  // set rather than one per developer OS; CI captures it on Linux, and a macOS
  // run comparing against it reports font-rendering diffs that mean nothing.
  // See .github/skills/frontend-standards/testing.md before regenerating.
  snapshotPathTemplate:
    "{testDir}/{testFilePath}-snapshots/{arg}-{projectName}{ext}",
  // Comparison budgets, values and reasoning from otari-ai. Playwright takes
  // `Math.min` of `maxDiffPixels` and `width * height * maxDiffPixelRatio`, so
  // the flat cap governs any capture whose ratio budget is looser than it,
  // which is every tall one: a ratio alone scales with page height, so a long
  // page earns a proportionally huge allowance and can absorb a whole changed
  // paragraph without going red. `threshold` is the per-pixel color distance
  // that counts as a difference at all, set high enough that antialiasing does
  // not register as change.
  expect: {
    toHaveScreenshot: {
      maxDiffPixels: 2_000,
      maxDiffPixelRatio: 0.002,
      threshold: 0.12,
      animations: "disabled",
      caret: "hide",
    },
  },
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
  // A fourth, `hybrid`, stands apart from all three: it has a gateway of its own
  // (see `webServer` below), so it shares neither the database nor the order.
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
      // The screenshot suite is excluded because it has its own six projects
      // below; without this it would also run here, once, unthemed.
      // The hybrid spec is excluded because it belongs to a different
      // deployment: collected here it would run against the standalone gateway,
      // where every one of its assertions is false.
      testIgnore: [
        /dashboard\.spec\.ts/,
        /screenshots\//,
        /parity\.hybrid\.spec\.ts/,
      ],
      dependencies: ["seed"],
      use: { ...devices["Desktop Chrome"] },
    },
    {
      // The same dashboard bundle, served by a gateway attached to a control
      // plane rather than owning its own data. What a hybrid deployment may
      // show an operator is decided by the server, so asserting it needs a
      // second deployment and not a second page: hence its own gateway, its own
      // base URL, and no dependency on the three projects above (it has no
      // database to seed and shares none of their state).
      name: "hybrid",
      testMatch: /parity\.hybrid\.spec\.ts/,
      // Read-only, unlike the behavioral projects that mutate one shared
      // database, so a retry is safe here for the same reason it is on the
      // screenshot projects, and absorbs CI contention.
      retries: process.env.CI ? 1 : 0,
      use: { ...devices["Desktop Chrome"], baseURL: HYBRID_BASE_URL },
    },
    ...screenshotProjects,
  ],
  // Two gateways, both booted before any project runs. The screenshot projects
  // use the hybrid one only for their hybrid landing registry; it opens no database
  // and costs a process, which is cheaper than making its lifetime conditional on
  // which projects were selected.
  webServer: [
    {
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
    {
      command: "bash e2e/serve-hybrid.sh",
      url: `${HYBRID_BASE_URL}/health`,
      // Same opt-in as above, for consistency rather than for the reset: this
      // gateway holds no state to leave dirty, but a stray process on :8010 in a
      // mode of its own would be a confusing thing to run against.
      reuseExistingServer: !!process.env.PLAYWRIGHT_REUSE_SERVER,
      timeout: 120_000,
      stdout: "pipe",
      stderr: "pipe",
    },
  ],
})
