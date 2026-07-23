// Drives the Otari dashboard through a full sweep of its pages and records a
// video (webm) plus a screenshot at each stop. Run by scripts/demo_gif/record.sh
// against a gateway already serving the seeded bundle on :8000.
//
// Playwright lives in web/node_modules, so resolve it from there regardless of
// where this file sits.
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdirSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));
const webRequire = createRequire(resolve(here, "../../web/package.json"));
const { chromium } = webRequire("@playwright/test");

const BASE = process.env.BASE_URL || "http://127.0.0.1:8000";
const MASTER_KEY = process.env.OTARI_MASTER_KEY || "otari-demo-key";
const OUT = process.env.OUT_DIR || resolve(here, "artifacts");
const VIDEO_DIR = resolve(OUT, "video");
const SHOT_DIR = resolve(OUT, "shots");
mkdirSync(VIDEO_DIR, { recursive: true });
mkdirSync(SHOT_DIR, { recursive: true });

// Record a little larger than the final GIF (720w) so the downscale antialiases.
const SIZE = { width: 1440, height: 900 };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// The full sweep, in sidebar display order. `settle` gives slower pages extra
// time to finish fetching before we pause on them.
const STOPS = [
  { label: "Overview", heading: "Overview", dwell: 1900, scroll: true },
  { label: "Activity", heading: "Activity", dwell: 1500 },
  { label: "Usage", heading: "Usage", dwell: 2000, scroll: true, settle: 700 },
  // Give the provider health monitor a moment to resolve so the video captures
  // the "5 of 5 reachable" state, not the initial "checking" transient.
  { label: "Providers", heading: "Providers", dwell: 1900, settle: 1400 },
  { label: "Models", heading: "Models", dwell: 1600, settle: 800 },
  { label: "Aliases", heading: "Aliases", dwell: 1400 },
  { label: "Users", heading: "Users", dwell: 1800 },
  { label: "API keys", heading: "API keys", dwell: 1800 },
  { label: "Budgets", heading: "Budgets", dwell: 1800 },
];

async function gentleScroll(page) {
  // Slowly reveal below-the-fold content, then return to top.
  const steps = 4;
  for (let i = 1; i <= steps; i++) {
    await page.mouse.wheel(0, 300);
    await sleep(260);
  }
  await sleep(350);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  await sleep(350);
}

const browser = await chromium.launch();

// Sign in in a throwaway (unrecorded) context so the recording can start already
// on the dashboard. The session is an HttpOnly cookie plus a localStorage marker
// (otari.dashboard.hasSession) that makes the SPA render signed-in on load;
// storageState() carries both into the recorded context.
const authContext = await browser.newContext({ viewport: SIZE });
const authPage = await authContext.newPage();
await authPage.goto(BASE + "/", { waitUntil: "networkidle" });
await authPage.locator('input[type="password"]').fill(MASTER_KEY);
await authPage.locator('input[type="password"]').press("Enter");
await authPage.getByRole("navigation").getByRole("link", { name: "Providers" }).waitFor();
const storageState = await authContext.storageState();
await authContext.close();

const context = await browser.newContext({
  viewport: SIZE,
  recordVideo: { dir: VIDEO_DIR, size: SIZE },
  reducedMotion: "no-preference",
  storageState,
});
const page = await context.newPage();

try {
  // Start already signed in, on the Overview. Wait for the data to populate (a
  // "… ago" timestamp in Recent activity) so the sweep opens on live content;
  // the brief initial loading skeleton is trimmed from the final encode
  // (START_TRIM in record.sh).
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  // Require the seeded data to load; if it never does, fail rather than record
  // an empty dashboard (record.sh runs under `set -e`, so this aborts the run
  // before the committed GIF is overwritten).
  await page.getByText(/ago/).first().waitFor({ timeout: 8000 });
  await sleep(700);
  await page.screenshot({ path: resolve(SHOT_DIR, "00-overview.png") });

  // --- Full sweep ------------------------------------------------------------
  let i = 0;
  for (const stop of STOPS) {
    i += 1;
    const link = page.getByRole("navigation").getByRole("link", { name: stop.label, exact: true });
    await link.click();
    // Require the page heading so a broken navigation aborts the run instead of
    // silently recording an incomplete tour. If a page title changes, update the
    // matching `heading` in STOPS.
    await page.getByRole("heading", { name: stop.heading }).first().waitFor({ timeout: 6000 });
    if (stop.settle) await sleep(stop.settle);
    await sleep(stop.dwell);
    const n = String(i).padStart(2, "0");
    const slug = stop.label.toLowerCase().replace(/[^a-z0-9]+/g, "-");
    await page.screenshot({ path: resolve(SHOT_DIR, `${n}-${slug}.png`) });
    if (stop.scroll) await gentleScroll(page);
  }

  // Land back on Overview to close the loop.
  await page.getByRole("navigation").getByRole("link", { name: "Overview", exact: true }).click();
  await sleep(900);
} finally {
  // Close the page/context to flush the video to disk.
  await page.close();
  await context.close();
  await browser.close();
}

console.log("Tour complete. Video in", VIDEO_DIR, "screenshots in", SHOT_DIR);
