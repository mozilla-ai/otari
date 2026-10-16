// Drives the Otari dashboard through its most useful pages and records a video
// (webm) plus a screenshot at each stop. Run by scripts/demo_gif/record.sh
// against a gateway already serving the seeded bundle on :8000, with
// mock_provider.py answering the Playground on :8099.
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

// Record a little larger than the final GIF so the downscale antialiases.
const SIZE = { width: 1440, height: 900 };
const PROMPT = "What's the best way to retry failed LLM calls?";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Headless video has no pointer, so draw one: an arrow that follows the mouse
// and a ring that pulses on press, which is what makes each click legible.
const CURSOR_SCRIPT = `
window.addEventListener("DOMContentLoaded", () => {
  const cursor = document.createElement("div");
  cursor.innerHTML = '<svg width="22" height="22" viewBox="0 0 24 24"><path d="M4 2l16 9.5-7.2 1.6L9.6 20z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  Object.assign(cursor.style, { position: "fixed", left: "0", top: "0", zIndex: 2147483647, pointerEvents: "none", transform: "translate(-200px,-200px)" });
  const ring = document.createElement("div");
  Object.assign(ring.style, { position: "fixed", left: "-14px", top: "-14px", width: "28px", height: "28px", borderRadius: "50%", border: "2px solid #0098A4", zIndex: 2147483646, pointerEvents: "none", opacity: "0" });
  document.documentElement.append(ring, cursor);
  let x = -200, y = -200;
  addEventListener("mousemove", (e) => { x = e.clientX; y = e.clientY; cursor.style.transform = "translate(" + (x - 3) + "px," + (y - 2) + "px)"; }, true);
  addEventListener("mousedown", () => {
    ring.animate(
      [{ transform: "translate(" + x + "px," + y + "px) scale(0.4)", opacity: 0.9 }, { transform: "translate(" + x + "px," + y + "px) scale(1.4)", opacity: 0 }],
      { duration: 420, easing: "ease-out" },
    );
  }, true);
});
`;

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
await authPage.getByRole("navigation", { name: "Sidebar" }).getByRole("link", { name: "Usage" }).waitFor();
const storageState = await authContext.storageState();
await authContext.close();

const context = await browser.newContext({
  viewport: SIZE,
  recordVideo: { dir: VIDEO_DIR, size: SIZE },
  reducedMotion: "no-preference",
  // SQLite hands some timestamps back without an offset; pinning the browser to
  // UTC keeps "last used" reading as the past rather than hours in the future.
  timezoneId: "UTC",
  storageState,
});
await context.addInitScript(CURSOR_SCRIPT);
const page = await context.newPage();

async function glide(x, y, steps = 18) {
  await page.mouse.move(x, y, { steps });
}
async function click(locator) {
  await locator.scrollIntoViewIfNeeded();
  const box = await locator.boundingBox();
  if (!box) throw new Error(`no box for ${locator}`);
  await glide(box.x + Math.min(box.width / 2, 60), box.y + box.height / 2);
  await sleep(110);
  await page.mouse.down();
  await page.mouse.up();
}
const sidebar = () => page.getByRole("navigation", { name: "Sidebar" });
const nav = (name) => sidebar().getByRole("link", { name, exact: true });

let shot = 0;
async function snap(slug) {
  shot += 1;
  await page.screenshot({ path: resolve(SHOT_DIR, `${String(shot).padStart(2, "0")}-${slug}.png`) });
}
// Require the page heading so a broken navigation aborts the run instead of
// silently recording an incomplete tour. If a page title changes, update it here.
async function arrive(heading) {
  await page.getByRole("heading", { name: heading, level: 1 }).waitFor({ timeout: 8000 });
}
async function scrollBy(dy, steps = 5) {
  for (let i = 0; i < steps; i++) {
    await page.mouse.wheel(0, dy / steps);
    await sleep(70);
  }
}

try {
  // Start already signed in, on the Overview. Require the seeded data to load (a
  // "… ago" timestamp in Recent activity); if it never does, fail rather than
  // record an empty dashboard (record.sh runs under `set -e`, so this aborts the
  // run before the committed GIF is overwritten). The brief initial loading
  // skeleton is trimmed from the final encode (START_TRIM in record.sh).
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  await page.getByText(/ago/).first().waitFor({ timeout: 8000 });
  await sleep(1000);
  await snap("overview");

  // --- Overview: sweep the spend chart so its tooltip tracks the bars --------
  // The bars sit between the section title and its caption.
  const title = await page.getByText("Spend, last 30 days", { exact: true }).last().boundingBox();
  const caption = await page.getByText(/^Daily totals for the selected workspace/).boundingBox();
  const barY = title.y + (caption.y - title.y) * 0.72;
  await glide(caption.x + caption.width * 0.6, barY, 26);
  await glide(SIZE.width - 60, barY, 44);
  await sleep(700);

  // --- Usage: group the chart by model ----------------------------------------
  await click(nav("Usage"));
  await arrive("Usage & analytics");
  await sleep(900);
  await click(page.getByRole("button", { name: /No grouping/ }));
  await sleep(350);
  await click(page.getByRole("option", { name: "By model" }));
  await sleep(2000);
  await snap("usage");
  await scrollBy(520, 6);
  await sleep(1100);

  // --- Activity: open the request a fallback rescued --------------------------
  await click(nav("Activity"));
  await arrive("Activity");
  await sleep(1000);
  const rescued = page.getByRole("row").filter({ hasText: "gemini-3.8-flash" }).first();
  await click(rescued);
  await sleep(2200);
  await snap("activity");

  // --- Playground: the same prompt, two models, side by side ------------------
  await click(page.getByRole("link", { name: "Playground", exact: true }));
  await arrive("Playground");
  await sleep(600);
  await click(page.getByText("Compare", { exact: true }));
  await sleep(500);
  await click(page.getByRole("button", { name: "Model A" }));
  await sleep(300);
  await click(page.getByRole("dialog", { name: "Model A" }).getByRole("button", { name: "openai/gpt-6.1-sol", exact: true }));
  await sleep(300);
  await click(page.getByRole("button", { name: "Model B" }));
  await sleep(300);
  await click(page.getByRole("dialog", { name: "Model B" }).getByRole("button", { name: "deepseek/deepseek-v4-pro", exact: true }));
  await sleep(300);
  const composer = page.getByRole("textbox", { name: "Message" });
  await click(composer);
  await composer.pressSequentially(PROMPT, { delay: 22 });
  await sleep(300);
  await composer.press("Enter");
  // Both replies stream from the mock; wait for the slower one to finish and
  // its turn readout (tokens, cost) to land.
  await page.getByText("Give up after ~5 attempts").waitFor({ timeout: 20000 });
  await sleep(1800);
  await snap("playground");

  // --- Models catalog ---------------------------------------------------------
  await click(nav("Models"));
  await arrive("Models");
  await sleep(1900);
  await snap("models");

  // --- Routing policies -------------------------------------------------------
  await click(sidebar().getByRole("button", { name: "Routing" }));
  await sleep(250);
  await click(nav("Policies"));
  await arrive("Routing");
  await sleep(1900);
  await snap("routing");

  // --- API keys ---------------------------------------------------------------
  await click(nav("API keys"));
  await arrive("API keys");
  await sleep(1500);
  await snap("keys");

  // --- Organization: members, budgets, providers ------------------------------
  await click(page.getByRole("complementary").getByRole("link", { name: "Organization" }));
  await arrive("Members");
  await sleep(1600);
  await snap("members");
  await click(page.getByRole("link", { name: "Spend & budgets" }));
  await arrive("Budgets");
  await sleep(1600);
  await snap("budgets");
  await click(page.getByRole("link", { name: "Deployment providers" }));
  await arrive("Deployment providers");
  // Let the provider health monitor resolve so the video captures "7 of 7
  // reachable", not the initial "checking" transient.
  await page.getByText(/7 of 7 providers reachable/).waitFor({ timeout: 10000 });
  await sleep(1500);
  await snap("providers");

  // Land back on Overview to close the loop.
  await click(page.getByRole("link", { name: /Back to/ }));
  await sidebar().waitFor();
  await sleep(400);
  await click(nav("Overview"));
  await arrive("Overview");
  await sleep(1200);
} finally {
  // Close the page/context to flush the video to disk.
  await page.close();
  await context.close();
  await browser.close();
}

console.log("Tour complete. Video in", VIDEO_DIR, "screenshots in", SHOT_DIR);
