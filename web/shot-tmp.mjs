import { chromium } from "playwright";
const OUT = "/tmp/claude-0/-workspace-otari-worktrees-tools/0aec66e8-14ba-499e-84ee-f2750482a23b/scratchpad/shots";
const BASE = "http://100.80.227.1:8000";
const browser = await chromium.launch();
const page = await (await browser.newContext({ viewport: { width: 1600, height: 1000 } })).newPage();
page.on("pageerror", (e) => console.log("PAGE ERROR:", e.message));
page.on("console", (m) => { if (m.type() === "error") console.log("CONSOLE ERROR:", m.text().slice(0, 200)); });
await page.goto(BASE + "/", { waitUntil: "domcontentloaded" });
await page.evaluate(async () => {
  await fetch("/v1/auth/session", { method: "POST", headers: { "Content-Type": "application/json" },
    credentials: "include", body: JSON.stringify({ master_key: "smoke-master-key" }) });
  localStorage.setItem("otari.dashboard.hasSession", "1");
});
await page.goto(BASE + "/#/activity?tool=web_search", { waitUntil: "domcontentloaded" });
await page.reload({ waitUntil: "domcontentloaded" });
await page.waitForTimeout(4000);
console.log("headers:", (await page.locator("th").allInnerTexts()).join(" | "));
console.log("rows:", await page.locator("tbody tr").count());
console.log("tool pills:", await page.locator('[aria-label^="Gateway tools"]').count());
const firstRow = page.locator("tbody tr").first();
console.log("first row text:", (await firstRow.innerText()).replace(/\n/g, " | "));
await page.screenshot({ path: `${OUT}/their-activity.png` });
await browser.close();
