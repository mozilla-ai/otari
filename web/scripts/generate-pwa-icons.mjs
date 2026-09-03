#!/usr/bin/env node
// Regenerates web/public/pwa/*.png (the PWA install icons) from the Otari
// mark in web/public/favicon.svg: white mark on the brand teal, centered by
// its own aspect ratio (273x250, not square) so it is never stretched.
// Uses Playwright's already-installed Chromium to rasterize, rather than
// adding an image-processing dependency.
import { chromium } from "@playwright/test";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const webRoot = path.resolve(__dirname, "..");
const publicDir = path.join(webRoot, "public");
const pwaDir = path.join(publicDir, "pwa");

const BRAND = "#0098A4";
const MARK_WIDTH = 273;
const MARK_HEIGHT = 250;

// height of the mark as a fraction of the icon's own height; "any" purpose
// icons can use the full canvas, maskable icons need a safe zone since the
// OS may crop up to ~20% off each edge with a circle/squircle mask.
const FILL_RATIO_ANY = 0.6;
const FILL_RATIO_MASKABLE = 0.4;

const TARGETS = [
  { file: "apple-touch-icon.png", size: 180, purpose: "any" },
  { file: "icon-192.png", size: 192, purpose: "any" },
  { file: "icon-512.png", size: 512, purpose: "any" },
  { file: "icon-maskable-512.png", size: 512, purpose: "maskable" },
];

async function main() {
  const faviconSvg = await readFile(path.join(publicDir, "favicon.svg"), "utf8");
  const whiteMarkSvg = faviconSvg.replace(/fill="#0098A4"/i, 'fill="#ffffff"');
  if (!whiteMarkSvg.includes('fill="#ffffff"')) {
    throw new Error("favicon.svg fill did not match #0098A4; update the recolor regex");
  }

  const browser = await chromium.launch();
  const page = await browser.newPage();

  for (const target of TARGETS) {
    const fillRatio = target.purpose === "maskable" ? FILL_RATIO_MASKABLE : FILL_RATIO_ANY;
    const markHeight = target.size * fillRatio;
    const markWidth = (markHeight * MARK_WIDTH) / MARK_HEIGHT;

    const html = `<!doctype html><html><head><style>
      *{margin:0;padding:0;box-sizing:border-box}
      html,body{width:${target.size}px;height:${target.size}px;background:${BRAND};overflow:hidden}
      .mark{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:${markWidth}px;height:${markHeight}px}
      .mark svg{width:100%;height:100%;display:block}
    </style></head><body><div class="mark">${whiteMarkSvg}</div></body></html>`;

    await page.setViewportSize({ width: target.size, height: target.size });
    await page.setContent(html);
    await page.waitForTimeout(30); // let the inline SVG paint before capture
    const outPath = path.join(pwaDir, target.file);
    await page.screenshot({ path: outPath });
    console.log(`wrote ${path.relative(webRoot, outPath)} (${target.size}x${target.size}, ${target.purpose})`);
  }

  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
