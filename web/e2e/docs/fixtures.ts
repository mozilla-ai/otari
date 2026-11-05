import path from "node:path"

import type { Locator, Page } from "@playwright/test"

import { waitForStable } from "../screenshots/fixtures"

// The docs' images, written in place rather than compared against a baseline:
// what fails a run is a flow the doc describes no longer working, and the
// images themselves are whatever the dashboard renders today. The release
// publish regenerates them (otari-docs.yml), so the site never shows an old one.
const SCREENSHOTS_DIR = path.resolve(
  import.meta.dirname,
  "../../../docs/assets/screenshots",
)

/**
 * Capture a page or one element of it as `docs/assets/screenshots/<name>.png`.
 *
 * The name is what a docs page links to, and
 * tests/unit/test_docs_screenshots.py reads it off this call, so pass it as a
 * string literal.
 */
export async function docScreenshot(
  target: Page | Locator,
  name: string,
): Promise<void> {
  const page = "page" in target ? target.page() : target
  await waitForStable(page)
  await target.screenshot({
    path: path.join(SCREENSHOTS_DIR, `${name}.png`),
    animations: "disabled",
    caret: "hide",
  })
}
