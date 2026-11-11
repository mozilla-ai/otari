import path from "node:path"

import type { Locator, Page } from "@playwright/test"

import { API_ROOT } from "@/shared/api/client"

import { authHeaders, expectOk, nav } from "../helpers"
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

/** A management API collection, read as the master key. */
export async function listed<T>(page: Page, path: string): Promise<T[]> {
  const response = await page.request.get(`${API_ROOT}${path}`, {
    headers: authHeaders,
  })
  await expectOk(response, `list ${path}`)
  const body = (await response.json()) as T[] | { data: T[] }
  return Array.isArray(body) ? body : body.data
}

/** Send one management API call as the master key, and fail on an error. */
export async function call(
  page: Page,
  method: "POST" | "PATCH" | "DELETE",
  path: string,
  data?: unknown,
): Promise<void> {
  const response = await page.request.fetch(`${API_ROOT}${path}`, {
    method,
    headers: authHeaders,
    data,
  })
  await expectOk(response, `${method} ${path}`)
}

/**
 * Open a page nested under a sidebar group.
 *
 * `openNested` in ../helpers matches the child exactly, which misses a row the
 * rail badges ("Web search" reads "Web search Beta"), so this one matches the
 * start of the name.
 */
export async function openNestedPage(
  page: Page,
  group: string,
  child: string,
): Promise<void> {
  const toggle = nav(page).getByRole("button", { name: group, exact: true })
  if ((await toggle.getAttribute("aria-expanded")) !== "true") {
    await toggle.click()
  }
  await nav(page)
    .getByRole("link", { name: new RegExp(`^${child}`) })
    .click()
}
