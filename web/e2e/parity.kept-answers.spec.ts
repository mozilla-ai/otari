import { expect, test } from "@playwright/test"

import { login } from "./helpers"

// This build's request policy keeps no answer across loads; only a build that
// replaces it may opt in (`RequestPolicy.keptAnswers`). So a signed-in load and
// a reload here must leave nothing behind and wait for every answer, exactly as
// before the switch existed, and the marks the boot timing reads say so.
test("the dashboard Otari serves keeps no answer across loads", async ({
  page,
}) => {
  await login(page)
  await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible()

  await page.reload()
  await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible()
  await page.waitForLoadState("networkidle")

  const { keys, marks } = await page.evaluate(() => ({
    keys: Object.keys(window.localStorage).sort(),
    marks: performance
      .getEntriesByType("mark")
      .map((entry) => entry.name)
      .filter((name) => /^(boot:kept-|boot:remount|ctx:)/.test(name))
      .sort(),
  }))
  expect(keys).not.toContain("otari.dashboard.bootstrap")
  expect(keys).not.toContain("otari.dashboard.organizationContext")
  expect(marks).toEqual([
    "boot:kept-bootstrap:off",
    "ctx:current",
    "ctx:kept:off",
  ])
})
