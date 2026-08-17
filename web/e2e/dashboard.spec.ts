import { expect, test } from "@playwright/test"

import { login, MASTER_KEY, nav } from "./helpers"

// One shared gateway + DB, so the flows build on each other and must run in order.
test.describe.configure({ mode: "serial" })

test.describe("dashboard core flows", () => {
  test("first-run overview guides the operator to provider setup", async ({
    page,
  }) => {
    await login(page)
    await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible()
    await expect(page.getByText("Get started with Otari")).toBeVisible()
  })

  test("add a provider from onboarding, and it appears in the table", async ({
    page,
  }) => {
    await login(page)
    await page.getByRole("button", { name: "Add your first provider" }).click()
    await expect(page.getByText("Welcome to Otari")).toBeVisible()

    await page.getByRole("button", { name: "Add your first provider" }).click()
    await page.getByRole("button", { name: "Custom endpoint" }).click()
    await page.getByLabel("Name").fill("e2e-llm")
    await page.getByLabel("API base").fill("http://e2e-box:8000/v1")
    await page.getByRole("button", { name: "Add provider" }).click()

    await expect(page.getByText("e2e-llm")).toBeVisible()
    // Onboarding clears once a provider exists.
    await expect(page.getByText("Welcome to Otari")).toBeHidden()
  })

  test("navigate the management pages", async ({ page }) => {
    await login(page)
    for (const name of [
      "Models",
      "Routing",
      "Users",
      "Budgets",
      "Settings",
      "Providers",
    ]) {
      await nav(page).getByRole("link", { name }).click()
      // Exact match: the Budgets onboarding heading ("No budgets yet") would
      // otherwise also substring-match the page title.
      await expect(
        page.getByRole("heading", { name, exact: true }),
      ).toBeVisible()
    }
  })

  test("create a budget", async ({ page }) => {
    await login(page)
    await nav(page).getByRole("link", { name: "Budgets" }).click()
    await page.getByRole("button", { name: "Create your first budget" }).click()
    await page.getByLabel("Name (optional)").fill("e2e-budget")
    await page.getByLabel("Spending limit (USD)").fill("100")
    await page.getByRole("button", { name: "Create budget" }).click()

    // The shared table renders on react-aria, so non-row-header cells are gridcells.
    await expect(page.getByRole("gridcell", { name: "$100.00" })).toBeVisible()
    await expect(page.getByText("e2e-budget")).toBeVisible()
    await expect(page.getByText("No budgets yet")).toBeHidden()
  })

  test("create a user and assign the budget", async ({ page }) => {
    await login(page)
    await nav(page).getByRole("link", { name: "Users" }).click()
    // A bootstrap virtual user already exists (from the first-run key), so use the
    // header action, not the empty-state button. It is removed when the form opens,
    // leaving the form's own "Create user" as the only match.
    await page.getByRole("button", { name: "Create user" }).click()
    // Role-scoped: the rows behind this form carry "Copy user id" controls, whose
    // accessible names also contain "user id".
    await page
      .getByRole("textbox", { name: /User ID/ })
      .fill("alice@example.com")
    // The budget created by the prior test is the only non-default option.
    await page.getByLabel("Budget").selectOption({ index: 1 })
    await page.getByRole("button", { name: "Create user" }).click()

    const row = page.getByRole("row", { name: /alice@example\.com/ })
    await expect(row).toBeVisible()
    // The assigned budget's name renders in the user's Budget cell.
    await expect(row.getByText("e2e-budget")).toBeVisible()
  })

  test("create an API key owned by a chosen user", async ({ page }) => {
    await login(page)
    await nav(page).getByRole("link", { name: "API keys" }).click()
    // A bootstrap key already exists, so use the header action, not onboarding.
    await page.getByRole("button", { name: "Create key" }).click()
    await page.getByLabel("Name").fill("ci-bot")
    // Owner is required (user-first). Reuse the user created earlier; type it and
    // close the combobox popover so it does not aria-hide the submit button.
    await page
      .getByPlaceholder("Pick a user, or type a new id…")
      .fill("alice@example.com")
    await page.keyboard.press("Escape")
    await page.getByRole("button", { name: "Create key" }).click()

    // The one-time reveal appears; acknowledge it.
    await page.getByRole("button", { name: /saved this key/i }).click()

    const row = page.getByRole("row", { name: /ci-bot/ })
    await expect(row).toBeVisible()
    // The key is owned by the named user, not an anonymous virtual one.
    await expect(row.getByText("alice@example.com")).toBeVisible()
  })

  test("create a routing policy", async ({ page }) => {
    await login(page)
    await nav(page).getByRole("link", { name: "Routing" }).click()
    await page.getByRole("button", { name: "New policy" }).click()
    // Role-scoped for the same reason as the user form: policy rows carry a
    // "Copy policy name" control.
    await page.getByRole("textbox", { name: /Policy name/ }).fill("fast")
    // "Serves" is a model combobox (allows custom values); type the selector, then
    // close the popover so it does not aria-hide the submit button.
    await page.getByRole("combobox", { name: /Serves/ }).fill("openai:gpt-4o")
    await page.keyboard.press("Escape")
    await page.getByRole("button", { name: "Create policy" }).click()

    // The policy name is the table's row-header cell (react-aria rowheader).
    await expect(page.getByRole("rowheader", { name: "fast" })).toBeVisible()
  })

  test("grows a policy a fallback chain", async ({ page }) => {
    await login(page)
    await nav(page).getByRole("link", { name: "Routing" }).click()
    await page.getByRole("button", { name: "New policy" }).click()
    await page.getByRole("textbox", { name: /Policy name/ }).fill("chained")
    await page.getByRole("combobox", { name: /Serves/ }).fill("openai:gpt-4o")
    await page.keyboard.press("Escape")

    // The failure chain is summoned, not presented, so naming one model stays a
    // short task.
    await expect(page.getByText("If that fails, try")).toBeHidden()
    await page.getByRole("button", { name: /Add a fallback chain/ }).click()
    await page
      .getByRole("combobox", { name: /Fallback 1/ })
      .fill("anthropic:claude-3-5-haiku-latest")
    await page.keyboard.press("Escape")
    await page.getByRole("button", { name: "Create policy" }).click()

    // Scoped to the row this test created: "+1 on failure" anywhere on the page
    // would also be satisfied by another policy's chain, so a `chained` saved
    // without its fallback would still pass.
    const chained = page
      .getByRole("row")
      .filter({ has: page.getByRole("rowheader", { name: "chained" }) })
    await expect(chained).toBeVisible()
    await expect(chained).toContainText(/\+1 on failure/)
  })

  test("renames a policy in place", async ({ page }) => {
    // The rename target has to be free for the 409 not to fire. serve.sh wipes the
    // database, so it is on a first run; a re-run against a warm one still carries
    // the `renamed` this test left behind, and would fail on its own leftovers.
    const dropped = await page.request.delete("/v1/routing/policies/renamed", {
      headers: { Authorization: `Bearer ${MASTER_KEY}` },
    })
    expect([204, 404]).toContain(dropped.status())

    await login(page)
    await nav(page).getByRole("link", { name: "Routing" }).click()

    const chained = page
      .getByRole("row")
      .filter({ has: page.getByRole("rowheader", { name: "chained" }) })
    await chained.getByRole("button", { name: "Edit" }).click()
    await page.getByRole("textbox", { name: /Policy name/ }).fill("renamed")
    await page.getByRole("button", { name: "Save" }).click()

    // A rename moves the row rather than copying it, so the old name has to be
    // gone: two rows would mean callers could still reach the policy either way.
    const renamed = page
      .getByRole("row")
      .filter({ has: page.getByRole("rowheader", { name: "renamed" }) })
    await expect(renamed).toBeVisible()
    await expect(renamed).toContainText(/\+1 on failure/)
    await expect(page.getByRole("rowheader", { name: "chained" })).toBeHidden()
  })

  // The share card is the one flow whose output cannot be checked in jsdom: it
  // ends in a PNG, and jsdom has no canvas, no toBlob and no object URLs, so the
  // unit tests can only assert the wiring around a mocked rasterizer. Two bugs got
  // through that way, both fatal and both invisible to a green unit suite: drawing
  // an SVG from a blob: URL taints the canvas so toBlob() refuses outright, and a
  // long model list overflowed the fixed card frame so flex-shrink collapsed the
  // title to zero height. This test exists to catch that class of failure.
  test("shares the usage view as a real PNG", async ({ page }) => {
    // This suite starts on an empty database (serve.sh wipes it), and no earlier
    // flow creates usage, so the card would otherwise render its empty state.
    // /v1/usage/external-events writes usage rows with no provider call.
    const auth = {
      Authorization: `Bearer ${MASTER_KEY}`,
      "Content-Type": "application/json",
    }

    // Ingestion rejects usage for a user that does not exist, and this test owns
    // its own rather than depending on an earlier one in the serial order.
    const owner = "share-e2e@example.com"
    const created = await page.request.post("/v1/users", {
      headers: auth,
      data: { user_id: owner },
    })
    // A re-run against a warm DB is fine; only a genuine failure should fail here.
    expect([200, 201, 400, 409]).toContain(created.status())

    const seeded = await page.request.post("/v1/usage/external-events", {
      headers: auth,
      data: {
        source: "e2e-seed",
        user_id: owner,
        events: Array.from({ length: 12 }, (_, i) => ({
          source_event_id: `share-seed-${i}`,
          timestamp: new Date(Date.now() - (i + 1) * 3_600_000).toISOString(),
          provider: i % 2 === 0 ? "openai" : "groq",
          // A fully-qualified selector, so the card's name collapsing is exercised
          // on the shape that motivated it.
          model: i % 2 === 0 ? "gpt-4o" : "fireworks/accounts/llama-3.3-70b",
          input_tokens: 1000 + i * 50,
          output_tokens: 200 + i * 10,
          duration_ms: 400 + i,
        })),
      },
    })
    expect(seeded.ok(), await seeded.text()).toBe(true)

    await login(page)
    await nav(page).getByRole("link", { name: "Usage" }).click()

    // The affordance lives in the chart's own caption row and only exists when the
    // range has data to share.
    const share = page.getByRole("button", { name: "Share usage as an image" })
    await expect(share).toBeVisible()
    await share.click()

    const dialog = page.getByRole("alertdialog")
    await expect(dialog).toBeVisible()

    // The preview is the PNG itself, so asserting it decoded is asserting the
    // rasterizer produced a real image. naturalWidth stays 0 on a failed decode,
    // which is exactly what the tainted-canvas bug produced.
    const preview = dialog.getByAltText(
      "Preview of the usage card that will be shared",
    )
    await expect(preview).toBeVisible({ timeout: 20_000 })
    await expect
      .poll(
        async () => preview.evaluate((el: HTMLImageElement) => el.naturalWidth),
        { timeout: 20_000 },
      )
      .toBeGreaterThan(0)

    // The preview must settle. The rasterize effect once carried two arrays that
    // were rebuilt on every render, so its own setPreview re-armed the debounce
    // and the card re-encoded every 300ms for as long as the dialog stayed open.
    // jsdom cannot show this (rasterize throws there, and React bails on an
    // identical error string, so nothing re-renders); a real browser can.
    const firstSrc = await preview.evaluate((el: HTMLImageElement) => el.src)
    await page.waitForTimeout(2500)
    expect(await preview.evaluate((el: HTMLImageElement) => el.src)).toBe(
      firstSrc,
    )

    // The card node itself, not the dialog: it is rendered off-screen as a sibling
    // of the dialog's own section so it can be rasterized at full size.
    // Located by attribute, not by role: the off-screen copy is aria-hidden on
    // purpose, so it is deliberately absent from the accessibility tree.
    const card = page.locator('[aria-label^="Usage card"]')
    // The seeded selector is `fireworks/accounts/llama-3.3-70b`; the card prints
    // only the final model type.
    await expect(card).toContainText("llama-3.3-70b")
    await expect(card).not.toContainText("fireworks/accounts")
    // Hardcoded, never derived from the gateway's own host.
    await expect(card).toContainText("otari.ai")

    // Every row count must render in both shapes: the frame is fixed, so the rows
    // divide a height budget, and a band collapsing to zero is the regression.
    // Width is checked too, and for the same reason: a wide card once set its hero
    // at the square card's size, which ran the number off both edges of the frame.
    // This seed cannot reproduce that on its own (it prices nothing, so the hero is
    // a two-digit request count), and the unit suite covers the sizing itself; the
    // check is here so any future arrangement that spills sideways is caught in the
    // one place the card is measured after a real layout.
    for (const shape of ["Square", "Wide"]) {
      await dialog.getByRole("button", { name: shape }).click()
      for (const rows of ["1", "9"]) {
        await dialog.getByRole("button", { name: rows, exact: true }).click()
        const bands = await page.evaluate(() => {
          const node = document.querySelector<HTMLElement>(
            '[role="img"][aria-label^="Usage card"]',
          )
          // Explicit, so an unmounted card reports itself rather than throwing a
          // TypeError that points at the evaluate call.
          if (node === null) {
            return { heights: [] as number[], overflows: false, missing: true }
          }
          const box = node.getBoundingClientRect()
          return {
            heights: Array.from(node.children).map((c) =>
              Math.round(c.getBoundingClientRect().height),
            ),
            overflows:
              node.scrollHeight > Math.round(box.height) ||
              node.scrollWidth > Math.round(box.width),
            missing: false,
          }
        })
        expect(bands.missing, `${shape}/${rows} rendered no card`).toBe(false)
        expect(
          bands.heights.length,
          `${shape}/${rows} rendered no bands`,
        ).toBeGreaterThan(0)
        expect(
          bands.heights.filter((h) => h === 0),
          `${shape}/${rows} collapsed a band`,
        ).toEqual([])
        expect(bands.overflows, `${shape}/${rows} overflowed the frame`).toBe(
          false,
        )
      }
    }

    // Download is the only terminal action that can be asserted: Playwright cannot
    // read an image off the clipboard, so "Copy image" is deliberately untested.
    await dialog.getByRole("button", { name: "Square" }).click()
    const download = page.waitForEvent("download")
    await dialog.getByRole("button", { name: "Download PNG" }).click()
    const file = await download
    expect(file.suggestedFilename()).toMatch(
      /^otari-usage-\d{4}-\d{2}-\d{2}.*\.png$/,
    )

    const path = await file.path()
    const { readFileSync } = await import("node:fs")
    const bytes = readFileSync(path)
    // PNG magic number, then the IHDR width/height, which prove the card was
    // rasterized at its declared size rather than as an empty or clipped canvas.
    expect(bytes.subarray(0, 8).toString("hex")).toBe("89504e470d0a1a0a")
    expect(bytes.readUInt32BE(16)).toBe(2160)
    expect(bytes.readUInt32BE(20)).toBe(2160)
  })
})
