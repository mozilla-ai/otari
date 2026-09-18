import { expect, test } from "@playwright/test"

test("welcome copies each complete snippet and resets its feedback", async ({
  page,
  context,
}) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"])
  await page.goto("/welcome")
  await expect(
    page.getByRole("heading", { name: "Your gateway is running." }),
  ).toBeVisible()

  for (const name of [
    "key export command",
    "client install command",
    "chat completion example",
  ]) {
    const button = page.getByRole("button", { name: `Copy ${name}` })
    const snippet = await page
      .getByLabel(name, { exact: false })
      .filter({
        has: page.locator("code"),
      })
      .textContent()
    await button.focus()
    await page.keyboard.press("Enter")
    await expect(page.getByRole("status")).toHaveText(`Copied ${name}.`)
    await expect(button).toHaveText("Copied")
    expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(
      snippet,
    )
    await expect(button).toHaveText("")
  }
})

for (const unavailable of [false, true]) {
  test(`welcome explains ${unavailable ? "unavailable" : "denied"} clipboard access`, async ({
    page,
  }) => {
    await page.addInitScript((missing) => {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: missing
          ? undefined
          : { writeText: () => Promise.reject(new Error("Permission denied")) },
      })
    }, unavailable)
    await page.goto("/welcome")
    await page.clock.install()
    await page.clock.pauseAt(new Date())
    const copy = page.getByRole("button", { name: "Copy key export command" })
    await copy.click()
    await page.clock.runFor(1)
    await expect(page.getByRole("status")).toHaveText(
      "Could not copy. Select the code and copy it manually.",
    )
    await copy.click()
    await expect(page.getByRole("status")).toHaveText("")
    await page.clock.runFor(1)
    await expect(page.getByRole("status")).toHaveText(
      "Could not copy. Select the code and copy it manually.",
    )
    await expect(
      page.getByText("Could not copy.", { exact: false }).first(),
    ).toBeVisible()
    await expect(
      page.getByRole("button", { name: "Copy key export command" }),
    ).not.toHaveText("Copied")
  })
}

test("welcome fits a phone while keeping long code scrollable", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto("/welcome")
  const { scrollWidth, clientWidth } = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }))
  expect(scrollWidth).toBeLessThanOrEqual(clientWidth)
  const snippet = page.getByLabel("Chat completion example", { exact: true })
  await snippet.focus()
  const overflow = await snippet.evaluate((node) => ({
    scrollWidth: node.scrollWidth,
    clientWidth: node.clientWidth,
    overflowX: getComputedStyle(node).overflowX,
  }))
  expect(overflow.scrollWidth).toBeGreaterThan(overflow.clientWidth)
  expect(["auto", "scroll"]).toContain(overflow.overflowX)
  for (const button of await page
    .getByRole("button", { name: /^Copy / })
    .all()) {
    const box = await button.boundingBox()
    expect(box?.width).toBeGreaterThanOrEqual(44)
    expect(box?.height).toBeGreaterThanOrEqual(44)
  }
})
