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
    await expect(button).toHaveText("Copied")
    await expect(page.getByRole("status")).toHaveText(`Copied ${name}.`)
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
    await page.getByRole("button", { name: "Copy key export command" }).click()
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
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    390,
  )
  const snippet = page.getByLabel("Chat completion example", { exact: true })
  await snippet.focus()
  await page.keyboard.press("End")
  expect(
    await snippet.evaluate((node) => node.scrollWidth > node.clientWidth),
  ).toBe(true)
  for (const button of await page
    .getByRole("button", { name: /^Copy / })
    .all()) {
    const box = await button.boundingBox()
    expect(box?.width).toBeGreaterThanOrEqual(44)
    expect(box?.height).toBeGreaterThanOrEqual(44)
  }
})
