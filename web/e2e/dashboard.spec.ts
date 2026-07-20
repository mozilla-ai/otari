import { expect, type Page, test } from "@playwright/test";

// Matches web/e2e/otari.yml. The login step needs a known key.
const MASTER_KEY = "e2e-master-key";

async function login(page: Page): Promise<void> {
  await page.goto("/");
  await page.locator('input[type="password"]').fill(MASTER_KEY);
  await page.locator('input[type="password"]').press("Enter");
  // The sidebar appears once authenticated, regardless of which page the index
  // redirect lands on.
  await expect(page.getByRole("link", { name: "Providers" })).toBeVisible();
}

// One shared gateway + DB, so the flows build on each other and must run in order.
test.describe.configure({ mode: "serial" });

test.describe("dashboard core flows", () => {
  test("first-run onboarding is shown before any provider exists", async ({ page }) => {
    await login(page);
    await expect(page.getByText("Welcome to Otari")).toBeVisible();
  });

  test("add a provider from onboarding, and it appears in the table", async ({ page }) => {
    await login(page);
    await expect(page.getByText("Welcome to Otari")).toBeVisible();

    await page.getByRole("button", { name: "Add your first provider" }).click();
    await page.getByRole("button", { name: "Custom endpoint" }).click();
    await page.getByLabel("Name").fill("e2e-llm");
    await page.getByLabel("API base").fill("http://e2e-box:8000/v1");
    await page.getByRole("button", { name: "Add provider" }).click();

    await expect(page.getByText("e2e-llm")).toBeVisible();
    // Onboarding clears once a provider exists.
    await expect(page.getByText("Welcome to Otari")).toBeHidden();
  });

  test("navigate the management pages", async ({ page }) => {
    await login(page);
    for (const name of ["Models", "Aliases", "Settings", "Providers"]) {
      await page.getByRole("link", { name }).click();
      await expect(page.getByRole("heading", { name })).toBeVisible();
    }
  });

  test("create an alias", async ({ page }) => {
    await login(page);
    await page.getByRole("link", { name: "Aliases" }).click();
    await page.getByRole("button", { name: "New alias" }).click();
    await page.getByLabel("Alias name").fill("fast");
    // Target is a model combobox (allows custom values); type the selector, then
    // close the popover so it does not aria-hide the submit button.
    await page.getByRole("combobox", { name: /Target/ }).fill("openai:gpt-4o");
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Create alias" }).click();

    await expect(page.getByRole("cell", { name: "fast" })).toBeVisible();
  });
});
