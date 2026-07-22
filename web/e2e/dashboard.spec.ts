import { expect, type Page, test } from "@playwright/test";

// Matches web/e2e/otari.yml. The login step needs a known key.
const MASTER_KEY = "e2e-master-key";

// Scope link lookups to the sidebar navigation landmark. The Overview landing
// page has tile-links whose names substring-collide with sidebar items (e.g.
// "Providers healthy", "Active users", "No budgets configured"), so an unscoped
// getByRole("link", { name }) is ambiguous there.
const nav = (page: Page) => page.getByRole("navigation");

async function login(page: Page): Promise<void> {
  await page.goto("/");
  await page.locator('input[type="password"]').fill(MASTER_KEY);
  await page.locator('input[type="password"]').press("Enter");
  // The sidebar appears once authenticated, regardless of the index landing
  // page.
  await expect(nav(page).getByRole("link", { name: "Providers" })).toBeVisible();
}

// One shared gateway + DB, so the flows build on each other and must run in order.
test.describe.configure({ mode: "serial" });

test.describe("dashboard core flows", () => {
  test("first-run overview guides the operator to provider setup", async ({ page }) => {
    await login(page);
    await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible();
    await expect(page.getByText("Get started with Otari")).toBeVisible();
  });

  test("add a provider from onboarding, and it appears in the table", async ({ page }) => {
    await login(page);
    await page.getByRole("button", { name: "Add your first provider" }).click();
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
    for (const name of ["Models", "Aliases", "Users", "Budgets", "Settings", "Providers"]) {
      await nav(page).getByRole("link", { name }).click();
      // Exact match: the Budgets onboarding heading ("No budgets yet") would
      // otherwise also substring-match the page title.
      await expect(page.getByRole("heading", { name, exact: true })).toBeVisible();
    }
  });

  test("create a budget", async ({ page }) => {
    await login(page);
    await nav(page).getByRole("link", { name: "Budgets" }).click();
    await page.getByRole("button", { name: "Create your first budget" }).click();
    await page.getByLabel("Name (optional)").fill("e2e-budget");
    await page.getByLabel("Spending limit (USD)").fill("100");
    await page.getByRole("button", { name: "Create budget" }).click();

    await expect(page.getByRole("cell", { name: "$100.00" })).toBeVisible();
    await expect(page.getByText("e2e-budget")).toBeVisible();
    await expect(page.getByText("No budgets yet")).toBeHidden();
  });

  test("create a user and assign the budget", async ({ page }) => {
    await login(page);
    await nav(page).getByRole("link", { name: "Users" }).click();
    // A bootstrap virtual user already exists (from the first-run key), so use the
    // header action, not the empty-state button. It is removed when the form opens,
    // leaving the form's own "Create user" as the only match.
    await page.getByRole("button", { name: "Create user" }).click();
    await page.getByLabel("User ID").fill("alice@example.com");
    // The budget created by the prior test is the only non-default option.
    await page.getByLabel("Budget").selectOption({ index: 1 });
    await page.getByRole("button", { name: "Create user" }).click();

    const row = page.getByRole("row", { name: /alice@example\.com/ });
    await expect(row).toBeVisible();
    // The assigned budget's name renders in the user's Budget cell.
    await expect(row.getByText("e2e-budget")).toBeVisible();
  });

  test("create an API key owned by a chosen user", async ({ page }) => {
    await login(page);
    await nav(page).getByRole("link", { name: "API keys" }).click();
    // A bootstrap key already exists, so use the header action, not onboarding.
    await page.getByRole("button", { name: "Create key" }).click();
    await page.getByLabel("Name").fill("ci-bot");
    // Owner is required (user-first). Reuse the user created earlier; type it and
    // close the combobox popover so it does not aria-hide the submit button.
    await page.getByPlaceholder("Pick a user, or type a new id…").fill("alice@example.com");
    await page.keyboard.press("Escape");
    await page.getByRole("button", { name: "Create key" }).click();

    // The one-time reveal appears; acknowledge it.
    await page.getByRole("button", { name: /saved this key/i }).click();

    const row = page.getByRole("row", { name: /ci-bot/ });
    await expect(row).toBeVisible();
    // The key is owned by the named user, not an anonymous virtual one.
    await expect(row.getByText("alice@example.com")).toBeVisible();
  });

  test("create an alias", async ({ page }) => {
    await login(page);
    await nav(page).getByRole("link", { name: "Aliases" }).click();
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
