import { expect, test, type Page } from "@playwright/test";

import {
  addFilterValue,
  filterChip,
  gotoRoute,
  login,
  openFilterPickers,
  table,
  tableRows,
} from "./helpers";
import { COUNTS, PARITY, UNPRICED_MODEL_KEY } from "./parity-data";

// The bulk-action test destroys the rows it selects, so the flows run in order.
test.describe.configure({ mode: "serial" });

const rows = (page: Page) => tableRows(page, "Activity log");

// Every assertion in this file is scoped to the fixture's own source. The log is
// gateway-wide and the suite shares one database with the onboarding flows, so an
// unscoped count would assert on whatever ran before it rather than on the
// filter under test.
const SCOPED = `/activity?source=${PARITY.source}`;

// Every fixture row, which is one page at the default size. Asserted with
// `toHaveCount` rather than read with `count()`: the log is fetched after the
// route resolves, so a bare read races the first paint and sees an empty table.
const ALL = COUNTS.priced + COUNTS.unpriced + COUNTS.errors + COUNTS.scratch;

// The detail panel is rendered as an extra row under the one that was clicked
// (DataTable's renderDetail), so it is reached through the wrapper that row
// injects rather than by role: it is a panel, not a landmark.
const detailPanel = (page: Page) => page.locator(".otari-detail-reveal");

// Open a row's inline detail. Clicking the row header (the Model cell) rather
// than the row keeps the press off the selection checkbox, which would toggle
// selection instead of firing the row action.
async function openDetail(page: Page, row: ReturnType<typeof rows>): Promise<void> {
  await row.getByRole("rowheader").click();
  await expect(detailPanel(page).getByText("Request detail")).toBeVisible();
}

test.describe("activity log", () => {
  test("the status filter narrows the log, and its chip clears it", async ({ page }) => {
    await login(page);
    await gotoRoute(page, SCOPED);
    await expect(rows(page)).toHaveCount(ALL);

    await openFilterPickers(page);
    await page.getByLabel("Status").selectOption("error");

    // The chip is the page's own statement that the filter is applied, so it is
    // asserted alongside the rows rather than instead of them.
    await expect(filterChip(page, "Status", "Error")).toBeVisible();
    await expect(rows(page)).toHaveCount(COUNTS.errors);
    for (const row of await rows(page).all()) {
      await expect(row).toContainText("error");
    }

    // Clearing from the chip must restore the full log, not merely blank the
    // select: the chip's ✕ is the only affordance on a narrow viewport, where the
    // picker row is collapsed.
    await filterChip(page, "Status", "Error").getByRole("button").click();
    await expect(filterChip(page, "Status", "Error")).toBeHidden();
    await expect(rows(page)).toHaveCount(ALL);
  });

  test("priced and unpriced requests are two halves of the same window", async ({ page }) => {
    await login(page);

    await gotoRoute(page, `${SCOPED}&priced=true`);
    await expect(filterChip(page, "Priced", "Priced")).toBeVisible();
    await expect(rows(page)).toHaveCount(COUNTS.priced);
    // Only the priced model carries a pricing row, so the partition is by model.
    for (const row of await rows(page).all()) {
      await expect(row).toContainText(PARITY.models.priced.model);
    }

    await gotoRoute(page, `${SCOPED}&priced=false`);
    await expect(rows(page)).toHaveCount(COUNTS.unpriced + COUNTS.errors + COUNTS.scratch);
    // The two halves have to reconcile: a row counted in neither (or in both)
    // would mean "priced" and cost IS NULL had drifted apart.
    for (const row of await rows(page).all()) {
      await expect(row).not.toContainText(PARITY.models.priced.model);
    }
  });

  test("model and user filters compose, and Clear all drops them together", async ({ page }) => {
    await login(page);
    await gotoRoute(page, SCOPED);
    await openFilterPickers(page);

    await addFilterValue(page, "Model", PARITY.models.unpriced.model);
    await expect(filterChip(page, "Model", PARITY.models.unpriced.model)).toBeVisible();
    // The unpriced model carries both the succeeding and the failing rows.
    await expect(rows(page)).toHaveCount(COUNTS.unpriced + COUNTS.errors);

    // Filters intersect rather than replace: this user owns those same rows, so
    // adding them must not change the count, while a user who owns none empties it.
    await addFilterValue(page, "User", PARITY.users.light);
    await expect(rows(page)).toHaveCount(COUNTS.unpriced + COUNTS.errors);
    await addFilterValue(page, "User", PARITY.users.heavy);
    await expect(rows(page)).toHaveCount(COUNTS.unpriced + COUNTS.errors);

    // Clear all drops every filter in one press, the source scoping included, so
    // the log widens past the fixture rather than back to it. The control removes
    // itself once there is nothing left to clear, which is the tightest statement
    // that no filter survived.
    await page.getByRole("button", { name: "Clear all" }).click();
    await expect(page.getByRole("button", { name: "Clear all" })).toBeHidden();
    await expect(filterChip(page, "Model", PARITY.models.unpriced.model)).toBeHidden();
    await expect(filterChip(page, "Source", PARITY.source)).toBeHidden();
    await expect.poll(() => rows(page).count()).toBeGreaterThan(COUNTS.unpriced + COUNTS.errors);
  });

  test("a shared URL restores its filters, page size and page", async ({ page }) => {
    await login(page);
    // The whole filter + pagination state lives in the URL so a narrowed view is
    // shareable and survives the back button.
    await gotoRoute(page, `${SCOPED}&model=${PARITY.models.priced.model}&size=25`);
    await expect(filterChip(page, "Model", PARITY.models.priced.model)).toBeVisible();
    await expect(rows(page)).toHaveCount(25);
    await expect(page.getByText(`1–25 of ${COUNTS.priced}`)).toBeVisible();

    await page.getByRole("button", { name: "Next page" }).click();
    await expect(rows(page)).toHaveCount(COUNTS.priced - 25);
    await expect(page.getByText(`26–${COUNTS.priced} of ${COUNTS.priced}`)).toBeVisible();

    // A bookmarked deep page opens where it was left, rather than snapping back
    // to the first page as the window re-anchors on mount.
    await gotoRoute(page, `${SCOPED}&model=${PARITY.models.priced.model}&size=25&page=1`);
    await expect(page.getByText(`26–${COUNTS.priced} of ${COUNTS.priced}`)).toBeVisible();
  });

  test("narrowing the window drops the rows outside it", async ({ page }) => {
    await login(page);
    await gotoRoute(page, SCOPED);
    await expect(rows(page)).toHaveCount(ALL);

    // The fixture spans the last twenty hours, so an hour of it is a strict
    // subset. The preset re-anchors the window and re-queries; a preset that only
    // repainted the histogram would leave the count unchanged.
    await page.getByRole("button", { name: "1h", exact: true }).click();
    await expect.poll(() => rows(page).count()).toBeLessThan(ALL);

    // "All" is unbounded rather than a wider preset, so every fixture row is back.
    await page.getByRole("button", { name: "All", exact: true }).click();
    await expect(rows(page)).toHaveCount(ALL);
  });

  test("a request's detail names what the row cannot fit", async ({ page }) => {
    await login(page);
    await gotoRoute(page, `${SCOPED}&model=${PARITY.models.priced.model}`);
    await openDetail(page, rows(page).first());

    const detail = detailPanel(page);
    // The provenance a row does not have room for. Endpoint is "external" for an
    // imported row, which is how an operator tells it from gateway traffic.
    await expect(detail.getByText("external", { exact: true })).toBeVisible();
    await expect(detail.getByText(PARITY.source, { exact: true })).toBeVisible();
    await expect(detail.getByText(PARITY.sessions.heavy, { exact: true })).toBeVisible();
    await expect(detail.getByText(PARITY.users.heavy, { exact: true })).toBeVisible();

    // A priced row carries billing meters, so its billed-token figure is real
    // rather than the em-dash an unmetered row shows.
    await expect(detail.getByText("Billed tokens")).toBeVisible();
    await expect(detail.getByText("Cost", { exact: true })).toBeVisible();
    await expect(detail).not.toContainText("This request carries no cost.");

    // The token column splits its total into the composition it was billed on,
    // which is the whole reason the cell is a bar and not a number.
    await expect(
      rows(page).first().getByRole("img", { name: /Token composition:.*Cache read/ }),
    ).toBeVisible();

    await detailPanel(page).getByRole("button", { name: "Close" }).click();
    await expect(detailPanel(page)).toHaveCount(0);
  });

  test("an uncosted request offers to price the model behind it", async ({ page }) => {
    await login(page);
    await gotoRoute(page, `${SCOPED}&model=${PARITY.models.unpriced.model}`);
    await openDetail(page, rows(page).first());

    const detail = detailPanel(page);
    await expect(detail).toContainText("This request carries no cost.");
    // The offer names the pricing key the row bills against, which is
    // `provider:model` and not the bare model the row displays: a price stored
    // under the bare name would never be read.
    await expect(detail.getByText(UNPRICED_MODEL_KEY)).toBeVisible();

    await detail.getByRole("button", { name: "Price this model" }).click();
    const dialog = page.getByRole("alertdialog", { name: "Price this model" });
    await expect(dialog).toBeVisible();
    // Carried over from the request that was open, so the operator does not
    // retype a selector they just read.
    await expect(dialog.getByLabel("Model key")).toHaveValue(UNPRICED_MODEL_KEY);
  });

  test("bulk-selects imported rows and deletes them", async ({ page }) => {
    await login(page);
    // A dedicated model, so consuming these rows cannot make an earlier
    // assertion unreproducible.
    await gotoRoute(page, `${SCOPED}&model=${PARITY.models.scratch.model}`);
    await expect(rows(page)).toHaveCount(COUNTS.scratch);

    // Selection is offered because these are imported rows
    // (counts_toward_budget=false); enforced gateway rows are disabled by design.
    // `force` because the selection box is a styled span drawn over a visually
    // hidden input, so the input is never the hit target for an ordinary click.
    await table(page, "Activity log")
      .getByRole("checkbox", { name: "Select all rows" })
      .check({ force: true });
    const bar = page.getByRole("toolbar", { name: "Bulk actions" });
    await expect(bar).toContainText(`${COUNTS.scratch} selected`);

    await bar.getByRole("button", { name: "Delete" }).click();
    const confirm = page.getByRole("alertdialog");
    await expect(confirm).toContainText(`Delete ${COUNTS.scratch} imported rows?`);
    await confirm.getByRole("button", { name: "Delete" }).click();

    // Asserted through the empty state rather than a row count of zero: an empty
    // DataTable still renders one row, carrying the empty message.
    await expect(table(page, "Activity log")).toContainText("No requests match these filters.");
    await expect(table(page, "Activity log")).not.toContainText(PARITY.models.scratch.model);
  });
});
