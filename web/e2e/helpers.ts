import {
  type APIResponse,
  expect,
  type Locator,
  type Page,
} from "@playwright/test"

// Matches web/e2e/otari.yml. The login step needs a known key.
export const MASTER_KEY = "e2e-master-key"

// Scope link lookups to the sidebar navigation landmark. The Overview landing
// page has tile-links whose names substring-collide with sidebar items (e.g.
// "Providers healthy", "Active users", "No budgets configured"), so an unscoped
// getByRole("link", { name }) is ambiguous there.
// The sidebar specifically: the header's breadcrumb is a navigation landmark
// too, so an unnamed query now matches both.
export const nav = (page: Page): Locator =>
  page.getByRole("navigation", { name: "Sidebar" })

/**
 * The page's own title, scoped to the content pane.
 *
 * A nav group is a disclosure now, and HeroUI puts its trigger inside a heading
 * (`h3.disclosure__heading`), so the rail contributes headings of its own:
 * `getByRole("heading", { name: "Routing" })` matches the group's row as well as
 * the page's `h1` and fails strict mode with two elements. Scoping to `main` is
 * what `nav()` does in the other direction, and it stays right however the rail's
 * markup changes.
 */
export const pageHeading = (page: Page, name: string): Locator =>
  page.getByRole("main").getByRole("heading", { name, exact: true })

export async function login(page: Page): Promise<void> {
  await page.goto("/")
  await page.locator('input[type="password"]').fill(MASTER_KEY)
  await page.locator('input[type="password"]').press("Enter")
  // The sidebar appears once authenticated, regardless of the index landing
  // page.
  await expect(nav(page).getByRole("link", { name: "Providers" })).toBeVisible()
}

// The dashboard authenticates with a session cookie, but the seeding and
// assertion helpers below talk to the management API directly as the master key,
// which the auth dependencies prefer over the cookie. That keeps them usable
// before `login` has run.
/**
 * Cross from the workspace rail onto the organization one.
 *
 * The two sidebars never render together, so a spec about an organization
 * destination has to enter that context first; the footer entry is the only way
 * in, matching the navigation prototype.
 */
export async function openOrganization(page: Page): Promise<void> {
  await page.getByRole("link", { name: "Organization", exact: true }).click()
  await expect(
    nav(page).getByRole("link", { name: "Members & roles" }),
  ).toBeVisible()
}

/**
 * Expand a sidebar group and open one of the destinations nested under it.
 *
 * Routing and Tools nest their pages, so their sidebar row expands rather than
 * navigating; a spec reaching a child has to open the group first. Idempotent
 * on an already-open group, which is what arriving from a child route leaves.
 */
export async function openNested(
  page: Page,
  group: string,
  child: string,
): Promise<void> {
  const toggle = nav(page).getByRole("button", { name: group, exact: true })
  if ((await toggle.getAttribute("aria-expanded")) !== "true") {
    await toggle.click()
  }
  await nav(page).getByRole("link", { name: child, exact: true }).click()
}

export const authHeaders = {
  Authorization: `Bearer ${MASTER_KEY}`,
  "Content-Type": "application/json",
} as const

/**
 * Retire the first-request setup guide for every workspace on the deployment.
 *
 * A fixture step, not a flow: the guide is offered until a *gateway* request in
 * the workspace succeeds, and the suite's usage is imported rather than served,
 * so without this it sits at the top of the Overview every later spec reads.
 * Permanent and idempotent server-side, so calling it on a warm database is
 * fine.
 */
export async function dismissSetupGuide(page: Page): Promise<void> {
  const listed = await page.request.get("/v1/workspaces", {
    headers: authHeaders,
  })
  await expectOk(listed, "list workspaces")
  const { data } = (await listed.json()) as { data: { id: string }[] }
  for (const workspace of data) {
    const dismissed = await page.request.post(
      `/v1/workspaces/${workspace.id}/activation/dismiss`,
      { headers: authHeaders },
    )
    await expectOk(dismissed, `dismiss the setup guide in ${workspace.id}`)
  }
}

// Open a client route directly. The router runs on hash history, so a
// filtered view is a hash URL and its query string is what `useUrlState` reads.
export async function gotoRoute(page: Page, route: string): Promise<void> {
  await page.goto(`/#${route}`)
}

// Fail with the server's own body rather than a bare status code: a 422 from the
// content-free ingest schema names the offending field, and losing that turns a
// seeding typo into an unexplained assertion failure three tests later.
export async function expectOk(
  response: APIResponse,
  what: string,
): Promise<void> {
  expect(
    response.ok(),
    `${what}: ${response.status()} ${await response.text()}`,
  ).toBe(true)
}

// ---------- filters ----------

// The filter pickers live behind the "Add filter" toggle (see FilterChips), so a
// select or combobox cannot be driven until the region is revealed. Idempotent:
// the toggle reads "Done" once open, so a second call is a no-op rather than a
// close.
export async function openFilterPickers(page: Page): Promise<void> {
  // One control under two names, so waiting on it covers a page that has not
  // painted yet. A bare `isVisible()` on "Add filter" does not wait, so straight
  // after a navigation it reports false and the region is never opened.
  const toggle = page.getByRole("button", { name: /^(Add filter|Done)$/ })
  await expect(toggle).toBeVisible()
  if ((await toggle.innerText()).trim() === "Add filter") {
    await toggle.click()
  }
  await expect(page.getByRole("button", { name: "Done" })).toBeVisible()
}

// An applied filter renders as a chip carrying its dimension and value, whatever
// set it. This is the assertion that a filter was actually applied rather than
// merely typed, and the ✕ that clears it hangs off the same pill.
export function filterChip(page: Page, label: string, value: string): Locator {
  return page
    .locator("span")
    .filter({ hasText: new RegExp(`^${escapeRegExp(`${label}:${value}`)}$`) })
    .first()
}

// Put a combobox's suggestion popover away, and wait until it is actually gone.
//
// This is not cosmetic. React-aria marks the rest of the page `aria-hidden` while
// a popover is open, so every `getByRole` outside it resolves to nothing: a table
// that is plainly on screen reads as zero rows, and a chip that was just added
// reads as absent. Blur as well as Escape, because these boxes open on focus and
// committing a value clears the query, which re-opens the list under the cursor.
export async function dismissComboBox(box: Locator): Promise<void> {
  await box.press("Escape")
  await box.blur()
  await expect(box).not.toHaveAttribute("aria-expanded", "true")
}

// Commit a value into one of the multi-value filter comboboxes. They allow
// custom values, so Enter is what commits a typed id.
export async function addFilterValue(
  page: Page,
  label: string,
  value: string,
): Promise<void> {
  const box = page.getByRole("combobox", { name: label })
  await box.fill(value)
  await box.press("Enter")
  await dismissComboBox(box)
}

// ---------- tables ----------

// The shared DataTable is a react-aria table, which exposes itself as a `table`
// or as a `grid` depending on whether selection is on. Several pages render more
// than one, so a locator has to be scoped; scoping on the aria-label rather than
// on the role keeps one helper working for both shapes.
export function table(page: Page, ariaLabel: string): Locator {
  return page.locator(`[aria-label="${ariaLabel}"]`)
}

// Data rows only. The header is a `row` too, so it is excluded by requiring the
// row-header cell that every data row carries. Note an empty table still renders
// one such row, holding its empty message: assert the message rather than a count
// of zero.
export function tableRows(page: Page, ariaLabel: string): Locator {
  return table(page, ariaLabel)
    .getByRole("row")
    .filter({ has: page.getByRole("rowheader") })
}

// ---------- stat tiles ----------

// A StatCard renders its label and its value as sibling spans with no
// programmatic association between them, so the value is reached through the
// label. Scoped to `within` where a bare label would also match a table column
// header of the same name (Usage has both a "Requests" tile and a "Requests"
// column).
export function tileValue(
  page: Page,
  label: string,
  within?: Locator,
): Locator {
  return (within ?? page)
    .getByText(label, { exact: true })
    .locator("xpath=following-sibling::span[1]")
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}
