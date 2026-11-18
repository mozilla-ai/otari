import { screen, waitFor, within } from "@testing-library/react"
import type { UserEvent } from "@testing-library/user-event"
import { expect } from "vitest"

/**
 * A table row's actions live behind its three-dot trigger, named "Row
 * actions" (`RowActionRow`) or "Actions for …" (`RowMenu`), plus the reasons
 * of any refused item. These open those menus the way a person does.
 */
const TRIGGER = /^(Row actions|Actions for )/

/** The three-dot triggers inside `scope`, one per row. */
export function rowMenuTriggers(scope: HTMLElement = document.body) {
  return within(scope).queryAllByRole("button", { name: TRIGGER })
}

async function closeMenu(user: UserEvent) {
  await user.keyboard("{Escape}")
  await waitFor(() => expect(screen.queryByRole("menu")).toBeNull())
}

/**
 * The item named `name` in the first row menu inside `scope` that holds one,
 * left open so it can be pressed or inspected, or null after trying every
 * menu there (each one closed again). The name is the item's accessible
 * name: the action's `ariaLabel` when it has one, otherwise its label as the
 * menu shows it (`Edit…`).
 */
export async function findRowMenuItem(
  user: UserEvent,
  name: string | RegExp,
  scope: HTMLElement = document.body,
): Promise<HTMLElement | null> {
  for (const trigger of rowMenuTriggers(scope)) {
    await user.click(trigger)
    const menu = await screen.findByRole("menu")
    const item = within(menu).queryByRole("menuitem", { name })
    if (item) return item
    await closeMenu(user)
  }
  return null
}

/**
 * Like `findRowMenuItem`, but waits for a row menu to appear first and fails
 * when none holds the item.
 */
export async function getRowMenuItem(
  user: UserEvent,
  name: string | RegExp,
  scope: HTMLElement = document.body,
): Promise<HTMLElement> {
  await waitFor(() => expect(rowMenuTriggers(scope).length).toBeGreaterThan(0))
  const item = await findRowMenuItem(user, name, scope)
  if (!item) throw new Error(`No row menu holds an item named ${String(name)}`)
  return item
}

/** Opens the row menu holding `name` and presses that item. */
export async function pressRowAction(
  user: UserEvent,
  name: string | RegExp,
  scope: HTMLElement = document.body,
) {
  await user.click(await getRowMenuItem(user, name, scope))
}

/** The names of the items in the one row menu inside `scope`, which it closes again. */
export async function rowMenuItemNames(
  user: UserEvent,
  scope: HTMLElement,
): Promise<string[]> {
  const [trigger] = rowMenuTriggers(scope)
  if (!trigger) return []
  await user.click(trigger)
  const menu = await screen.findByRole("menu")
  // An item's own label, not the sentence under it: react-aria names an item
  // by the label it points at unless the action gave it a name of its own.
  const names = within(menu)
    .getAllByRole("menuitem")
    .map(
      (item) =>
        item.getAttribute("aria-label") ??
        document.getElementById(item.getAttribute("aria-labelledby") ?? "")
          ?.textContent ??
        "",
    )
  await closeMenu(user)
  return names
}
