import { screen, within } from "@testing-library/react"
import type { UserEvent } from "@testing-library/user-event"

// A FilterSelect is a HeroUI Select, so its control is a button that opens a
// listbox popover rather than a native <select>: `user.selectOptions` cannot
// drive it, and react-aria names the button with the current value *and* the
// label ("Error Status"), which is why these match the end of the name.
//
// The suffix is what makes `scope` matter: one label being the suffix of
// another's on the same page ("Status" against "Error Status") would resolve to
// the wrong control and drive it silently rather than fail to match. No call
// site collides today, but pass a container to search `within` when one does,
// the way `pickOption` in `e2e/helpers.ts` takes its own scope.
const named = (label: string) => ({
  name: (name: string) => name === label || name.endsWith(` ${label}`),
})

/** The trigger of the `FilterSelect` labeled `label`. */
export function selectTrigger(label: string, scope?: HTMLElement) {
  return (scope ? within(scope) : screen).getByRole("button", named(label))
}

/**
 * Pick an option from a `FilterSelect`: open it, then click the option, which
 * is named by its visible label rather than by its value. The popover is
 * portaled to the body, so the option is always found on `screen` even when the
 * trigger was found inside `scope`.
 */
export async function pickOption(
  user: UserEvent,
  select: string | HTMLElement,
  option: string | RegExp,
  scope?: HTMLElement,
) {
  await user.click(
    typeof select === "string"
      ? await (scope ? within(scope) : screen).findByRole(
          "button",
          named(select),
        )
      : select,
  )
  await user.click(await screen.findByRole("option", { name: option }))
}
