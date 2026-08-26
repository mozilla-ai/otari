import { screen } from "@testing-library/react"
import type { UserEvent } from "@testing-library/user-event"

// A FilterSelect is a HeroUI Select, so its control is a button that opens a
// listbox popover rather than a native <select>: `user.selectOptions` cannot
// drive it, and react-aria names the button with the current value *and* the
// label ("Error Status"), which is why these match the end of the name.
const named = (label: string) => ({
  name: (name: string) => name === label || name.endsWith(` ${label}`),
})

/** The trigger of the `FilterSelect` labeled `label`. */
export function selectTrigger(label: string) {
  return screen.getByRole("button", named(label))
}

/**
 * Pick an option from a `FilterSelect`: open it, then click the option, which
 * is named by its visible label rather than by its value.
 */
export async function pickOption(
  user: UserEvent,
  select: string | HTMLElement,
  option: string | RegExp,
) {
  await user.click(
    typeof select === "string"
      ? await screen.findByRole("button", named(select))
      : select,
  )
  await user.click(await screen.findByRole("option", { name: option }))
}
