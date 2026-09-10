import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { Select } from "./Select"

const OPTIONS = [
  { value: "fallback", label: "Fallback" },
  { value: "round-robin", label: "Round robin" },
]

describe("Select", () => {
  it("offers only the caller's options when nothing is selected", async () => {
    // The empty string means "nothing selected" here, not a value to preserve.
    // Carrying it as its own option put a blank row at the top of the list that
    // an operator could pick, which is what this pins.
    render(
      <Select
        label="Routing strategy"
        value=""
        onChange={() => {}}
        options={OPTIONS}
      />,
    )
    await userEvent.click(screen.getByRole("button"))
    const options = await screen.findAllByRole("option")
    expect(options.map((option) => option.textContent)).toEqual([
      "Fallback",
      "Round robin",
    ])
  })

  it("carries a value no option holds, so the field says what is set", async () => {
    // A URL naming a withdrawn choice, or a stored setting whose option is
    // gone. react-aria would otherwise print its own "Select an item" where the
    // current value belongs.
    render(
      <Select
        label="Routing strategy"
        value="cheapest-first"
        onChange={() => {}}
        options={OPTIONS}
      />,
    )
    await userEvent.click(screen.getByRole("button"))
    const options = await screen.findAllByRole("option")
    expect(options.map((option) => option.textContent)).toEqual([
      "cheapest-first",
      "Fallback",
      "Round robin",
    ])
  })

  it("reports the value rather than the key react-aria carries", async () => {
    const onChange = vi.fn()
    render(
      <Select
        label="Routing strategy"
        value=""
        onChange={onChange}
        options={OPTIONS}
      />,
    )
    await userEvent.click(screen.getByRole("button"))
    await userEvent.click(
      await screen.findByRole("option", { name: "Fallback" }),
    )
    // Not "v:fallback": the prefix exists so an empty value is distinguishable
    // from an unmade choice, and it comes back off on the way out.
    expect(onChange).toHaveBeenCalledWith("fallback")
  })
})
