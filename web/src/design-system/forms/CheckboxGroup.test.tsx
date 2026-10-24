import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it, vi } from "vitest"

import { CheckboxGroup } from "./CheckboxGroup"

const OPTIONS = [
  { value: "1", label: "Monday" },
  { value: "5", label: "Friday" },
  { value: "6", label: "Saturday", isDisabled: true },
]

function Harness({
  initial = [],
  onChange,
  ...rest
}: {
  initial?: string[]
  onChange?: (next: string[]) => void
} & Partial<Parameters<typeof CheckboxGroup>[0]>) {
  const [value, setValue] = useState<string[]>(initial)
  return (
    <CheckboxGroup
      label="Reset on"
      value={value}
      options={OPTIONS}
      onChange={(next) => {
        setValue(next)
        onChange?.(next)
      }}
      {...rest}
    />
  )
}

describe("CheckboxGroup", () => {
  it("names the group and every option", () => {
    render(<Harness />)
    const group = screen.getByRole("group", { name: "Reset on" })
    expect(
      within(group).getByRole("checkbox", { name: "Monday" }),
    ).toBeInTheDocument()
    expect(
      within(group).getByRole("checkbox", { name: "Friday" }),
    ).toBeInTheDocument()
  })

  it("reports every selected value, not only the one pressed", async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Harness initial={["1"]} onChange={onChange} />)

    await user.click(screen.getByRole("checkbox", { name: "Friday" }))
    expect(onChange).toHaveBeenCalledWith(["1", "5"])
  })

  it("unchecks a selected option rather than re-adding it", async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Harness initial={["1", "5"]} onChange={onChange} />)

    await user.click(screen.getByRole("checkbox", { name: "Monday" }))
    expect(onChange).toHaveBeenCalledWith(["5"])
  })

  it("marks the selected options checked", () => {
    render(<Harness initial={["5"]} />)
    expect(screen.getByRole("checkbox", { name: "Monday" })).not.toBeChecked()
    expect(screen.getByRole("checkbox", { name: "Friday" })).toBeChecked()
  })

  it("refuses a disabled option", async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(<Harness onChange={onChange} />)

    const saturday = screen.getByRole("checkbox", { name: "Saturday" })
    expect(saturday).toBeDisabled()
    await user.click(saturday)
    expect(onChange).not.toHaveBeenCalled()
  })

  it("disables every option when the group is disabled", () => {
    render(<Harness isDisabled />)
    for (const name of ["Monday", "Friday"]) {
      expect(screen.getByRole("checkbox", { name })).toBeDisabled()
    }
  })

  it("announces the error on the group, and only while invalid", () => {
    const { rerender } = render(
      <Harness errorMessage="Pick at least one weekday." />,
    )
    expect(screen.queryByText("Pick at least one weekday.")).toBeNull()

    rerender(<Harness isInvalid errorMessage="Pick at least one weekday." />)
    expect(
      screen.getByRole("group", { name: "Reset on" }),
    ).toHaveAccessibleDescription("Pick at least one weekday.")
  })

  it("keeps the label for assistive technology when it is hidden", () => {
    render(<Harness hideLabel />)
    expect(screen.getByRole("group", { name: "Reset on" })).toBeInTheDocument()
  })

  it("describes each option with its own description", () => {
    render(
      <Harness
        options={[
          { value: "a", label: "Email", description: "To the owners." },
          { value: "b", label: "Banner", description: "While over budget." },
        ]}
      />,
    )
    expect(
      screen.getByRole("checkbox", { name: "Email" }),
    ).toHaveAccessibleDescription("To the owners.")
    expect(
      screen.getByRole("checkbox", { name: "Banner" }),
    ).toHaveAccessibleDescription("While over budget.")
  })

  it("describes the group without joining an option's name", () => {
    render(<Harness description="Resets at 00:00 UTC." />)
    expect(screen.getByText("Resets at 00:00 UTC.")).toBeInTheDocument()
    expect(
      screen.getByRole("checkbox", { name: "Monday" }),
    ).toHaveAccessibleName("Monday")
  })
})
