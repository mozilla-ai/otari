import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import { RadioGroup } from "./RadioGroup"

const OPTIONS = [
  { value: "30d", label: "30 days" },
  { value: "1y", label: "1 year" },
  { value: "forever", label: "Keep everything" },
]

const LABEL = "Usage retention"

function Live({ initial = "30d" }: { initial?: string }) {
  const [value, setValue] = useState(initial)
  return (
    <RadioGroup
      label={LABEL}
      value={value}
      onChange={setValue}
      options={OPTIONS}
    />
  )
}

describe("RadioGroup", () => {
  it("renders a real radio input per option", () => {
    // The bug this exists for: only HeroUI's compound Radio root was mounted, so
    // no `role=radio` reached the DOM and the options were dead text.
    render(<Live />)
    const radios = screen.getAllByRole("radio")
    expect(radios).toHaveLength(OPTIONS.length)
    for (const option of OPTIONS) {
      expect(
        screen.getByRole("radio", { name: option.label }),
      ).toBeInTheDocument()
    }
  })

  it("marks the group's value as checked", () => {
    render(<Live initial="1y" />)
    expect(screen.getByRole("radio", { name: "1 year" })).toBeChecked()
    expect(screen.getByRole("radio", { name: "30 days" })).not.toBeChecked()
  })

  it("moves the selection when a non-default option is pressed", async () => {
    const user = userEvent.setup()
    render(<Live />)

    await user.click(screen.getByRole("radio", { name: "Keep everything" }))

    expect(screen.getByRole("radio", { name: "Keep everything" })).toBeChecked()
    expect(screen.getByRole("radio", { name: "30 days" })).not.toBeChecked()
  })

  it("names the group by its visible label", () => {
    render(<Live />)
    expect(screen.getByRole("radiogroup", { name: LABEL })).toBeInTheDocument()
  })
})
