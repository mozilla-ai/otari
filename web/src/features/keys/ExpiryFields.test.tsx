import { fireEvent, render, screen } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import { ExpiryFields } from "./ExpiryFields"

function Harness({ initial = "" }: { initial?: string }) {
  const [value, setValue] = useState(initial)
  return (
    <>
      <ExpiryFields
        label="Expires"
        value={value}
        onChange={setValue}
        description="Leave blank for a key that never expires."
      />
      <output data-testid="value">{value}</output>
    </>
  )
}

const dateInput = () => screen.getByLabelText("Expires")
const timeInput = () => screen.getByLabelText("Time")
const value = () => screen.getByTestId("value").textContent

describe("ExpiryFields", () => {
  it("expires at midnight when only a date is picked", () => {
    render(<Harness />)
    expect(timeInput()).toHaveValue("00:00")

    fireEvent.change(dateInput(), { target: { value: "2026-09-23" } })

    expect(value()).toBe("2026-09-23T00:00")
  })

  it("keeps a time chosen before the date", () => {
    render(<Harness />)
    fireEvent.change(timeInput(), { target: { value: "14:30" } })
    expect(value()).toBe("")

    fireEvent.change(dateInput(), { target: { value: "2026-09-23" } })

    expect(value()).toBe("2026-09-23T14:30")
  })

  it("clears the expiry when the date is cleared", () => {
    render(<Harness initial="2026-09-23T09:15" />)
    expect(timeInput()).toHaveValue("09:15")

    fireEvent.change(dateInput(), { target: { value: "" } })

    expect(value()).toBe("")
    expect(timeInput()).toHaveValue("09:15")
  })
})
