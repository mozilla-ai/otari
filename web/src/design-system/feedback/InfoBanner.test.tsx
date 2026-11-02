import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { InfoBanner } from "./InfoBanner"

describe("InfoBanner", () => {
  it("offers no close control for a standing condition", () => {
    render(<InfoBanner>Model discovery is off.</InfoBanner>)

    expect(screen.getByText("Model discovery is off.")).toBeInTheDocument()
    expect(screen.queryByRole("button")).not.toBeInTheDocument()
  })

  it("offers a named close control when the caller can put it away", async () => {
    const onDismiss = vi.fn()
    render(
      <InfoBanner onDismiss={onDismiss} dismissLabel="Dismiss the invitation">
        A line the reader may put away.
      </InfoBanner>,
    )

    await userEvent.click(
      screen.getByRole("button", { name: "Dismiss the invitation" }),
    )

    expect(onDismiss).toHaveBeenCalledTimes(1)
  })

  it("names the close control for what it does when no label is given", () => {
    render(<InfoBanner onDismiss={vi.fn()}>A line.</InfoBanner>)

    expect(screen.getByRole("button", { name: "Dismiss" })).toBeInTheDocument()
  })
})
