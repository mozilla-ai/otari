import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { SetPriceDialog } from "@/features/models/SetPriceDialog"
import { renderWithRouter } from "@/tests/router"

// Rendered open with no model key field, so no request is made: the dialog
// hands its rates to `onSubmit`, which each caller wires to its own endpoint.
async function renderDialog(
  props: Partial<Parameters<typeof SetPriceDialog>[0]> = {},
) {
  const onSubmit = vi.fn(async () => undefined)
  await renderWithRouter(
    <SetPriceDialog
      isOpen
      onOpenChange={vi.fn()}
      onSubmit={onSubmit}
      {...props}
    />,
  )
  return onSubmit
}

describe("SetPriceDialog", () => {
  it("offers no unit when repricing imported rows, which are costed by token", async () => {
    await renderDialog()

    expect(screen.queryByRole("radio", { name: "Requests" })).toBeNull()
    expect(screen.getByLabelText("Input $ / 1M")).toBeInTheDocument()
  })

  it("sends a token price without a unit where none was offered", async () => {
    const user = userEvent.setup()
    const onSubmit = await renderDialog()

    await user.type(screen.getByLabelText("Input $ / 1M"), "1")
    await user.type(screen.getByLabelText("Output $ / 1M"), "2")
    await user.click(screen.getByRole("button", { name: "Set price" }))

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith(
        { input_price_per_million: 1, output_price_per_million: 2 },
        "",
      ),
    )
  })

  it("switches to one per-thousand rate for a request price", async () => {
    const user = userEvent.setup()
    const onSubmit = await renderDialog({ chooseUnit: true })

    expect(screen.getByRole("radio", { name: "Tokens" })).toBeChecked()
    await user.click(screen.getByRole("radio", { name: "Requests" }))
    expect(screen.queryByLabelText("Output $ / 1M")).toBeNull()
    expect(screen.queryByLabelText("Cache read $ / 1M")).toBeNull()
    await user.type(screen.getByLabelText("Price per 1,000 requests"), "2")
    await user.click(screen.getByRole("button", { name: "Set price" }))

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith(
        {
          input_price_per_million: 2000,
          output_price_per_million: 0,
          unit: "requests",
        },
        "",
      ),
    )
  })

  it("opens in the unit it is given", async () => {
    await renderDialog({ chooseUnit: true, initialUnit: "images" })

    expect(screen.getByRole("radio", { name: "Images" })).toBeChecked()
    expect(screen.getByLabelText("Price per 1,000 images")).toBeInTheDocument()
  })
})
