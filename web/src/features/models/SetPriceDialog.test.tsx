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
  it("offers no unit unless asked to, and prices by token", async () => {
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
          unit: "request",
        },
        "",
      ),
    )
  })

  it("sends a second of video as the block a video is billed from", async () => {
    const user = userEvent.setup()
    const onSubmit = await renderDialog({ chooseUnit: true })

    await user.click(screen.getByRole("radio", { name: "Seconds of video" }))
    await user.type(screen.getByLabelText("Price per second of video"), "0.06")
    await user.click(screen.getByRole("button", { name: "Set price" }))

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith(
        {
          input_price_per_million: 60000,
          output_price_per_million: 0,
          unit: "second",
          rates: { video: { output: { price: 0.06, unit: "second" } } },
        },
        "",
      ),
    )
  })

  it("sends a minute of audio's input and output rates", async () => {
    const user = userEvent.setup()
    const onSubmit = await renderDialog({ chooseUnit: true })

    await user.click(screen.getByRole("radio", { name: "Minutes of audio" }))
    await user.type(screen.getByLabelText("Audio in, per minute"), "0.003")
    await user.type(screen.getByLabelText("Audio out, per minute"), "0.015")
    await user.click(screen.getByRole("button", { name: "Set price" }))

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith(
        {
          input_price_per_million: 3000,
          output_price_per_million: 15000,
          unit: "minute",
        },
        "",
      ),
    )
  })

  it("sends a search price per thousand as the per-million fields", async () => {
    const user = userEvent.setup()
    const onSubmit = await renderDialog({ chooseUnit: true })

    await user.click(screen.getByRole("radio", { name: "Web searches" }))
    await user.type(screen.getByLabelText("Price per 1,000 searches"), "10")
    await user.click(screen.getByRole("button", { name: "Set price" }))

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith(
        {
          input_price_per_million: 10000,
          output_price_per_million: 0,
          unit: "web_search",
        },
        "",
      ),
    )
  })

  it("opens in the unit it is given", async () => {
    await renderDialog({ chooseUnit: true, initialUnit: "image" })

    expect(screen.getByRole("radio", { name: "Images" })).toBeChecked()
    expect(screen.getByLabelText("Price per 1,000 images")).toBeInTheDocument()
  })
})
