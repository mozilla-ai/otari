import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { ModelDetailsDialog } from "@/features/models/ModelDetailsDialog"
import { renderWithRouter } from "@/tests/router"

async function renderDialog(
  props: Partial<Parameters<typeof ModelDetailsDialog>[0]> = {},
) {
  const onSubmit = vi.fn(async () => undefined)
  await renderWithRouter(
    <ModelDetailsDialog
      isOpen
      onOpenChange={vi.fn()}
      selector="tensorfold:GLM-5.3-Flash"
      current={undefined}
      onSubmit={onSubmit}
      {...props}
    />,
  )
  return onSubmit
}

describe("ModelDetailsDialog", () => {
  it("opens on what is stored, and sends a cleared field as null", async () => {
    const user = userEvent.setup()
    const onSubmit = await renderDialog({
      current: {
        context_window: 1048576,
        max_output_tokens: 65536,
        reasoning: true,
        input_modalities: ["text", "image"],
      },
    })

    expect(screen.getByLabelText("Context window")).toHaveValue("1048576")
    await user.clear(screen.getByLabelText("Max output tokens"))
    await user.click(screen.getByRole("button", { name: "Save details" }))

    await waitFor(() =>
      expect(onSubmit).toHaveBeenCalledWith({
        context_window: 1048576,
        max_output_tokens: null,
        tool_call: null,
        reasoning: true,
        structured_output: null,
        input_modalities: ["text", "image"],
      }),
    )
  })

  it("refuses a token count that is not a whole number", async () => {
    const user = userEvent.setup()
    const onSubmit = await renderDialog()

    await user.type(screen.getByLabelText("Context window"), "128k")

    expect(screen.getByRole("button", { name: "Save details" })).toBeDisabled()
    expect(onSubmit).not.toHaveBeenCalled()
  })
})
