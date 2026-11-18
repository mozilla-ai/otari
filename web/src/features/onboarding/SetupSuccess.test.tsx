import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { expect, it, vi } from "vitest"

import type { ActivationAttempt } from "@/client"
import { Dialog } from "@/design-system/feedback/Dialog"
import { setupSuccessFace } from "@/features/onboarding/SetupSuccess"
import { activationAttempt } from "@/tests/fixtures"

function renderFace(attempt?: ActivationAttempt, onOpenActivity = vi.fn()) {
  render(
    <Dialog
      isOpen
      onOpenChange={vi.fn()}
      size="lg"
      {...setupSuccessFace({
        attempt,
        workspaceName: "Acme",
        onOpenActivity,
      })}
    />,
  )
  return onOpenActivity
}

it("reports the request that landed, in the workspace it landed in", async () => {
  renderFace(activationAttempt())

  const dialog = await screen.findByRole("dialog", {
    name: "Your first call went through",
  })
  expect(dialog).toHaveTextContent("It landed in Acme.")
  expect(within(dialog).getByText("412 ms")).toBeInTheDocument()
  expect(within(dialog).getByText("openai:gpt-4o-mini")).toBeInTheDocument()
  expect(within(dialog).getByText("openai")).toBeInTheDocument()
  expect(
    within(dialog).getByText("77777777-7777-7777-7777-777777777777"),
  ).toBeInTheDocument()
  expect(within(dialog).getByRole("status")).toHaveTextContent("Setup complete")
})

it("drops what the gateway did not report rather than showing it empty", async () => {
  renderFace(
    activationAttempt({ latency_ms: null, cost_usd: null, provider: null }),
  )

  const dialog = await screen.findByRole("dialog")
  expect(within(dialog).queryByText("Latency")).not.toBeInTheDocument()
  expect(within(dialog).queryByText("Cost")).not.toBeInTheDocument()
  expect(within(dialog).queryByText("Provider")).not.toBeInTheDocument()
  expect(within(dialog).getByText("Model")).toBeInTheDocument()
})

it("still says setup is complete when the gateway reported no attempt", async () => {
  renderFace(undefined)

  const dialog = await screen.findByRole("dialog", {
    name: "Your first call went through",
  })
  expect(within(dialog).queryByText("The request")).not.toBeInTheDocument()
  expect(within(dialog).getByRole("status")).toHaveTextContent("Setup complete")
})

it("continues to the activity log", async () => {
  const user = userEvent.setup()
  const onOpenActivity = renderFace(activationAttempt())

  await user.click(
    await screen.findByRole("button", { name: "Continue to the activity log" }),
  )
  expect(onOpenActivity).toHaveBeenCalledOnce()
})
