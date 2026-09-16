import { render, screen, waitFor } from "@testing-library/react"
import { StrictMode } from "react"
import { expect, it, vi } from "vitest"
import { SetupSuccess } from "@/features/onboarding/SetupSuccess"

const { fire } = vi.hoisted(() => ({
  fire: vi.fn().mockResolvedValue(undefined),
}))
vi.mock("@/features/onboarding/setupConfetti", () => ({
  fireSetupConfetti: fire,
}))

it("anchors the burst to the mounted dialog and cancels it on unmount", async () => {
  const { unmount } = render(
    <StrictMode>
      <SetupSuccess onDismiss={vi.fn()} onOpenActivity={vi.fn()} />
    </StrictMode>,
  )
  const dialog = await screen.findByRole("dialog", {
    name: "Your first call went through",
  })
  await waitFor(() =>
    expect(fire).toHaveBeenCalledWith(dialog, expect.any(AbortSignal)),
  )
  const signals = fire.mock.calls.map(([, signal]) => signal as AbortSignal)
  expect(signals.some((signal) => !signal.aborted)).toBe(true)
  unmount()
  expect(signals.every((signal) => signal.aborted)).toBe(true)
})
