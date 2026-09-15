import { act, render, waitFor } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { SetupOrb } from "@/features/onboarding/SetupOrb"

vi.mock("thinking-orbs", () => ({
  ThinkingOrb: ({ state, paused }: { state: string; paused: boolean }) => (
    <canvas aria-label={state} data-paused={paused} />
  ),
}))

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

it("crossfades from searching to working, then removes the outgoing canvas", async () => {
  const { container, rerender } = render(<SetupOrb phase="waiting" />)
  await waitFor(() =>
    expect(container.querySelectorAll("canvas")).toHaveLength(1),
  )
  vi.useFakeTimers()
  rerender(<SetupOrb phase="checking" />)
  expect(container.querySelectorAll("canvas")).toHaveLength(2)
  expect(container.querySelector('[aria-label="working"]')).toBeInTheDocument()
  await act(async () => {
    await vi.advanceTimersByTimeAsync(420)
  })
  expect(container.querySelectorAll("canvas")).toHaveLength(1)
  expect(container.querySelector('[aria-label="working"]')).toBeInTheDocument()
})

it("switches directly between frozen phases with reduced motion", async () => {
  vi.stubGlobal("matchMedia", () => ({
    matches: true,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  }))
  const { container, rerender } = render(<SetupOrb phase="waiting" />)
  await waitFor(() =>
    expect(container.querySelectorAll("canvas")).toHaveLength(1),
  )
  rerender(<SetupOrb phase="checking" />)
  expect(container.querySelectorAll("canvas")).toHaveLength(1)
  expect(container.querySelector("canvas")).toHaveAttribute(
    "data-paused",
    "true",
  )
  expect(container.querySelector("canvas")).toHaveAttribute(
    "aria-label",
    "working",
  )
})
