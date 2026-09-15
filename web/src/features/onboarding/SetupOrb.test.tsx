import { fireEvent, render, waitFor } from "@testing-library/react"
import { afterEach, expect, it, vi } from "vitest"
import { SetupOrb } from "@/features/onboarding/SetupOrb"

vi.mock("thinking-orbs", () => ({
  ThinkingOrb: ({ state, paused }: { state: string; paused: boolean }) => (
    <canvas aria-label={state} data-paused={paused} />
  ),
}))

afterEach(() => {
  vi.unstubAllGlobals()
})

function finishAnimation(element: HTMLElement) {
  // jsdom lacks AnimationEvent, so React listens for the WebKit event name.
  fireEvent(element, new Event("webkitAnimationEnd", { bubbles: true }))
}

it("crossfades from searching to working, then removes the outgoing canvas", async () => {
  const { container, rerender } = render(<SetupOrb phase="waiting" />)
  await waitFor(() =>
    expect(container.querySelectorAll("canvas")).toHaveLength(1),
  )
  rerender(<SetupOrb phase="checking" />)
  expect(container.querySelectorAll("canvas")).toHaveLength(2)
  expect(container.querySelector('[aria-label="working"]')).toBeInTheDocument()
  const incoming = container.querySelector('[aria-label="working"]')
    ?.parentElement?.parentElement
  expect(incoming).toBeInTheDocument()
  finishAnimation(incoming!)
  expect(container.querySelectorAll("canvas")).toHaveLength(2)
  const outgoing = container.querySelector('[aria-label="searching"]')
    ?.parentElement?.parentElement
  expect(outgoing).toBeInTheDocument()
  finishAnimation(outgoing!)
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
