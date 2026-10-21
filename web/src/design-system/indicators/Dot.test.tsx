import { render } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { Dot } from "./Dot"

describe("Dot", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("starts a pulsing dot's ring on the document's clock, so every dot pings in step", () => {
    // jsdom runs no animations; the stub stands in for the ring's CSS animation.
    const ring = { startTime: 1234 as number | null }
    const getAnimations = vi.fn(() => [ring])
    Object.defineProperty(HTMLElement.prototype, "getAnimations", {
      configurable: true,
      value: getAnimations,
    })
    const { container } = render(<Dot pulse className="bg-success" />)
    expect(container.firstElementChild).toHaveClass("otari-dot-pulse")
    expect(getAnimations).toHaveBeenCalledWith({ subtree: true })
    expect(ring.startTime).toBe(0)
    delete (HTMLElement.prototype as { getAnimations?: unknown }).getAnimations
  })

  it("leaves a still dot without a ring", () => {
    const { container } = render(<Dot className="bg-success" />)
    expect(container.firstElementChild).not.toHaveClass("otari-dot-pulse")
  })
})
