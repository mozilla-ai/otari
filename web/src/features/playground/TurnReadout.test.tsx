import { render, screen } from "@testing-library/react"
import { expect, it } from "vitest"
import { TurnReadout } from "./TurnReadout"

it("keeps unknown cost and timing out of the readout", () => {
  render(
    <TurnReadout
      usage={{
        promptTokens: 12,
        completionTokens: 4,
        cachedTokens: 0,
        totalMs: 1000,
        ttftMs: undefined,
        tokensPerSecond: undefined,
        costUsd: undefined,
      }}
    />,
  )
  expect(screen.getByText("12 in · 4 out · 1.0s")).toBeInTheDocument()
  expect(screen.queryByText(/\$/)).not.toBeInTheDocument()
  expect(screen.queryByText(/ttft/)).not.toBeInTheDocument()
  expect(screen.queryByText(/cached/)).not.toBeInTheDocument()
})
