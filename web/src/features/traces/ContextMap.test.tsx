import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it } from "vitest"

import { ContextMapView } from "./ContextMap"
import { buildContextMap, type ContextWeights } from "./contextMap"

const WEIGHTS: ContextWeights = {
  system: 4_000,
  tools: 12_000,
  user: 3_000,
  assistant: 5_000,
  toolResults: 30_000,
  files: 4_000,
  toolCount: 12,
  messageCount: 41,
}

function estimated() {
  return buildContextMap({
    inputTokens: 39_600,
    windowTokens: 200_000,
    weights: WEIGHTS,
  })
}

describe("ContextMapView", () => {
  it("heads the map with the model and its window, the tokens used and the share of it", () => {
    render(<ContextMapView map={estimated()} model="claude-sonnet-4-5" />)

    expect(screen.getByText("claude-sonnet-4-5 [200k]")).toBeInTheDocument()
    expect(screen.getByText("39.6k / 200k")).toBeInTheDocument()
    expect(screen.getByText("20%")).toBeInTheDocument()
  })

  it("draws a focusable block for each part and one for the free window", () => {
    render(<ContextMapView map={estimated()} model="claude-sonnet-4-5" />)
    const map = screen.getByRole("group", { name: "Context window map" })

    const blocks = within(map).getAllByRole("button")
    expect(
      blocks
        .map((block) => block.getAttribute("aria-label")?.split(":")[0])
        .sort(),
    ).toEqual([
      "Files",
      "Free",
      "Messages",
      "System prompt",
      "Tool definitions",
      "Tool results",
    ])
    expect(
      within(map).getByRole("button", {
        name: /^Free: Free · 160\.4k tokens · 80\.2%/,
      }),
    ).toBeInTheDocument()
    expect(
      within(map).getByRole("button", {
        name: /^Tool results: .*Estimated from request size/,
      }),
    ).toBeInTheDocument()
    for (const block of blocks)
      expect(block).not.toHaveAttribute("tabindex", "-1")
  })

  it("writes a count on a block big enough, with its category on a larger one that has one to add", () => {
    render(
      <ContextMapView
        map={buildContextMap({
          inputTokens: 100_000,
          cacheReadTokens: 80_000,
          weights: WEIGHTS,
          windowTokens: 200_000,
        })}
      />,
    )
    const map = screen.getByRole("group", { name: "Context window map" })

    // The weights win over the cache split here, so this is a tool result block, whose category is its title.
    const results = within(map).getByRole("button", { name: /^Tool results/ })
    expect(within(results).getByText("Tool results")).toBeInTheDocument()
    expect(results.textContent).toMatch(/Tool results\d+(\.\d)?k$/)
    const tools = within(map).getByRole("button", { name: /^Tool definitions/ })
    expect(tools.textContent).toMatch(/\d+(\.\d)?k · Tools$/)
  })

  // react-aria's hover never fires under jsdom (see Tooltip.test.tsx), so the tooltip opens by keyboard.
  it("names a block's share to one decimal in its tooltip when a keyboard reaches it", async () => {
    const user = userEvent.setup()
    render(<ContextMapView map={estimated()} />)
    const tools = screen.getByRole("button", { name: /^Tool definitions/ })

    for (let step = 0; step < 10 && document.activeElement !== tools; step += 1)
      await user.tab()

    const tooltip = await screen.findByRole("tooltip")
    expect(within(tooltip).getByText("Tool definitions")).toBeInTheDocument()
    expect(
      within(tooltip).getByText(/^Tools · 8\.2k tokens · 4\.1%$/),
    ).toBeInTheDocument()
    expect(within(tooltip).getByText("12 tool definitions")).toBeInTheDocument()
    expect(
      within(tooltip).getByText("Estimated from request size"),
    ).toBeInTheDocument()
  })

  it("says the split is an estimate and the total is not", () => {
    render(<ContextMapView map={estimated()} />)

    expect(screen.getByText("Estimated")).toBeInTheDocument()
    expect(
      screen.getByText(
        /Split estimated from request size, scaled to the provider's exact input-token count/,
      ),
    ).toBeInTheDocument()
  })

  it("states the exact cache counts in text beside an estimated map, with no tiles for them", () => {
    render(
      <ContextMapView
        map={buildContextMap({
          inputTokens: 39_600,
          cacheReadTokens: 31_200,
          cacheWriteTokens: 2_400,
          windowTokens: 200_000,
          weights: WEIGHTS,
        })}
      />,
    )

    expect(
      screen.getByText(
        /Exact from the provider: 31\.2k read from the cache, 2\.4k written to it, 6\.0k fresh input/,
      ),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /^Cache read/ }),
    ).not.toBeInTheDocument()
    expect(screen.getByText("Estimated")).toBeInTheDocument()
  })

  it("draws the cache split with the provider's own counts, and no estimate marker", () => {
    render(
      <ContextMapView
        map={buildContextMap({
          inputTokens: 48_210,
          cacheReadTokens: 41_900,
          cacheWriteTokens: 6_010,
          windowTokens: 200_000,
        })}
      />,
    )

    expect(screen.queryByText("Estimated")).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", {
        name: /^Cache read: .*Read from the provider's cache/,
      }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /^Cache write/ }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /^Fresh input/ }),
    ).toBeInTheDocument()
    expect(screen.getByText(/the provider's own counts/)).toBeInTheDocument()
  })

  it("shows the used tokens and says the window is unknown when it is", () => {
    render(
      <ContextMapView
        map={buildContextMap({ inputTokens: 1_500, weights: WEIGHTS })}
        model="mystery-1"
      />,
    )

    expect(screen.getByText("mystery-1")).toBeInTheDocument()
    expect(screen.getByText("1.5k tokens")).toBeInTheDocument()
    expect(screen.getByText("Window unknown")).toBeInTheDocument()
    expect(
      screen.getByText(/context window of this model is unknown/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /^Free/ }),
    ).not.toBeInTheDocument()
  })

  it("caps an input that overran the window and says why", () => {
    render(
      <ContextMapView
        map={buildContextMap({
          inputTokens: 250_000,
          windowTokens: 200_000,
          cacheReadTokens: 100_000,
        })}
        model="m"
      />,
    )

    expect(screen.getByText(">100%")).toBeInTheDocument()
    expect(screen.getByText(/larger than the 200k window/)).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /^Free/ }),
    ).not.toBeInTheDocument()
  })

  it("says so instead of drawing a map when there are no input tokens", () => {
    render(<ContextMapView map={buildContextMap({ inputTokens: 0 })} />)

    expect(screen.getByText(/recorded no input tokens/)).toBeInTheDocument()
    expect(
      screen.queryByRole("group", { name: "Context window map" }),
    ).not.toBeInTheDocument()
  })

  it("hides the text a small block has no room for but keeps it named", () => {
    render(
      <ContextMapView
        map={buildContextMap({
          inputTokens: 100_000,
          weights: {
            ...WEIGHTS,
            system: 1_000,
            tools: 0,
            files: 0,
            user: 0,
            assistant: 0,
            toolResults: 999_000,
          },
        })}
      />,
    )

    const system = screen.getByRole("button", { name: /^System prompt/ })
    expect(system.textContent).toBe("")
    expect(system).toHaveAccessibleName(/System prompt: System · \d+ tokens/)
  })
})
