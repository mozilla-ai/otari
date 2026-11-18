import { act, render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ContextMapView } from "./ContextMapView"
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
  it("draws a focusable block for each part and one for the free window", () => {
    render(<ContextMapView map={estimated()} />)
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
        name: /^Free: 160\.4k tokens · 80\.2%/,
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
    expect(tooltip).toHaveTextContent(/^Tool definitionsTools ·/)
    expect(within(tooltip).getByText("12 tool definitions")).toBeInTheDocument()
    // The band's note says once that the split is estimated; the card does not repeat it.
    expect(
      within(tooltip).queryByText("Estimated from request size"),
    ).not.toBeInTheDocument()
  })

  it("opens a block's card on a click or a tap, which has no hover to open it", async () => {
    const user = userEvent.setup()
    render(<ContextMapView map={estimated()} />)

    await user.click(screen.getByRole("button", { name: /^Messages/ }))

    const tooltip = await screen.findByRole("tooltip")
    expect(within(tooltip).getByText("41 messages")).toBeInTheDocument()
  })

  it("does not repeat a title that is its own category in the tooltip", async () => {
    const user = userEvent.setup()
    render(<ContextMapView map={estimated()} />)
    const results = screen.getByRole("button", { name: /^Tool results/ })

    for (
      let step = 0;
      step < 10 && document.activeElement !== results;
      step += 1
    )
      await user.tab()

    const tooltip = await screen.findByRole("tooltip")
    expect(within(tooltip).getAllByText(/Tool results/)).toHaveLength(1)
    expect(
      within(tooltip).getByText(/^\d+(\.\d)?k tokens · \d+\.\d%$/),
    ).toBeInTheDocument()
  })

  it("leads its note with which request the map is of", () => {
    render(
      <ContextMapView map={estimated()} lead="From the last of 3 requests." />,
    )

    expect(
      screen.getByText(/^From the last of 3 requests\. The split is estimated/),
    ).toBeInTheDocument()
  })

  it("says the split is an estimate and the total is not", () => {
    render(<ContextMapView map={estimated()} />)

    expect(
      screen.getByText(
        "The split is estimated from its size; the total is exact.",
      ),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", {
        name: /^Tool results: .*Estimated from request size$/,
      }),
    ).toBeInTheDocument()
  })

  it("says how much of an estimated map was read from the cache, with no tiles for it", () => {
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
      screen.getByText(/79% of it was read from the cache\.$/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /^Cache read/ }),
    ).not.toBeInTheDocument()
  })

  it("does not call counts that overrun the input exact", () => {
    render(
      <ContextMapView
        map={buildContextMap({
          inputTokens: 100,
          cacheReadTokens: 90,
          cacheWriteTokens: 90,
          windowTokens: 1_000,
          weights: WEIGHTS,
        })}
      />,
    )

    expect(screen.queryByText(/from the cache/)).not.toBeInTheDocument()
    expect(screen.getByText(/The split is estimated/)).toBeInTheDocument()
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

    expect(screen.queryByText(/estimated/)).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", {
        name: /^Cache read: .*Read from the provider's cache$/,
      }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /^Cache write/ }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /^Fresh input/ }),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/^The split is the provider's own\./),
    ).toBeInTheDocument()
  })

  it("says the window is unknown when it is, and draws no free space", () => {
    render(
      <ContextMapView
        map={buildContextMap({ inputTokens: 1_500, weights: WEIGHTS })}
      />,
    )

    expect(
      screen.getByText(
        /This model's window is unknown, so free space is not shown\.$/,
      ),
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
      />,
    )

    expect(
      screen.getByText(/^More than the 200k window the catalog lists/),
    ).toHaveClass("text-warning")
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

  it("never gives a sliver of a block a negative size, padding or border that would grow it past its share", () => {
    render(
      <ContextMapView
        map={buildContextMap({
          inputTokens: 3,
          cacheReadTokens: 1,
          windowTokens: 200_000,
        })}
      />,
    )
    const map = screen.getByRole("group", { name: "Context window map" })

    const tiny = within(map)
      .getAllByRole("button")
      .filter((block) => !block.getAttribute("aria-label")?.startsWith("Free"))
    expect(tiny).toHaveLength(2)
    for (const block of tiny) {
      expect(Number.parseFloat(block.style.width)).toBeGreaterThanOrEqual(0)
      expect(Number.parseFloat(block.style.height)).toBeGreaterThanOrEqual(0)
      expect(block).not.toHaveClass("p-2")
      expect(block).not.toHaveClass("border")
      expect(block).toHaveAccessibleName(/tokens/)
    }
  })

  it("pads a block only where its title is drawn", () => {
    render(<ContextMapView map={estimated()} />)

    const free = screen.getByRole("button", { name: /^Free/ })
    expect(free).toHaveClass("p-2")
    expect(free).toHaveClass("border")
  })
})

describe("ContextMapView motion", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("guards every block's transition for reduced motion and lifts it only on hover", () => {
    render(<ContextMapView map={estimated()} />)

    for (const block of screen.getAllByRole("button")) {
      expect(block).toHaveClass("motion-reduce:transition-none!")
      expect(block).toHaveClass("motion-reduce:hover:translate-y-0")
      expect(block).toHaveClass("hover:-translate-y-px")
      expect(block).not.toHaveClass("focus-visible:-translate-y-px")
      expect(block).not.toHaveClass("hover:shadow-elevation-md")
      expect(block.style.transitionProperty).not.toContain("box-shadow")
    }
  })

  it("staggers the entrance in 30ms steps, delaying nothing but opacity, within 300ms", () => {
    render(<ContextMapView map={estimated()} />)

    const delays = screen
      .getAllByRole("button")
      .map((block) => block.style.transitionDelay.split(", ").at(-1))
    expect(delays).toEqual(["0ms", "30ms", "60ms", "90ms", "120ms", "120ms"])
    const first = screen.getAllByRole("button")[0]
    const rows = first.style.transitionProperty.split(", ")
    expect(rows.at(-1)).toBe("opacity")
    expect(first.style.transitionDelay.split(", ").slice(0, -1)).toEqual(
      rows.slice(0, -1).map(() => "0ms"),
    )
  })

  it("keeps a duration and a curve for each property it transitions", () => {
    render(<ContextMapView map={estimated()} />)
    const style = screen.getAllByRole("button")[0].style

    const count = style.transitionProperty.split(", ").length
    expect(style.transitionDuration.split(", ")).toHaveLength(count)
    expect(style.transitionDelay.split(", ")).toHaveLength(count)
    expect(style.transitionProperty).toContain("width")
  })

  it("snaps the blocks while the container is resized and animates them again when the data changes", () => {
    let fire: (width: number) => void = () => {}
    vi.stubGlobal(
      "ResizeObserver",
      class {
        constructor(private readonly callback: ResizeObserverCallback) {}
        observe(target: Element) {
          fire = (width) =>
            act(() =>
              this.callback(
                [{ target, contentRect: { width } } as ResizeObserverEntry],
                this as unknown as ResizeObserver,
              ),
            )
          fire(800)
        }
        unobserve() {}
        disconnect() {}
      },
    )
    const { rerender } = render(<ContextMapView map={estimated()} />)
    const geometry = () =>
      screen
        .getAllByRole("button")[0]
        .style.transitionProperty.includes("width")

    expect(geometry()).toBe(true)
    fire(500)
    expect(geometry()).toBe(false)
    fire(520)
    expect(geometry()).toBe(false)
    rerender(
      <ContextMapView
        map={buildContextMap({
          inputTokens: 60_000,
          windowTokens: 200_000,
          weights: WEIGHTS,
        })}
      />,
    )
    expect(geometry()).toBe(true)
  })
})

function measuredAt(width: number) {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(private readonly callback: ResizeObserverCallback) {}
      observe(target: Element) {
        this.callback(
          [{ target, contentRect: { width } } as ResizeObserverEntry],
          this as unknown as ResizeObserver,
        )
      }
      unobserve() {}
      disconnect() {}
    },
  )
}

describe("ContextMapView in a narrow panel", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("writes the shorter category on a block too narrow for its title", () => {
    measuredAt(280)
    render(<ContextMapView map={estimated()} />)
    const map = screen.getByRole("group", { name: "Context window map" })

    const tools = within(map).getByRole("button", { name: /^Tool definitions/ })
    expect(tools.textContent).not.toContain("Tool definitions")
  })
})
