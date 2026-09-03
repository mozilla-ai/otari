import { fireEvent, render } from "@testing-library/react"
import { useState } from "react"
import { describe, expect, it } from "vitest"

import { TableScrollFrame } from "./surface"

/**
 * The cue that appears with the state and leaves with it.
 *
 * A table at rest has no internal verticals, so the pinned lane's boundary is
 * drawn only while the table is scrolled off its left edge, and everything
 * downstream of that is a CSS rule keyed on `data-scrolled`. Which means the
 * attribute is the whole contract: if it is missing, stale, or never flips, the
 * boundary is simply always there or never there and nothing throws.
 *
 * The listener has to go on `.table__scroll-container`, which is HeroUI's
 * element and not something a call site can reach, so these tests stand in a
 * div with that class rather than mounting a real table.
 */
describe("TableScrollFrame", () => {
  const frame = (container: HTMLElement) =>
    container.firstElementChild as HTMLElement

  const Scroller = ({ scrollLeft = 0 }: { scrollLeft?: number }) => (
    <div
      className="table__scroll-container"
      ref={(el) => {
        // Refs are attached before effects run, so this is the only way to
        // present the frame with a scroller that is already off its left edge
        // at the moment its effect first reads one.
        if (el) el.scrollLeft = scrollLeft
      }}
    />
  )

  it("reports the resting state on mount rather than waiting for a scroll", () => {
    // Without the mount-time sync the attribute is absent until the first
    // scroll event, and a CSS rule keyed on `false` would not match a table
    // nobody has touched yet.
    const { container } = render(
      <TableScrollFrame className="otari-keys-table">
        <Scroller />
      </TableScrollFrame>,
    )
    expect(frame(container).dataset.scrolled).toBe("false")
  })

  it("reports a table that mounts already scrolled, before any event fires", () => {
    // Restored filters and deep links both land on a table whose scroller has a
    // non-zero offset before anybody has scrolled it in this session.
    const { container } = render(
      <TableScrollFrame className="otari-keys-table">
        <Scroller scrollLeft={120} />
      </TableScrollFrame>,
    )
    expect(frame(container).dataset.scrolled).toBe("true")
  })

  it("follows the table across the left edge and back", () => {
    const { container } = render(
      <TableScrollFrame className="otari-keys-table">
        <Scroller />
      </TableScrollFrame>,
    )
    const scroller = container.querySelector(
      ".table__scroll-container",
    ) as HTMLElement
    scroller.scrollLeft = 120
    fireEvent.scroll(scroller)
    expect(frame(container).dataset.scrolled).toBe("true")
    scroller.scrollLeft = 0
    fireEvent.scroll(scroller)
    expect(frame(container).dataset.scrolled).toBe("false")
  })

  it("keeps its own class, because that is what the page's CSS selects on", () => {
    const { container } = render(
      <TableScrollFrame className="otari-keys-table">
        <Scroller />
      </TableScrollFrame>,
    )
    expect(frame(container).className).toBe("otari-keys-table")
  })

  it("keeps working across a re-render", () => {
    // Every one of these frames sits inside a page that re-renders on filter,
    // pagination and query settle, so a listener that survives only the first
    // render would look correct in isolation and dead in the product.
    const Harness = () => {
      const [n, setN] = useState(0)
      return (
        <>
          <button type="button" onClick={() => setN(n + 1)}>
            rerender {n}
          </button>
          <TableScrollFrame className="otari-keys-table">
            <Scroller />
          </TableScrollFrame>
        </>
      )
    }
    const { container, getByRole } = render(<Harness />)
    fireEvent.click(getByRole("button"))
    fireEvent.click(getByRole("button"))

    const scroller = container.querySelector(
      ".table__scroll-container",
    ) as HTMLElement
    scroller.scrollLeft = 120
    fireEvent.scroll(scroller)
    expect(
      (container.querySelector(".otari-keys-table") as HTMLElement).dataset
        .scrolled,
    ).toBe("true")
  })

  it("leaves exactly one listener behind however often it renders", () => {
    // The effect re-runs on every render, so the guarantee is not "attached
    // once" but "cleaned up as often as attached". Counting the net is what
    // stays true whether or not the effect later grows a dependency array.
    const added: unknown[] = []
    const removed: unknown[] = []
    const Harness = () => {
      const [n, setN] = useState(0)
      return (
        <>
          <button type="button" onClick={() => setN(n + 1)}>
            rerender {n}
          </button>
          <TableScrollFrame className="otari-keys-table">
            <div
              className="table__scroll-container"
              ref={(el) => {
                if (!el || "__counted" in el) return
                Object.assign(el, { __counted: true })
                const add = el.addEventListener.bind(el)
                const remove = el.removeEventListener.bind(el)
                el.addEventListener = (...args: Parameters<typeof add>) => {
                  if (args[0] === "scroll") added.push(args[1])
                  return add(...args)
                }
                el.removeEventListener = (
                  ...args: Parameters<typeof remove>
                ) => {
                  if (args[0] === "scroll") removed.push(args[1])
                  return remove(...args)
                }
              }}
            />
          </TableScrollFrame>
        </>
      )
    }
    const { getByRole, unmount } = render(<Harness />)
    fireEvent.click(getByRole("button"))
    fireEvent.click(getByRole("button"))
    expect(added.length - removed.length).toBe(1)

    unmount()
    expect(added.length - removed.length).toBe(0)
  })

  it("does nothing at all when there is no scroller inside it", () => {
    // Some callers render an empty state instead of a table, and HeroUI's
    // element is simply not there. The frame still has to mount.
    const { container } = render(
      <TableScrollFrame className="otari-keys-table">
        <p>No keys yet.</p>
      </TableScrollFrame>,
    )
    expect(frame(container).dataset.scrolled).toBeUndefined()
  })
})
