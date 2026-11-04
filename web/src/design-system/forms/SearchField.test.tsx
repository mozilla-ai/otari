import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { startTransition, useState } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { SEARCH_SETTLE_MS, SearchField } from "./SearchField"

// A caller that commits what was typed the way the router commits a search
// param: in a transition, after the keystroke's own event has finished.
function TransitionCaller({ initial = "" }: { initial?: string }) {
  const [value, setValue] = useState(initial)
  return (
    <>
      <SearchField
        label="Search"
        value={value}
        onChange={(next) => startTransition(() => setValue(next))}
      />
      <button type="button" onClick={() => setValue("")}>
        Reset elsewhere
      </button>
      <output aria-label="Committed">{value}</output>
    </>
  )
}

// A caller that commits what it was handed only when told to: the oldest
// value alone, the newest, or none of it, as when a newer navigation wins.
// Pressed with `fireEvent`, which leaves focus and the caret in the field.
function QueuedCaller() {
  const [value, setValue] = useState("")
  const [queue, setQueue] = useState<readonly string[]>([])
  const commitOldest = () => {
    const [oldest, ...rest] = queue
    if (oldest === undefined) return
    setValue(oldest)
    setQueue(rest)
  }
  return (
    <>
      <SearchField
        label="Search"
        value={value}
        onChange={(next) => setQueue((current) => [...current, next])}
      />
      <button type="button" onClick={commitOldest}>
        Commit oldest
      </button>
      <button
        type="button"
        onClick={() => {
          setValue(queue.at(-1) ?? value)
          setQueue([])
        }}
      >
        Commit all
      </button>
      <button type="button" onClick={() => setQueue([])}>
        Drop all
      </button>
      <output aria-label="Committed">{value}</output>
    </>
  )
}

// A caller that keeps only the first five characters.
function TruncatingCaller() {
  const [value, setValue] = useState("")
  return (
    <SearchField
      label="Search"
      value={value}
      onChange={(next) => setValue(next.slice(0, 5))}
    />
  )
}

describe("SearchField", () => {
  it("keeps the caret where it was when the caller commits later", async () => {
    render(<TransitionCaller />)
    const user = userEvent.setup()
    const field = screen.getByRole("searchbox", { name: "Search" })

    await user.type(field, "deepseek")
    await user.keyboard("{ArrowLeft>4/}XY")

    expect(field).toHaveValue("deepXYseek")
    expect(screen.getByLabelText("Committed")).toHaveTextContent("deepXYseek")
  })

  it("empties when the caller clears the value from elsewhere", async () => {
    render(<TransitionCaller initial="deepseek" />)
    const user = userEvent.setup()
    const field = screen.getByRole("searchbox", { name: "Search" })
    expect(field).toHaveValue("deepseek")

    await user.click(screen.getByRole("button", { name: "Reset elsewhere" }))

    expect(field).toHaveValue("")
  })

  it("empties from its own clear button and reports the empty value", async () => {
    render(<TransitionCaller initial="deepseek" />)
    const user = userEvent.setup()

    await user.click(screen.getByRole("button", { name: "Clear search" }))

    expect(screen.getByRole("searchbox", { name: "Search" })).toHaveValue("")
    await waitFor(() =>
      expect(screen.getByLabelText("Committed")).toBeEmptyDOMElement(),
    )
  })

  describe("when the caller has not caught up", () => {
    beforeEach(() => {
      vi.useFakeTimers({ shouldAdvanceTime: true })
    })

    afterEach(() => {
      vi.useRealTimers()
    })

    const settle = () =>
      act(() => {
        vi.advanceTimersByTime(SEARCH_SETTLE_MS)
      })

    it("keeps typing when the caller commits an older keystroke", async () => {
      render(<QueuedCaller />)
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      const field = screen.getByRole("searchbox", { name: "Search" })
      await user.type(field, "deepseek")
      fireEvent.click(screen.getByRole("button", { name: "Commit all" }))

      await user.keyboard("{ArrowLeft>4/}XY")
      fireEvent.click(screen.getByRole("button", { name: "Commit oldest" }))
      expect(screen.getByLabelText("Committed")).toHaveTextContent("deepXseek")
      await user.keyboard("Z")

      expect(field).toHaveValue("deepXYZseek")
      fireEvent.click(screen.getByRole("button", { name: "Commit all" }))
      settle()
      expect(field).toHaveValue("deepXYZseek")
    })

    it("waits out a slow commit rather than undoing it", async () => {
      render(<QueuedCaller />)
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      const field = screen.getByRole("searchbox", { name: "Search" })
      await user.type(field, "codex")

      act(() => {
        vi.advanceTimersByTime(SEARCH_SETTLE_MS / 2)
      })
      expect(field).toHaveValue("codex")
      fireEvent.click(screen.getByRole("button", { name: "Commit all" }))
      settle()

      expect(field).toHaveValue("codex")
    })

    it("shows the caller's value once typing stops if it dropped what was typed", async () => {
      render(<QueuedCaller />)
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      const field = screen.getByRole("searchbox", { name: "Search" })
      await user.type(field, "codex")

      fireEvent.click(screen.getByRole("button", { name: "Drop all" }))
      settle()

      expect(field).toHaveValue("")
    })

    it("shows what the caller kept once typing stops", async () => {
      render(<TruncatingCaller />)
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      const field = screen.getByRole("searchbox", { name: "Search" })

      await user.type(field, "deepseek")
      settle()

      expect(field).toHaveValue("deeps")
    })
  })
})
