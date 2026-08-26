import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  CopyableValue,
  CopyButton,
  EmptyState,
  FilterMultiComboBox,
  PageLoading,
  RefreshButton,
  StatCard,
} from "@/shared/components/ui"

describe("StatCard", () => {
  it("renders its label and value", () => {
    render(<StatCard label="Tracked cost" value="$12.34" />)
    expect(screen.getByText("Tracked cost")).toBeInTheDocument()
    expect(screen.getByText("$12.34")).toBeInTheDocument()
  })

  it("fits its grid track and avoids double padding", () => {
    // min-w-0 lets the tile shrink to its grid track (a fixed min-width overflowed
    // and overlapped the neighbour at two-up on mobile); p-0 zeroes HeroUI's own
    // card padding so it does not stack with Card.Content's and double the height.
    const { container } = render(<StatCard label="Requests" value="0" />)
    // Assert on the rendered root element rather than HeroUI's internal ".card"
    // class, so a library-internal class rename can't silently break this.
    const root = container.firstElementChild!
    expect(root.className).toContain("min-w-0")
    expect(root.className).toContain("p-0")
  })

  it("puts the trend under the value, on one row with the hint", () => {
    render(
      <StatCard
        label="Tracked cost"
        value="$12.34"
        trend={<span>up 4.2%</span>}
        hint="3 unpriced"
      />,
    )
    const value = screen.getByText("$12.34")
    const trend = screen.getByText("up 4.2%")
    const hint = screen.getByText("3 unpriced")
    // Below the value, not sharing its row.
    expect(value.parentElement).not.toContainElement(trend)
    expect(
      value.compareDocumentPosition(trend) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
    // On one row with the hint, and ahead of it, rather than stacked above it.
    expect(trend.parentElement).toBe(hint.parentElement)
    expect(
      trend.compareDocumentPosition(hint) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy()
  })

  it("reserves the aside row's height for a charted tile that has nothing to say", () => {
    // Otherwise a tile whose only aside was its delta loses the row, and its
    // sparkline rides above its neighbors' in the same grid row. Asserted on
    // the class for the same reason as the tile's own min-w-0 above: jsdom does
    // no layout, so the reservation is only observable as the utility.
    const { container } = render(
      <StatCard
        label="Tokens"
        value="3.3M"
        chart={<svg aria-label="trend" />}
      />,
    )
    expect(container.querySelector(".min-h-10\\.5")).not.toBeNull()
  })

  it("reserves nothing for a tile with neither aside nor chart", () => {
    const { container } = render(
      <StatCard label="Avg latency" value="1.33 s" />,
    )
    expect(container.querySelector(".min-h-10\\.5")).toBeNull()
  })
})

describe("RefreshButton", () => {
  it("fires onRefresh and shows a freshness label", async () => {
    const user = userEvent.setup()
    const onRefresh = vi.fn()
    render(
      <RefreshButton onRefresh={onRefresh} updatedAt={Date.now() - 5_000} />,
    )
    expect(screen.getByText(/Updated/)).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Refresh" }))
    expect(onRefresh).toHaveBeenCalledOnce()
  })

  it("hides the timestamp before the first load and disables while fetching", () => {
    const onRefresh = vi.fn()
    render(<RefreshButton onRefresh={onRefresh} isFetching updatedAt={0} />)
    expect(screen.queryByText(/Updated/)).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Refresh" })).toBeDisabled()
  })
})

describe("CopyButton", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("writes the value to the clipboard and confirms over the icon", async () => {
    const user = userEvent.setup()
    render(<CopyButton value="anthropic:claude-opus-4" label="model id" />)

    // Nothing is shown until a copy happens: this reports an event, so it must
    // not open on hover the way a hint tooltip would.
    await user.hover(screen.getByRole("button", { name: "Copy model id" }))
    expect(screen.queryByText("Copied!")).not.toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Copy model id" }))

    expect(await navigator.clipboard.readText()).toBe("anthropic:claude-opus-4")
    expect(await screen.findByText("Copied!")).toBeInTheDocument()
  })

  it("says the copy was blocked rather than claiming one when no path works", async () => {
    // The Clipboard API refuses and jsdom has no document.execCommand, so both
    // paths in copyToClipboard are exhausted.
    const user = userEvent.setup()
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(
      new Error("not a secure context"),
    )
    render(<CopyButton value="openai:gpt-4o" label="model id" />)

    await user.click(screen.getByRole("button", { name: "Copy model id" }))

    expect(await screen.findByText(/Copy blocked/)).toBeInTheDocument()
    expect(screen.queryByText("Copied!")).not.toBeInTheDocument()
  })

  it("keeps the confirmation out of the cell, so it cannot reflow the row", async () => {
    const user = userEvent.setup()
    const { container } = render(
      <CopyButton value="openai:gpt-4o" label="model id" />,
    )
    expect(container.textContent).toBe("")

    await user.click(screen.getByRole("button", { name: "Copy model id" }))
    await screen.findByText("Copied!")

    // The confirmation is an overlay, not a sibling of the id it copied.
    expect(container.textContent).toBe("")
  })

  it("clears the confirmation on its own", async () => {
    const user = userEvent.setup()
    render(<CopyButton value="openai:gpt-4o" label="model id" />)

    await user.click(screen.getByRole("button", { name: "Copy model id" }))
    expect(await screen.findByText("Copied!")).toBeInTheDocument()

    // Real timers: the clipboard write is a promise, and driving userEvent with
    // fake ones deadlocks against it. 1.5s dismissal, so 3s is a safe ceiling.
    await waitFor(
      () => expect(screen.queryByText("Copied!")).not.toBeInTheDocument(),
      { timeout: 3_000 },
    )
  })
})

describe("CopyableValue", () => {
  it("copies the value, which need not be what is displayed", async () => {
    const user = userEvent.setup()
    render(
      <CopyableValue value="openai:gpt-4o-2024-11-20" label="model id">
        gpt-4o-2024-11-20
      </CopyableValue>,
    )

    expect(screen.getByText("gpt-4o-2024-11-20")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Copy model id" }))
    expect(await navigator.clipboard.readText()).toBe(
      "openai:gpt-4o-2024-11-20",
    )
  })

  it("keeps a row press from starting on the value, so a drag can highlight it", () => {
    // The whole reason highlighting an id in a table used to fail: react-aria's
    // row press toggles selection on pointer down, and that re-render lands
    // mid-drag and discards the browser's nascent selection (#478). The value
    // stops the pointer sequence from reaching the row.
    const onRowPointerDown = vi.fn()
    const onRowMouseDown = vi.fn()
    render(
      // biome-ignore lint/a11y/noStaticElementInteractions: a stand-in for the table row whose handlers this test proves are not reached
      <div onPointerDown={onRowPointerDown} onMouseDown={onRowMouseDown}>
        <CopyableValue
          value="anthropic:claude-opus-4-5-20251101"
          label="model id"
        />
      </div>,
    )

    const value = screen.getByText("anthropic:claude-opus-4-5-20251101")
    fireEvent.pointerDown(value, {
      pointerId: 1,
      pointerType: "mouse",
      button: 0,
    })
    fireEvent.mouseDown(value, { button: 0 })

    expect(onRowPointerDown).not.toHaveBeenCalled()
    expect(onRowMouseDown).not.toHaveBeenCalled()
    // Selectable in its own right, so an inherited `user-select: none` from a
    // press elsewhere in the row cannot suppress it.
    expect(value.className).toContain("select-text")
  })
})

describe("EmptyState", () => {
  it("renders the title as a heading with its description", () => {
    render(
      <EmptyState
        title="No budgets yet"
        description="Create one to cap spending."
      />,
    )
    expect(
      screen.getByRole("heading", { name: "No budgets yet" }),
    ).toBeInTheDocument()
    expect(screen.getByText("Create one to cap spending.")).toBeInTheDocument()
  })

  it("fires onAction when the call to action is pressed", async () => {
    const user = userEvent.setup()
    const onAction = vi.fn()
    render(
      <EmptyState
        title="No API keys yet"
        actionLabel="Create your first key"
        onAction={onAction}
      />,
    )
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    expect(onAction).toHaveBeenCalledOnce()
  })

  it("omits the action entirely for a purely informational empty state", () => {
    render(
      <EmptyState
        title="No usage yet"
        description="Spend appears here once traffic flows."
      />,
    )
    expect(screen.queryByRole("button")).not.toBeInTheDocument()
  })

  it("disables the call to action and suppresses onAction when isActionDisabled is set", async () => {
    const user = userEvent.setup()
    const onAction = vi.fn()
    render(
      <EmptyState
        title="Welcome"
        actionLabel="Add your first provider"
        onAction={onAction}
        isActionDisabled
      />,
    )
    const button = screen.getByRole("button", {
      name: "Add your first provider",
    })
    expect(button).toBeDisabled()
    await user.click(button)
    expect(onAction).not.toHaveBeenCalled()
  })
})

describe("PageLoading", () => {
  it("exposes a status role so the wait is announced", () => {
    render(<PageLoading />)
    const status = screen.getByRole("status")
    expect(status).toHaveTextContent("Loading…")
  })
})

describe("FilterMultiComboBox", () => {
  function Harness({
    onChange,
    maxValues,
    allowsCustom,
  }: {
    onChange?: (values: string[]) => void
    maxValues?: number
    allowsCustom?: boolean
  }) {
    const [values, setValues] = useState<string[]>([])
    return (
      <FilterMultiComboBox
        label="Model"
        values={values}
        onChange={(next) => {
          setValues(next)
          onChange?.(next)
        }}
        options={[
          { value: "gpt-5.6", label: "gpt-5.6" },
          { value: "claude-sonnet-5", label: "claude-sonnet-5" },
        ]}
        placeholder="All models"
        maxValues={maxValues}
        allowsCustom={allowsCustom}
      />
    )
  }

  it("accumulates picks and drops the picked options from the list", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.click(await screen.findByRole("option", { name: "gpt-5.6" }))
    expect(onChange).toHaveBeenLastCalledWith(["gpt-5.6"])

    // The list stays open on what is left, so a comparison is one gesture; a
    // picked value is not offered again (picking it twice would be a no-op).
    expect(
      screen.queryByRole("option", { name: "gpt-5.6" }),
    ).not.toBeInTheDocument()
    await user.click(
      await screen.findByRole("option", { name: "claude-sonnet-5" }),
    )
    expect(onChange).toHaveBeenLastCalledWith(["gpt-5.6", "claude-sonnet-5"])

    // The input reports the size of the selection: the values themselves are the
    // page's chips, not text crammed into the box.
    expect(input).toHaveAttribute("placeholder", "2 selected")
  })

  it("narrows the list as you type without committing the text", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.type(input, "claude")
    expect(
      screen.queryByRole("option", { name: "gpt-5.6" }),
    ).not.toBeInTheDocument()
    expect(
      await screen.findByRole("option", { name: "claude-sonnet-5" }),
    ).toBeInTheDocument()
    // Typing alone filters nothing: only a picked option becomes a filter value.
    expect(onChange).not.toHaveBeenCalled()
  })

  it("commits typed text on Enter when custom values are allowed", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} allowsCustom />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.type(input, "unlisted-model")
    await user.keyboard("{Enter}")

    expect(onChange).toHaveBeenLastCalledWith(["unlisted-model"])
  })

  it("does not also commit the query when Enter picks a highlighted option", async () => {
    // One press must add one value. Arrowing onto an option and pressing Enter
    // selects it; committing the partial text beside it would add "claude" as well.
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} allowsCustom />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.type(input, "claude")
    await user.keyboard("{ArrowDown}")
    await user.keyboard("{Enter}")

    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange).toHaveBeenLastCalledWith(["claude-sonnet-5"])
  })

  it("ignores Enter on typed text when custom values are not allowed", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.type(input, "not-an-option")
    await user.keyboard("{Enter}")

    expect(onChange).not.toHaveBeenCalled()
  })

  it("stops at the value ceiling the endpoints accept", async () => {
    const user = userEvent.setup()
    const onChange = vi.fn()
    render(<Harness onChange={onChange} maxValues={1} />)

    const input = screen.getByRole("combobox", { name: "Model" })
    await user.click(input)
    await user.click(await screen.findByRole("option", { name: "gpt-5.6" }))
    expect(onChange).toHaveBeenLastCalledWith(["gpt-5.6"])

    // Past the ceiling the remaining options are inert rather than a pick that
    // 422s every query on the page, and the input says the set is full.
    expect(input).toHaveAttribute("placeholder", "1 selected (max)")
    const remaining = await screen.findByRole("option", {
      name: "claude-sonnet-5",
    })
    expect(remaining).toHaveAttribute("aria-disabled", "true")
    await user.click(remaining)
    expect(onChange).toHaveBeenCalledTimes(1)
  })
})
