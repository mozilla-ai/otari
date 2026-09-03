import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { Segmented, Tab, TabRow } from "./surface"

/**
 * The two segmented shapes and the arguments their docstrings make.
 *
 * Both are documented as deliberate departures from the obvious ARIA pattern,
 * and neither departure was verified by anything: `Tab` is a plain `<button>`
 * with `aria-pressed` rather than a tablist, because these switch what a panel
 * shows without implementing a tab widget's roving-focus contract; `TabRow` is
 * a bare `<div>` with no role, because `role="group"` wants a `<fieldset>` and
 * `role="tablist"` would promise that contract; and `Segmented` uses native
 * radios precisely so the arrow-key and single-tab-stop behavior come for free
 * rather than being reimplemented.
 *
 * An argument in a docstring that nothing verifies is a decision that can be
 * reversed by accident, which is what these pin.
 *
 * One limit stated up front: jsdom implements neither radio arrow-key
 * navigation nor a browser's tab-stop grouping, so the tests below assert the
 * STRUCTURE that earns those behaviors (real inputs, one shared name) rather
 * than the behaviors themselves. Claiming to have tested arrow keys here would
 * be claiming the opposite of what ran.
 */
describe("Tab", () => {
  it("is a real button, not a div wearing a role", () => {
    // Everything else here follows from this: the keyboard activation, the
    // disabled semantics and the focus behavior are the element's, not ours.
    render(
      <Tab isActive={false} onPress={() => undefined}>
        Requests
      </Tab>,
    )
    const tab = screen.getByRole("button", { name: "Requests" })
    expect(tab.tagName).toBe("BUTTON")
    expect(tab).toHaveAttribute("type", "button")
  })

  it("exposes the choice as a pressed state", () => {
    const { rerender } = render(
      <Tab isActive={false} onPress={() => undefined}>
        Requests
      </Tab>,
    )
    expect(screen.getByRole("button", { name: "Requests" })).toHaveAttribute(
      "aria-pressed",
      "false",
    )
    rerender(
      <Tab isActive onPress={() => undefined}>
        Requests
      </Tab>,
    )
    // Present and `true`, not simply absent when inactive: a toggle that drops
    // the attribute reads as an ordinary button rather than as an unchosen one.
    expect(screen.getByRole("button", { name: "Requests" })).toHaveAttribute(
      "aria-pressed",
      "true",
    )
  })

  it("is deliberately not a tab widget", () => {
    // The docstring's actual argument: these do not implement roving focus, so
    // promising a tablist would describe a contract that is not kept.
    render(
      <Tab isActive onPress={() => undefined}>
        Requests
      </Tab>,
    )
    expect(screen.queryByRole("tab")).toBeNull()
    expect(screen.queryByRole("tablist")).toBeNull()
  })

  it("activates from the keyboard without any handler of its own", async () => {
    // Enter and Space both, because that is what a native button gives and it
    // is the entire reason for not hand-rolling this control.
    const user = userEvent.setup()
    const onPress = vi.fn()
    render(
      <Tab isActive={false} onPress={onPress}>
        Requests
      </Tab>,
    )
    await user.tab()
    expect(screen.getByRole("button", { name: "Requests" })).toHaveFocus()
    await user.keyboard("{Enter}")
    await user.keyboard(" ")
    expect(onPress).toHaveBeenCalledTimes(2)
  })

  it("marks the active one with a surface step and never with the accent", () => {
    // The accent is data ink and fills. Using it to say "this one" would put
    // the chart's own color on a navigation control.
    const { rerender } = render(
      <Tab isActive onPress={() => undefined}>
        Requests
      </Tab>,
    )
    const active = screen.getByRole("button", { name: "Requests" })
    // Token checks, not substring: `hover:text-foreground` contains
    // "text-foreground", so a substring assertion passes for an active tab that
    // only reaches its ink on hover. Same trap as the divider below.
    expect([...active.classList]).toContain("bg-surface-subtle")
    expect([...active.classList]).toContain("text-foreground")
    expect(active.className).not.toContain("accent")

    rerender(
      <Tab isActive={false} onPress={() => undefined}>
        Requests
      </Tab>,
    )
    const inactive = screen.getByRole("button", { name: "Requests" })
    // Bare muted text: an inactive segment has no fill at all, so the row reads
    // as one chosen thing rather than as a strip of buttons.
    expect([...inactive.classList]).toContain("text-muted")
    expect(inactive.className).not.toContain("bg-")
  })

  it("neither shrinks nor wraps", () => {
    // Both halves matter and for different reasons: squeezing would put the
    // same control at two widths on one page, wrapping would break the row's
    // height. The row scrolls instead of either.
    render(
      <Tab isActive={false} onPress={() => undefined}>
        A very long segment label
      </Tab>,
    )
    const tab = screen.getByRole("button")
    expect([...tab.classList]).toContain("shrink-0")
    expect([...tab.classList]).toContain("whitespace-nowrap")
  })
})

describe("TabRow", () => {
  it("carries no role of its own", () => {
    // The docstring's argument, pinned: `role="group"` needs a `<fieldset>` to
    // be valid and drags form semantics and a `<legend>` in with it, and
    // `role="tablist"` would promise roving focus that `Tab` does not
    // implement. The row is spacing; the tabs carry the semantics.
    const { container } = render(
      <TabRow>
        <Tab isActive onPress={() => undefined}>
          Requests
        </Tab>
      </TabRow>,
    )
    const row = container.firstElementChild as HTMLElement
    expect(row.tagName).toBe("DIV")
    expect(row).not.toHaveAttribute("role")
    expect(screen.queryByRole("tablist")).toBeNull()
    expect(screen.queryByRole("group")).toBeNull()
  })

  it("leaves its tabs individually addressable", () => {
    // What the row not having a role costs, and does not cost: each tab still
    // has its own name and its own pressed state, which is what a screen reader
    // actually needs here.
    render(
      <TabRow>
        <Tab isActive onPress={() => undefined}>
          Requests
        </Tab>
        <Tab isActive={false} onPress={() => undefined}>
          Errors
        </Tab>
      </TabRow>,
    )
    expect(screen.getByRole("button", { name: "Requests" })).toHaveAttribute(
      "aria-pressed",
      "true",
    )
    expect(screen.getByRole("button", { name: "Errors" })).toHaveAttribute(
      "aria-pressed",
      "false",
    )
  })

  it("scrolls rather than reflowing when the tabs overflow", () => {
    const { container } = render(
      <TabRow>
        <Tab isActive onPress={() => undefined}>
          Requests
        </Tab>
      </TabRow>,
    )
    const row = container.firstElementChild as HTMLElement
    expect([...row.classList]).toContain("overflow-x-auto")
    expect([...row.classList]).toContain("max-w-full")
  })
})

describe("Segmented", () => {
  const OPTIONS = [
    { value: "day", label: "Day" },
    { value: "week", label: "Week" },
    { value: "month", label: "Month" },
  ]

  const setup = (value = "week") => {
    const onChange = vi.fn()
    const view = render(
      <Segmented
        label="Bucket"
        options={OPTIONS}
        value={value}
        onChange={onChange}
      />,
    )
    return { onChange, view, user: userEvent.setup() }
  }

  it("is a named radio group of real radios", () => {
    // Native inputs and not `role="radio"` on buttons: the semantics here are
    // exactly a radio group's, so taking the real element brings the arrow-key
    // navigation and the one-tab-stop-per-group behavior with it rather than
    // reimplementing them. Those two are browser behaviors that jsdom does not
    // provide, so what is asserted is the structure that earns them.
    setup()
    const group = screen.getByRole("radiogroup", { name: "Bucket" })
    expect(group).toBeInTheDocument()
    const radios = screen.getAllByRole("radio")
    expect(radios).toHaveLength(3)
    for (const radio of radios) {
      expect(radio.tagName).toBe("INPUT")
      expect(radio).toHaveAttribute("type", "radio")
    }
  })

  it("puts every option on one shared name, which is what makes it a group", () => {
    // A radio group is defined by the shared `name`. Get this wrong and each
    // segment is its own single-option group: arrow keys stop working, every
    // segment takes a tab stop, and nothing looks different on screen.
    setup()
    const names = new Set(
      screen.getAllByRole("radio").map((radio) => radio.getAttribute("name")),
    )
    expect(names.size).toBe(1)
    expect([...names][0]).toBeTruthy()
  })

  it("gives two groups on one page different names", () => {
    // The name comes from `useId` rather than the label, so two of these in one
    // form do not silently merge into a single group.
    render(
      <>
        <Segmented
          label="Bucket"
          options={OPTIONS}
          value="day"
          onChange={() => undefined}
        />
        <Segmented
          label="Window"
          options={OPTIONS}
          value="day"
          onChange={() => undefined}
        />
      </>,
    )
    const first = screen
      .getByRole("radiogroup", { name: "Bucket" })
      .querySelector("input")
      ?.getAttribute("name")
    const second = screen
      .getByRole("radiogroup", { name: "Window" })
      .querySelector("input")
      ?.getAttribute("name")
    expect(first).toBeTruthy()
    expect(first).not.toBe(second)
  })

  it("checks exactly the current value", () => {
    setup("week")
    expect(screen.getByRole("radio", { name: "Week" })).toBeChecked()
    expect(screen.getByRole("radio", { name: "Day" })).not.toBeChecked()
    expect(screen.getByRole("radio", { name: "Month" })).not.toBeChecked()
  })

  it("reports the option that was chosen", async () => {
    const { onChange, user } = setup("week")
    await user.click(screen.getByRole("radio", { name: "Month" }))
    expect(onChange).toHaveBeenCalledTimes(1)
    expect(onChange).toHaveBeenCalledWith("month")
  })

  it("is operable by its visible label, not only by the hidden input", () => {
    // Each input is `sr-only` and the segment IS the label, so the label being
    // wired to its own input is the only thing making the control clickable.
    setup()
    const input = screen.getByRole("radio", { name: "Week" })
    expect([...input.classList]).toContain("sr-only")
    expect(input.closest("label")).not.toBeNull()
  })

  it("marks the selected segment with a surface step rather than a fill", () => {
    // A fill here would be the only filled thing in a form that has none, and
    // filling the chosen one the way a submit button is filled makes a recorded
    // value read as an action to take.
    setup("week")
    const selected = screen
      .getByRole("radio", { name: "Week" })
      .closest("label") as HTMLElement
    const unselected = screen
      .getByRole("radio", { name: "Day" })
      .closest("label") as HTMLElement
    expect([...selected.classList]).toContain("bg-surface-subtle")
    expect([...selected.classList]).toContain("text-foreground")
    expect(selected.className).not.toContain("accent")
    expect([...unselected.classList]).toContain("text-muted")
    expect(unselected.className).not.toContain("bg-")
  })

  it("draws one fewer rule than it has segments", () => {
    // A leading border on every segment but the first, so the count of rules
    // follows the count of segments however many there are, and the track never
    // opens or closes on a doubled edge.
    setup()
    const labels = screen
      .getAllByRole("radio")
      .map((radio) => radio.closest("label") as HTMLElement)
    for (const label of labels) {
      // `classList`, not `className.includes`: `first:border-l-0` contains the
      // substring "border-l", so a substring assertion here passes with the
      // dividing border deleted entirely. Found by breaking it.
      expect([...label.classList]).toContain("border-l")
      expect([...label.classList]).toContain("first:border-l-0")
    }
  })

  it("shows the focus ring on the label, since the input cannot show one", () => {
    // The input is visually hidden, so a focus ring drawn on it is invisible.
    // `has-[:focus-visible]` moves it to the segment the user can actually see.
    setup()
    const label = screen
      .getByRole("radio", { name: "Week" })
      .closest("label") as HTMLElement
    expect([...label.classList]).toContain(
      "has-[:focus-visible]:otari-focus-ring",
    )
  })

  it("neither shrinks nor wraps its segments", () => {
    setup()
    const label = screen
      .getByRole("radio", { name: "Week" })
      .closest("label") as HTMLElement
    expect([...label.classList]).toContain("shrink-0")
    expect([...label.classList]).toContain("whitespace-nowrap")
  })
})
