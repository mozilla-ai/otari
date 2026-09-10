import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { Button } from "../actions/Button"
import { ListDetail, ListDetailRow } from "./ListDetail"

function Frame({ isDetailShown }: { isDetailShown: boolean }) {
  return (
    <ListDetail
      listLabel="Policies"
      list={<ListDetailRow label="fast" isSelected onSelect={() => {}} />}
      detail={<p>detail</p>}
      isDetailShown={isDetailShown}
      onShowList={() => {}}
    />
  )
}

describe("ListDetail", () => {
  it("names both columns, so neither half is anonymous", () => {
    render(
      <ListDetail
        listLabel="Policies"
        list={null}
        detail={<p>Nothing open.</p>}
        detailLabel="Policy detail"
        isDetailShown={false}
        onShowList={() => {}}
      />,
    )

    expect(screen.getByRole("region", { name: "Policies" })).toContainElement(
      screen.getByRole("heading", { name: "Policies" }),
    )
    expect(
      screen.getByRole("region", { name: "Policy detail" }),
    ).toHaveTextContent("Nothing open.")
  })

  it("puts the create control in the column it adds to", () => {
    render(
      <ListDetail
        listLabel="Policies"
        listAction={<button type="button">Create policy</button>}
        list={null}
        detail={null}
        isDetailShown={false}
        onShowList={() => {}}
      />,
    )

    expect(screen.getByRole("region", { name: "Policies" })).toContainElement(
      screen.getByRole("button", { name: "Create policy" }),
    )
  })

  it("offers the way back only while a column is hidden", async () => {
    // The control is `md:hidden`, and jsdom applies no media query, so what is
    // asserted here is the other half: it exists while the detail column is
    // the one on screen and not while both are.
    const onShowList = vi.fn()
    const { rerender } = render(
      <ListDetail
        listLabel="Policies"
        list={null}
        detail={<p>fast</p>}
        isDetailShown
        onShowList={onShowList}
        backLabel="All policies"
      />,
    )

    await userEvent.click(screen.getByRole("button", { name: "All policies" }))
    expect(onShowList).toHaveBeenCalledOnce()

    rerender(
      <ListDetail
        listLabel="Policies"
        list={null}
        detail={<p>fast</p>}
        isDetailShown={false}
        onShowList={onShowList}
      />,
    )
    expect(
      screen.queryByRole("button", { name: /All policies|Back to the list/ }),
    ).not.toBeInTheDocument()
  })

  it("offers the empty column's action as a named control", async () => {
    // Named, and the size of a control. The column itself was the press target
    // once, which made a button as tall as the page whose accessible name was
    // whatever prose happened to be inside it.
    const onCreate = vi.fn()
    render(
      <ListDetail
        listLabel="Policies"
        list={null}
        isEmpty
        empty="Create your first policy"
        emptyAction={
          <Button variant="primary" onPress={onCreate}>
            Create policy
          </Button>
        }
        detail={null}
        isDetailShown={false}
        onShowList={() => {}}
      />,
    )

    expect(screen.getByText("Create your first policy")).toBeInTheDocument()
    await userEvent.click(screen.getByRole("button", { name: "Create policy" }))
    expect(onCreate).toHaveBeenCalledOnce()
  })

  it("shows the rows even when the empty message is a node it was handed", () => {
    // `empty` used to double as the flag, tested with `=== undefined`, so the
    // ordinary call-site idiom `empty={cond ? "None yet." : null}` blanked a
    // list that had rows in it.
    render(
      <ListDetail
        listLabel="Policies"
        list={<ListDetailRow label="fast" isSelected onSelect={() => {}} />}
        empty={null}
        detail={null}
        isDetailShown={false}
        onShowList={() => {}}
      />,
    )

    expect(screen.getByRole("button", { name: /fast/ })).toBeInTheDocument()
  })

  it("states the empty column without offering it to a reader who cannot write", () => {
    render(
      <ListDetail
        listLabel="Policies"
        list={null}
        isEmpty
        empty="No policies yet."
        detail={null}
        isDetailShown={false}
        onShowList={() => {}}
      />,
    )

    expect(screen.getByText("No policies yet.")).toBeInTheDocument()
    expect(screen.queryByRole("button")).not.toBeInTheDocument()
  })

  it("shows the list when nothing is empty about it", () => {
    render(
      <ListDetail
        listLabel="Policies"
        list={<p>fast</p>}
        detail={null}
        isDetailShown={false}
        onShowList={() => {}}
      />,
    )

    expect(screen.getByText("fast")).toBeInTheDocument()
  })
  it("moves focus into the column that appears when a record opens", async () => {
    // The blocking bug: below `md` the list column becomes `display: none`
    // while the pressed row still holds focus, which drops
    // `document.activeElement` to `<body>` at the moment a record opens.
    //
    // What this proves and what it does not: jsdom applies no CSS, so it cannot
    // show the column being hidden, and the original bug is invisible here for
    // that reason. It does prove the mechanism, that focus follows the swap,
    // which is the half this component owns. The width gating is CSS: the Back
    // control is `md:hidden`, so from `md` up the focus call lands on a hidden
    // element and does nothing.
    const { rerender } = render(<Frame isDetailShown={false} />)
    const row = screen.getByRole("button", { name: /fast/ })
    row.focus()
    expect(document.activeElement).toBe(row)

    rerender(<Frame isDetailShown />)
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: "Back to the list" }),
    )
  })

  it("returns focus to the open record when the list comes back", async () => {
    const { rerender } = render(<Frame isDetailShown />)
    rerender(<Frame isDetailShown={false} />)

    // The row that was open, found by `aria-current`, so focus returns to where
    // the reader was rather than to the top of the column.
    expect(document.activeElement).toBe(
      screen.getByRole("button", { name: /fast/ }),
    )
  })

  it("does not take focus on arrival", () => {
    // Only a change of `isDetailShown` moves focus. Stealing it on mount would
    // take it off whatever the operator was already doing on the page.
    render(<Frame isDetailShown={false} />)
    expect(document.activeElement).toBe(document.body)
  })
})

describe("ListDetailRow", () => {
  it("presses its label to open the record, not the whole row", async () => {
    const onSelect = vi.fn()
    render(
      <ListDetailRow label="fast" isSelected={false} onSelect={onSelect}>
        openai:gpt-5-mini
      </ListDetailRow>,
    )

    await userEvent.click(screen.getByRole("button", { name: /fast/ }))
    expect(onSelect).toHaveBeenCalledOnce()
    expect(screen.getByText("openai:gpt-5-mini")).toBeInTheDocument()
  })

  it("says which record is the one being shown", () => {
    render(
      <>
        <ListDetailRow label="fast" isSelected onSelect={() => {}} />
        <ListDetailRow label="cheap" isSelected={false} onSelect={() => {}} />
      </>,
    )

    expect(screen.getByRole("button", { name: "fast" })).toHaveAttribute(
      "aria-current",
      "true",
    )
    expect(screen.getByRole("button", { name: "cheap" })).not.toHaveAttribute(
      "aria-current",
    )
  })

  it("keeps a row's actions out of the press target that opens it", async () => {
    // Nested buttons are neither valid nor operable, so the label and the
    // actions are siblings. Pressing an action must not also open the record.
    const onSelect = vi.fn()
    const onDelete = vi.fn()
    render(
      <ListDetailRow
        label="cheap"
        isSelected={false}
        onSelect={onSelect}
        actions={
          <button type="button" onClick={onDelete}>
            Delete cheap
          </button>
        }
      />,
    )

    await userEvent.click(screen.getByRole("button", { name: "Delete cheap" }))
    expect(onDelete).toHaveBeenCalledOnce()
    expect(onSelect).not.toHaveBeenCalled()
  })
})
