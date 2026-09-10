import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import { ListDetail, ListDetailRow } from "./ListDetail"

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

  it("makes the empty column a real button, reachable by keyboard", async () => {
    // The whole column is the target, which a div with an onClick would also
    // manage while being unreachable without a pointer.
    const onEmptyPress = vi.fn()
    render(
      <ListDetail
        listLabel="Policies"
        list={null}
        empty="Create your first policy"
        onEmptyPress={onEmptyPress}
        detail={null}
        isDetailShown={false}
        onShowList={() => {}}
      />,
    )

    await userEvent.tab()
    await userEvent.keyboard("{Enter}")
    expect(onEmptyPress).toHaveBeenCalledOnce()
    expect(
      screen.getByRole("button", { name: "Create your first policy" }),
    ).toBeInTheDocument()
  })

  it("states the empty column without offering it to a reader who cannot write", () => {
    render(
      <ListDetail
        listLabel="Policies"
        list={null}
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
