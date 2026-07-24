import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import type { Selection, SortDescriptor } from "react-aria-components";
import { describe, expect, it, vi } from "vitest";

import { DataTable, type DataTableColumn } from "./DataTable";

interface Row {
  id: string;
  name: string;
  cost: number;
}

const ROWS: Row[] = [
  { id: "a", name: "Alpha", cost: 3 },
  { id: "b", name: "Bravo", cost: 1 },
  { id: "c", name: "Charlie", cost: 2 },
];

const COLUMNS: DataTableColumn<Row>[] = [
  { id: "name", header: "Name", cell: (r) => r.name, isRowHeader: true },
  { id: "cost", header: "Cost", cell: (r) => r.cost, align: "end", allowsSorting: true },
];

// Stable identity matters: DataTable caches rendered rows on getRowKey (and
// columns), so an inline arrow here would rebuild every row on each re-render.
const getRowKey = (r: Row) => r.id;

function base<T extends object>(props: T) {
  return { ariaLabel: "Test table", columns: COLUMNS, rows: ROWS, getRowKey, ...props };
}

describe("DataTable", () => {
  it("renders headers and cell content", () => {
    render(<DataTable {...base({})} />);
    expect(screen.getByRole("columnheader", { name: "Name" })).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /Alpha/ })).toBeInTheDocument();
    expect(screen.getByText("Charlie")).toBeInTheDocument();
  });

  it("shows a loading state with no rows", () => {
    render(<DataTable {...base({ rows: [], isLoading: true })} />);
    expect(screen.getByText("Loading…")).toBeInTheDocument();
  });

  it("shows the empty state when there are no rows and not loading", () => {
    render(<DataTable {...base({ rows: [], emptyContent: "Nothing here" })} />);
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
  });

  it("fires onRowAction with the row key when a row is activated", async () => {
    const user = userEvent.setup();
    const onRowAction = vi.fn();
    render(<DataTable {...base({ onRowAction })} />);
    await user.click(screen.getByRole("row", { name: /Bravo/ }));
    expect(onRowAction).toHaveBeenCalledWith("b");
  });

  it("selects rows via checkboxes and reports the selected keys", async () => {
    const user = userEvent.setup();

    function Harness() {
      const [selected, setSelected] = useState<Selection>(new Set());
      return (
        <>
          <span data-testid="count">{selected === "all" ? "all" : selected.size}</span>
          <DataTable {...base({ selectionMode: "multiple", selectedKeys: selected, onSelectionChange: setSelected })} />
        </>
      );
    }

    render(<Harness />);
    const alphaRow = screen.getByRole("row", { name: /Alpha/ });
    await user.click(within(alphaRow).getByRole("checkbox"));
    expect(screen.getByTestId("count")).toHaveTextContent("1");
  });

  it("does not select a disabled row", async () => {
    const user = userEvent.setup();

    function Harness() {
      const [selected, setSelected] = useState<Selection>(new Set());
      return (
        <>
          <span data-testid="count">{selected === "all" ? "all" : selected.size}</span>
          <DataTable
            {...base({
              selectionMode: "multiple",
              selectedKeys: selected,
              onSelectionChange: setSelected,
              disabledKeys: ["b"],
            })}
          />
        </>
      );
    }

    render(<Harness />);
    const bravoRow = screen.getByRole("row", { name: /Bravo/ });
    await user.click(within(bravoRow).getByRole("checkbox"));
    expect(screen.getByTestId("count")).toHaveTextContent("0");
  });

  it("does not re-render row cells when the selection changes", async () => {
    // Selection toggles re-render the controlling parent, and with hundreds of
    // rows per page a full row rebuild made each checkbox click lag by whole
    // seconds. Rendered rows are cached per row object (react-aria's items
    // collection), so a selection change must not re-invoke cell renderers.
    const user = userEvent.setup();
    const cellSpy = vi.fn((r: Row) => r.name);
    const spiedColumns: DataTableColumn<Row>[] = [
      { id: "name", header: "Name", cell: cellSpy, isRowHeader: true },
    ];

    function Harness() {
      const [selected, setSelected] = useState<Selection>(new Set());
      return (
        <DataTable
          {...base({
            columns: spiedColumns,
            selectionMode: "multiple",
            selectedKeys: selected,
            onSelectionChange: setSelected,
          })}
        />
      );
    }

    render(<Harness />);
    const callsAfterMount = cellSpy.mock.calls.length;

    const alphaRow = screen.getByRole("row", { name: /Alpha/ });
    await user.click(within(alphaRow).getByRole("checkbox"));
    expect(alphaRow).toHaveAttribute("aria-selected", "true");
    expect(cellSpy.mock.calls.length).toBe(callsAfterMount);
  });

  it("re-invokes cell renderers on selection when columns are unstable (the contract's failure mode)", async () => {
    // Companion to the caching test above: a caller that rebuilds `columns`
    // each render defeats the row cache, and every selection click re-renders
    // the whole page of rows again. This documents the cost of violating the
    // stability contract so it stays visible if a page regresses to inline
    // columns; the fix is memoization at the call site, not a DataTable change.
    const user = userEvent.setup();
    const cellSpy = vi.fn((r: Row) => r.name);

    function Harness() {
      const [selected, setSelected] = useState<Selection>(new Set());
      const unstableColumns: DataTableColumn<Row>[] = [
        { id: "name", header: "Name", cell: cellSpy, isRowHeader: true },
      ];
      return (
        <DataTable
          {...base({
            columns: unstableColumns,
            selectionMode: "multiple",
            selectedKeys: selected,
            onSelectionChange: setSelected,
          })}
        />
      );
    }

    render(<Harness />);
    const callsAfterMount = cellSpy.mock.calls.length;

    const alphaRow = screen.getByRole("row", { name: /Alpha/ });
    await user.click(within(alphaRow).getByRole("checkbox"));
    expect(alphaRow).toHaveAttribute("aria-selected", "true");
    expect(cellSpy.mock.calls.length).toBeGreaterThan(callsAfterMount);
  });

  it("renders the detail panel as the row immediately under the matching row", async () => {
    // Regression: after the shared-table migration the detail card rendered
    // below the whole table, so on a long page a row click looked like it did
    // nothing. The panel must open adjacent to the clicked row.
    const user = userEvent.setup();
    const onRowAction = vi.fn();
    const renderDetail = (r: Row) => <div>{`detail for ${r.name}`}</div>;
    render(<DataTable {...base({ onRowAction, detailKey: "b", renderDetail })} />);

    // The detail row is a portal-managed <tr> outside react-aria's collection
    // (no grid row role) and is inserted after react-aria commits its rows, so
    // adjacency is asserted on the DOM sibling once it lands.
    await waitFor(() => {
      const bravoRow = screen.getByRole("row", { name: /Bravo/ });
      expect(bravoRow.nextElementSibling?.textContent).toContain("detail for Bravo");
    });

    // Activating the detail row itself must not fire onRowAction (which would
    // re-toggle the panel in callers).
    await user.click(screen.getByText("detail for Bravo"));
    expect(onRowAction).not.toHaveBeenCalled();
  });

  it("keeps the detail panel attached through row reorders, removal, and return", async () => {
    // The host <tr> lives outside react-aria's collection, so react's row
    // reconciliation (sorting, filtering, pagination) must not strand or
    // duplicate it: the insert effect re-runs on row changes and its cleanup
    // removes the host when the target row disappears.
    const renderDetail = (r: Row) => <div>{`detail for ${r.name}`}</div>;
    const detailFor = (name: string) =>
      screen.getByRole("row", { name: new RegExp(name) }).nextElementSibling?.textContent ?? null;

    const { rerender } = render(<DataTable {...base({ detailKey: "b", renderDetail })} />);
    await waitFor(() => expect(detailFor("Bravo")).toContain("detail for Bravo"));

    // Reorder (a sort flipping Bravo to the top): panel follows its row.
    const reordered = [ROWS[1], ROWS[2], ROWS[0]];
    rerender(<DataTable {...base({ rows: reordered, detailKey: "b", renderDetail })} />);
    await waitFor(() => expect(detailFor("Bravo")).toContain("detail for Bravo"));
    expect(document.querySelectorAll(".otari-detail-row")).toHaveLength(1);

    // Target filtered out: panel is removed, nothing stranded.
    rerender(<DataTable {...base({ rows: [ROWS[0], ROWS[2]], detailKey: "b", renderDetail })} />);
    await waitFor(() => expect(document.querySelectorAll(".otari-detail-row")).toHaveLength(0));
    expect(screen.queryByText("detail for Bravo")).not.toBeInTheDocument();

    // Target returns (filter cleared): panel re-attaches with correct content.
    rerender(<DataTable {...base({ detailKey: "b", renderDetail })} />);
    await waitFor(() => expect(detailFor("Bravo")).toContain("detail for Bravo"));
    expect(document.querySelectorAll(".otari-detail-row")).toHaveLength(1);
  });

  it("still fires onRowAction on a row click while a selection is active", async () => {
    // react-aria's toggle behavior repurposes row clicks into selection
    // extension once any row is selected; DataTable intercepts those clicks so
    // the drill-in keeps working (checkboxes own selection).
    const user = userEvent.setup();
    const onRowAction = vi.fn();

    function Harness() {
      const [selected, setSelected] = useState<Selection>(new Set(["a"]));
      return (
        <>
          <span data-testid="count">{selected === "all" ? "all" : selected.size}</span>
          <DataTable
            {...base({
              onRowAction,
              selectionMode: "multiple",
              selectedKeys: selected,
              onSelectionChange: setSelected,
            })}
          />
        </>
      );
    }

    render(<Harness />);
    await user.click(within(screen.getByRole("row", { name: /Bravo/ })).getByText("Bravo"));
    expect(onRowAction).toHaveBeenCalledWith("b");
    expect(screen.getByTestId("count")).toHaveTextContent("1");
  });

  it("reports sort changes from a sortable column header", async () => {
    const user = userEvent.setup();

    function Harness() {
      const [sort, setSort] = useState<SortDescriptor | undefined>(undefined);
      return (
        <>
          <span data-testid="sort">{sort ? `${String(sort.column)}:${sort.direction}` : "none"}</span>
          <DataTable {...base({ sortDescriptor: sort, onSortChange: setSort })} />
        </>
      );
    }

    render(<Harness />);
    await user.click(screen.getByRole("columnheader", { name: "Cost" }));
    expect(screen.getByTestId("sort")).toHaveTextContent("cost:ascending");
  });
});
