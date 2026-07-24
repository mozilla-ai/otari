import { render, screen, within } from "@testing-library/react";
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

function base<T extends object>(props: T) {
  return { ariaLabel: "Test table", columns: COLUMNS, rows: ROWS, getRowKey: (r: Row) => r.id, ...props };
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
