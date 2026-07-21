import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Table, Td, Th, THead, Tr } from "@/components/Table";

// A two-column table with a body row, the minimal shape every page renders.
function renderTable() {
  return render(
    <Table>
      <THead>
        <Tr>
          <Th>Alpha</Th>
          <Th>Beta</Th>
        </Tr>
      </THead>
      <tbody>
        <Tr>
          <Td>1</Td>
          <Td>2</Td>
        </Tr>
      </tbody>
    </Table>,
  );
}

describe("Table resizable columns", () => {
  beforeEach(() => {
    // jsdom has no layout, so header cells report zero width. Pin a width so a
    // resize snapshots (and then nudges) a deterministic starting point.
    vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue({
      width: 120,
      height: 0,
      top: 0,
      left: 0,
      right: 120,
      bottom: 0,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    });
  });
  afterEach(() => vi.restoreAllMocks());

  it("renders a resize handle per column and stays auto-sized until used", () => {
    const { container } = renderTable();

    expect(screen.getAllByRole("separator", { name: "Resize column" })).toHaveLength(2);
    // Untouched: no fixed layout, no explicit column widths.
    expect(container.querySelector("colgroup")).toBeNull();
    expect(container.querySelector("table")).toHaveClass("w-full");
  });

  it("widens a column with the keyboard and pins the others", () => {
    const { container } = renderTable();
    const [firstHandle] = screen.getAllByRole("separator", { name: "Resize column" });

    firstHandle.focus();
    fireEvent.keyDown(firstHandle, { key: "ArrowRight" });

    const table = container.querySelector("table");
    expect(table).toHaveClass("table-fixed");
    const cols = container.querySelectorAll("col");
    // First column grew by one step (120 + 24); the second stayed pinned at 120.
    expect(cols[0]).toHaveStyle({ width: "144px" });
    expect(cols[1]).toHaveStyle({ width: "120px" });
  });

  it("narrows a column back down with the opposite arrow", () => {
    const { container } = renderTable();
    const [firstHandle] = screen.getAllByRole("separator", { name: "Resize column" });

    fireEvent.keyDown(firstHandle, { key: "ArrowRight" });
    fireEvent.keyDown(firstHandle, { key: "ArrowLeft" });

    expect(container.querySelectorAll("col")[0]).toHaveStyle({ width: "120px" });
  });
});
