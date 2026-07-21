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

// jsdom doesn't implement pointer capture, so stub it; the handlers gate their
// work on hasPointerCapture, which we make report the drag as captured.
function stubPointerCapture(el: HTMLElement) {
  el.setPointerCapture = vi.fn();
  el.releasePointerCapture = vi.fn();
  el.hasPointerCapture = vi.fn(() => true);
}

// jsdom's synthetic PointerEvent drops clientX, so dispatch a MouseEvent (which
// carries clientX) under the pointer event name, with pointerId attached for the
// capture checks. React fires its onPointer* handlers for the matching type.
function firePointer(el: HTMLElement, type: string, props: { clientX?: number; pointerId?: number }) {
  const event = new MouseEvent(type, { bubbles: true, cancelable: true, clientX: props.clientX ?? 0 });
  Object.defineProperty(event, "pointerId", { value: props.pointerId ?? 1 });
  fireEvent(el, event);
}

describe("Table resizable columns", () => {
  beforeEach(() => {
    // jsdom has no layout, so header cells report zero width. Pin a width so a
    // resize snapshots a deterministic starting point.
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

    expect(screen.getAllByTitle("Drag to resize")).toHaveLength(2);
    // Untouched: no fixed layout, no explicit column widths.
    expect(container.querySelector("colgroup")).toBeNull();
    expect(container.querySelector("table")).toHaveClass("w-full");
  });

  it("widens a column on drag and pins the others", () => {
    const { container } = renderTable();
    const [firstHandle] = screen.getAllByTitle("Drag to resize");
    stubPointerCapture(firstHandle);

    firePointer(firstHandle, "pointerdown", { pointerId: 1, clientX: 100 });
    firePointer(firstHandle, "pointermove", { pointerId: 1, clientX: 140 });
    firePointer(firstHandle, "pointerup", { pointerId: 1 });

    const table = container.querySelector("table");
    expect(table).toHaveClass("table-fixed");
    const cols = container.querySelectorAll("col");
    // First column grew by the drag delta (120 + 40); the second stayed pinned.
    expect(cols[0]).toHaveStyle({ width: "160px" });
    expect(cols[1]).toHaveStyle({ width: "120px" });
  });

  it("stops resizing when the pointer is canceled", () => {
    const { container } = renderTable();
    const [firstHandle] = screen.getAllByTitle("Drag to resize");
    stubPointerCapture(firstHandle);

    firePointer(firstHandle, "pointerdown", { pointerId: 1, clientX: 100 });
    firePointer(firstHandle, "pointermove", { pointerId: 1, clientX: 140 });
    firePointer(firstHandle, "pointercancel", { pointerId: 1 });
    // A move after cancel must be ignored: the drag state was cleared.
    firePointer(firstHandle, "pointermove", { pointerId: 1, clientX: 400 });

    expect(container.querySelectorAll("col")[0]).toHaveStyle({ width: "160px" });
  });
});
