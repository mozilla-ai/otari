import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { TablePagination } from "./TablePagination";

function setup(overrides: Partial<Parameters<typeof TablePagination>[0]> = {}) {
  const onPageChange = vi.fn();
  const onPageSizeChange = vi.fn();
  render(
    <TablePagination
      page={0}
      pageSize={100}
      total={4231}
      rowsOnPage={100}
      onPageChange={onPageChange}
      onPageSizeChange={onPageSizeChange}
      {...overrides}
    />,
  );
  return { onPageChange, onPageSizeChange };
}

describe("TablePagination", () => {
  it("shows a truthful range-of-total summary", () => {
    setup();
    expect(screen.getByText("1–100 of 4,231")).toBeInTheDocument();
    expect(screen.getByText("/ 43")).toBeInTheDocument();
  });

  it("disables first/prev on the first page", () => {
    setup();
    expect(screen.getByRole("button", { name: "First page" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Previous page" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Next page" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Last page" })).toBeEnabled();
  });

  it("jumps to the last page", async () => {
    const user = userEvent.setup();
    const { onPageChange } = setup();
    await user.click(screen.getByRole("button", { name: "Last page" }));
    expect(onPageChange).toHaveBeenCalledWith(42); // ceil(4231/100) - 1
  });

  it("commits a typed page, clamped to the page count", async () => {
    const user = userEvent.setup();
    const { onPageChange } = setup();
    const box = screen.getByRole("textbox", { name: "Page number" });
    await user.clear(box);
    await user.type(box, "999{enter}");
    expect(onPageChange).toHaveBeenCalledWith(42); // clamped to last page
  });

  it("changes rows per page", async () => {
    const user = userEvent.setup();
    const { onPageSizeChange } = setup();
    await user.selectOptions(screen.getByRole("combobox", { name: "Rows per page" }), "100");
    expect(onPageSizeChange).toHaveBeenCalledWith(100);
  });

  it("with an unknown total, hides last and uses the next fallback", () => {
    setup({ total: null, hasNextFallback: true, rowsOnPage: 100 });
    expect(screen.getByText("1–100")).toBeInTheDocument();
    expect(screen.queryByText(/\/ /)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Last page" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Next page" })).toBeEnabled();
  });

  it("with an unknown total and a short page, disables next", () => {
    setup({ total: null, hasNextFallback: false, rowsOnPage: 40, page: 2 });
    expect(screen.getByRole("button", { name: "Next page" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Previous page" })).toBeEnabled();
  });
});
