import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { FilterChips } from "./FilterChips";

describe("FilterChips", () => {
  it("renders a chip per active filter and clears one on ✕", async () => {
    const user = userEvent.setup();
    const clearModel = vi.fn();
    render(
      <FilterChips
        chips={[
          { key: "model", label: "Model", value: "gpt-5.6", onClear: clearModel },
          { key: "user", label: "User", value: "alice", onClear: vi.fn() },
        ]}
      >
        <div>pickers</div>
      </FilterChips>,
    );
    expect(screen.getByText("gpt-5.6")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Remove Model filter" }));
    expect(clearModel).toHaveBeenCalledOnce();
  });

  it("keeps the picker row hidden until 'Add filter' is toggled", async () => {
    const user = userEvent.setup();
    render(
      <FilterChips chips={[]}>
        <label>User picker</label>
      </FilterChips>,
    );
    const region = screen.getByText("User picker").closest("div")!;
    expect(region.className).toContain("hidden");
    await user.click(screen.getByRole("button", { name: "Add filter" }));
    expect(region.className).not.toContain("hidden");
  });

  it("offers 'Clear all' only when filters are active", async () => {
    const user = userEvent.setup();
    const clearAll = vi.fn();
    const { rerender } = render(
      <FilterChips chips={[]} onClearAll={clearAll}>
        <div />
      </FilterChips>,
    );
    expect(screen.queryByRole("button", { name: "Clear all" })).not.toBeInTheDocument();

    rerender(
      <FilterChips chips={[{ key: "user", label: "User", value: "alice", onClear: vi.fn() }]} onClearAll={clearAll}>
        <div />
      </FilterChips>,
    );
    await user.click(screen.getByRole("button", { name: "Clear all" }));
    expect(clearAll).toHaveBeenCalledOnce();
  });
});
