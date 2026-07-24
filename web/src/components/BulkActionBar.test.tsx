import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Button } from "@heroui/react";
import { describe, expect, it, vi } from "vitest";

import { BulkActionBar } from "./BulkActionBar";

describe("BulkActionBar", () => {
  it("shows the page selection count and actions", () => {
    render(
      <BulkActionBar
        selectedCount={3}
        allMatching={false}
        matchingTotal={4231}
        canSelectAllMatching={false}
        onSelectAllMatching={vi.fn()}
        onClear={vi.fn()}
      >
        <Button size="sm">Delete</Button>
      </BulkActionBar>,
    );
    expect(screen.getByText("3 selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument();
  });

  it("offers select-all-matching when the page is full and more match", async () => {
    const user = userEvent.setup();
    const onSelectAllMatching = vi.fn();
    render(
      <BulkActionBar
        selectedCount={100}
        allMatching={false}
        matchingTotal={4231}
        canSelectAllMatching
        onSelectAllMatching={onSelectAllMatching}
        onClear={vi.fn()}
      >
        <Button size="sm">Delete</Button>
      </BulkActionBar>,
    );
    await user.click(screen.getByRole("button", { name: /Select all 4,231 matching/ }));
    expect(onSelectAllMatching).toHaveBeenCalled();
  });

  it("reads the whole-filter selection when all matching", () => {
    render(
      <BulkActionBar
        selectedCount={100}
        allMatching
        matchingTotal={4231}
        canSelectAllMatching
        onSelectAllMatching={vi.fn()}
        onClear={vi.fn()}
      >
        <Button size="sm">Delete</Button>
      </BulkActionBar>,
    );
    expect(screen.getByText("All 4,231 matching rows selected")).toBeInTheDocument();
    // The affordance is gone once all-matching is active.
    expect(screen.queryByRole("button", { name: /Select all/ })).not.toBeInTheDocument();
  });
});
