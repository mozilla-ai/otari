import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { RefreshButton, StatCard } from "@/components/ui";

describe("StatCard", () => {
  it("renders its label and value", () => {
    render(<StatCard label="Tracked cost" value="$12.34" />);
    expect(screen.getByText("Tracked cost")).toBeInTheDocument();
    expect(screen.getByText("$12.34")).toBeInTheDocument();
  });

  it("fits its grid track and avoids double padding", () => {
    // min-w-0 lets the tile shrink to its grid track (a fixed min-width overflowed
    // and overlapped the neighbour at two-up on mobile); p-0 zeroes HeroUI's own
    // card padding so it does not stack with Card.Content's and double the height.
    const { container } = render(<StatCard label="Requests" value="0" />);
    // Assert on the rendered root element rather than HeroUI's internal ".card"
    // class, so a library-internal class rename can't silently break this.
    const root = container.firstElementChild!;
    expect(root.className).toContain("min-w-0");
    expect(root.className).toContain("p-0");
  });
});

describe("RefreshButton", () => {
  it("fires onRefresh and shows a freshness label", async () => {
    const user = userEvent.setup();
    const onRefresh = vi.fn();
    render(<RefreshButton onRefresh={onRefresh} updatedAt={Date.now() - 5_000} />);
    expect(screen.getByText(/Updated/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Refresh" }));
    expect(onRefresh).toHaveBeenCalledOnce();
  });

  it("hides the timestamp before the first load and disables while fetching", () => {
    const onRefresh = vi.fn();
    render(<RefreshButton onRefresh={onRefresh} isFetching updatedAt={0} />);
    expect(screen.queryByText(/Updated/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh" })).toBeDisabled();
  });
});
