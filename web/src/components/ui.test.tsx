import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { EmptyState, PageLoading, StatCard } from "@/components/ui";

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

describe("EmptyState", () => {
  it("renders the title as a heading with its description", () => {
    render(<EmptyState title="No budgets yet" description="Create one to cap spending." />);
    expect(screen.getByRole("heading", { name: "No budgets yet" })).toBeInTheDocument();
    expect(screen.getByText("Create one to cap spending.")).toBeInTheDocument();
  });

  it("fires onAction when the call to action is pressed", async () => {
    const user = userEvent.setup();
    const onAction = vi.fn();
    render(<EmptyState title="No API keys yet" actionLabel="Create your first key" onAction={onAction} />);
    await user.click(screen.getByRole("button", { name: "Create your first key" }));
    expect(onAction).toHaveBeenCalledOnce();
  });

  it("omits the action entirely for a purely informational empty state", () => {
    render(<EmptyState title="No usage yet" description="Spend appears here once traffic flows." />);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("disables the call to action and suppresses onAction when isActionDisabled is set", async () => {
    const user = userEvent.setup();
    const onAction = vi.fn();
    render(
      <EmptyState title="Welcome" actionLabel="Add your first provider" onAction={onAction} isActionDisabled />,
    );
    const button = screen.getByRole("button", { name: "Add your first provider" });
    expect(button).toBeDisabled();
    await user.click(button);
    expect(onAction).not.toHaveBeenCalled();
  });
});

describe("PageLoading", () => {
  it("exposes a status role so the wait is announced", () => {
    render(<PageLoading />);
    const status = screen.getByRole("status");
    expect(status).toHaveTextContent("Loading…");
  });
});
