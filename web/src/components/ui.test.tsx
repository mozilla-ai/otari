import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CopyableValue, CopyButton, EmptyState, PageLoading, RefreshButton, StatCard } from "@/components/ui";

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

describe("CopyButton", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("writes the value to the clipboard and confirms over the icon", async () => {
    const user = userEvent.setup();
    render(<CopyButton value="anthropic:claude-opus-4" label="model id" />);

    // Nothing is shown until a copy happens: this reports an event, so it must
    // not open on hover the way a hint tooltip would.
    await user.hover(screen.getByRole("button", { name: "Copy model id" }));
    expect(screen.queryByText("Copied!")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Copy model id" }));

    expect(await navigator.clipboard.readText()).toBe("anthropic:claude-opus-4");
    expect(await screen.findByText("Copied!")).toBeInTheDocument();
  });

  it("says the copy was blocked rather than claiming one when no path works", async () => {
    // The Clipboard API refuses and jsdom has no document.execCommand, so both
    // paths in copyToClipboard are exhausted.
    const user = userEvent.setup();
    vi.spyOn(navigator.clipboard, "writeText").mockRejectedValue(new Error("not a secure context"));
    render(<CopyButton value="openai:gpt-4o" label="model id" />);

    await user.click(screen.getByRole("button", { name: "Copy model id" }));

    expect(await screen.findByText(/Copy blocked/)).toBeInTheDocument();
    expect(screen.queryByText("Copied!")).not.toBeInTheDocument();
  });

  it("keeps the confirmation out of the cell, so it cannot reflow the row", async () => {
    const user = userEvent.setup();
    const { container } = render(<CopyButton value="openai:gpt-4o" label="model id" />);
    expect(container.textContent).toBe("");

    await user.click(screen.getByRole("button", { name: "Copy model id" }));
    await screen.findByText("Copied!");

    // The confirmation is an overlay, not a sibling of the id it copied.
    expect(container.textContent).toBe("");
  });

  it("clears the confirmation on its own", async () => {
    const user = userEvent.setup();
    render(<CopyButton value="openai:gpt-4o" label="model id" />);

    await user.click(screen.getByRole("button", { name: "Copy model id" }));
    expect(await screen.findByText("Copied!")).toBeInTheDocument();

    // Real timers: the clipboard write is a promise, and driving userEvent with
    // fake ones deadlocks against it. 1.5s dismissal, so 3s is a safe ceiling.
    await waitFor(() => expect(screen.queryByText("Copied!")).not.toBeInTheDocument(), { timeout: 3_000 });
  });
});

describe("CopyableValue", () => {
  it("copies the value, which need not be what is displayed", async () => {
    const user = userEvent.setup();
    render(
      <CopyableValue value="openai:gpt-4o-2024-11-20" label="model id">
        gpt-4o-2024-11-20
      </CopyableValue>,
    );

    expect(screen.getByText("gpt-4o-2024-11-20")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Copy model id" }));
    expect(await navigator.clipboard.readText()).toBe("openai:gpt-4o-2024-11-20");
  });

  it("keeps a row press from starting on the value, so a drag can highlight it", () => {
    // The whole reason highlighting an id in a table used to fail: react-aria's
    // row press toggles selection on pointer down, and that re-render lands
    // mid-drag and discards the browser's nascent selection (#478). The value
    // stops the pointer sequence from reaching the row.
    const onRowPointerDown = vi.fn();
    const onRowMouseDown = vi.fn();
    render(
      <div onPointerDown={onRowPointerDown} onMouseDown={onRowMouseDown}>
        <CopyableValue value="anthropic:claude-opus-4-5-20251101" label="model id" />
      </div>,
    );

    const value = screen.getByText("anthropic:claude-opus-4-5-20251101");
    fireEvent.pointerDown(value, { pointerId: 1, pointerType: "mouse", button: 0 });
    fireEvent.mouseDown(value, { button: 0 });

    expect(onRowPointerDown).not.toHaveBeenCalled();
    expect(onRowMouseDown).not.toHaveBeenCalled();
    // Selectable in its own right, so an inherited `user-select: none` from a
    // press elsewhere in the row cannot suppress it.
    expect(value.className).toContain("select-text");
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
