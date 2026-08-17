import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { ConfirmDialog } from "./ConfirmDialog";

const base = {
  heading: "Delete rows",
  body: "This cannot be undone.",
  confirmLabel: "Delete",
  isPending: false,
  onConfirm: vi.fn(),
  onOpenChange: vi.fn(),
};

describe("ConfirmDialog", () => {
  it("renders nothing while closed", () => {
    render(<ConfirmDialog {...base} isOpen={false} />);
    expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
  });

  it("shows the heading, body, and confirm label when open", () => {
    render(<ConfirmDialog {...base} isOpen />);
    const dialog = screen.getByRole("alertdialog");
    expect(dialog).toHaveTextContent("Delete rows");
    expect(dialog).toHaveTextContent("This cannot be undone.");
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument();
  });

  it("fires onConfirm when the confirm button is pressed", async () => {
    const onConfirm = vi.fn();
    const user = userEvent.setup();
    render(<ConfirmDialog {...base} isOpen onConfirm={onConfirm} />);
    await user.click(screen.getByRole("button", { name: "Delete" }));
    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("cancels via onOpenChange(false) and does not confirm", async () => {
    const onConfirm = vi.fn();
    const onOpenChange = vi.fn();
    const user = userEvent.setup();
    render(<ConfirmDialog {...base} isOpen onConfirm={onConfirm} onOpenChange={onOpenChange} />);
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("renders a mutation error", () => {
    render(<ConfirmDialog {...base} isOpen error={new Error("boom")} />);
    expect(screen.getByRole("alertdialog")).toHaveTextContent("boom");
  });

  it("disables Cancel while pending so the op can't be abandoned mid-flight", () => {
    render(<ConfirmDialog {...base} isOpen isPending />);
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();
  });
});
