import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { UsageGroupRow, UsageTotals } from "@/api/types";

import { ShareDialog } from "./ShareDialog";

// jsdom has no canvas, so the rasterizer cannot run for real here; mocked so the
// dialog's wiring around it is exercisable. The claim that the PNG itself is
// correct lives in the Playwright suite.
vi.mock("@/lib/shareImage", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/shareImage")>();
  return {
    ...actual,
    rasterize: vi.fn(async () => new Blob(["png"], { type: "image/png" })),
    canCopyImages: vi.fn(() => true),
    copyBlobAsImage: vi.fn(async () => true),
  };
});

const totals: UsageTotals = {
  cost: 3.44,
  prompt_tokens: 1_800_000,
  completion_tokens: 460_000,
  total_tokens: 2_260_000,
  cache_read_tokens: 331_600,
  cache_write_tokens: 0,
  request_count: 771,
  error_count: 22,
  avg_latency_ms: 808,
  unpriced_requests: 63,
  billed_input_tokens: 1_800_000,
  billed_output_tokens: 460_000,
};

const rows: UsageGroupRow[] = [
  { key: "llama-3.3-70b-versatile", cost: 0.2, tokens: 268_459, requests: 96, is_other: false },
  { key: "gpt-4o", cost: 2.4, tokens: 306_880, requests: 105, is_other: false },
  { key: "llama3.1:8b", cost: 0, tokens: 167_887, requests: 60, is_other: false },
];

function renderDialog() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ShareDialog
        totals={totals}
        series={[]}
        modelRows={rows}
        windowLabel="Jul 29 – Aug 11"
        scopeSuffix=""
        startIso="2026-07-29T00:00:00Z"
        endIso="2026-08-11T00:00:00Z"
        isStale={false}
        onClose={() => undefined}
      />
    </QueryClientProvider>,
  );
}

describe("ShareDialog", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("offers download as the terminal action, with no post-composition surface", () => {
    renderDialog();
    expect(screen.getByRole("button", { name: "Download PNG" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Post on X" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Suggested post text")).not.toBeInTheDocument();
  });

  it("re-leads the card when the lead stat changes", async () => {
    const user = userEvent.setup();
    renderDialog();
    const card = document.querySelector('[aria-label^="Usage card"]') as HTMLElement;
    // Requests is the default lead for this fixture.
    expect(within(card).getByText("771")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Tokens" }));
    expect(within(card).getByText("2.3M")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Tokens" })).toHaveAttribute("aria-pressed", "true");
  });

  it("does not explain the clipboard's origin rules", () => {
    renderDialog();
    expect(screen.queryByText(/secure \(https\) origin/i)).not.toBeInTheDocument();
  });

  it("does not offer cache hit rate as the card's lead", () => {
    // It can still appear as a secondary stat: it is a gateway-tuning number, not
    // a claim a reader outside the deployment can do anything with.
    renderDialog();
    expect(screen.queryByRole("button", { name: "Cache hits" })).not.toBeInTheDocument();
  });

  it("has exactly one control for the hero slot, so no two buttons share a name", () => {
    renderDialog();
    // An earlier draft had both a preset row and a "lead with" row, which produced
    // two different buttons named "Spend".
    const names = screen.getAllByRole("button").map((b) => b.textContent);
    expect(new Set(names).size).toBe(names.length);
  });

  it("presents as a modal dialog overlay, not an inline card", () => {
    renderDialog();
    const dialog = screen.getByRole("alertdialog");
    expect(dialog).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Download PNG" })).toBeInTheDocument();
  });

  it("carries no window or filter control: data scope comes from the page", () => {
    renderDialog();
    expect(screen.queryByLabelText("User")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Model")).not.toBeInTheDocument();
    expect(screen.getByText(/window and filters currently applied above/i)).toBeInTheDocument();
  });

  it("blocks the actions while the page's numbers are still in flight", () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <ShareDialog
          totals={totals}
          series={[]}
          modelRows={rows}
          windowLabel="Jul 29 – Aug 11"
          scopeSuffix=""
          startIso="2026-07-29T00:00:00Z"
          endIso="2026-08-11T00:00:00Z"
          isStale
          onClose={() => undefined}
        />
      </QueryClientProvider>,
    );
    expect(screen.getByRole("button", { name: "Download PNG" })).toBeDisabled();
    expect(screen.getByText(/Waiting for the current numbers/i)).toBeInTheDocument();
  });

  it("does not claim a copy that failed", async () => {
    const user = userEvent.setup();
    const shareImage = await import("@/lib/shareImage");
    // The clipboard write is refused (denied permission, insecure origin, ...).
    vi.mocked(shareImage.copyBlobAsImage).mockResolvedValue(false);
    renderDialog();
    await user.click(screen.getByRole("button", { name: "Copy image" }));
    expect(await screen.findByText(/could not be copied/i)).toBeInTheDocument();
    expect(screen.queryByText("Image copied")).not.toBeInTheDocument();
  });

  it("falls back per field when stored presentation holds an unknown value", () => {
    // A shape from an older build (or hand-edited storage) used to reach
    // CARD_SIZES[ratio] and crash on the destructure.
    localStorage.setItem("otari.share.presentation.v1", JSON.stringify({ ratio: "portrait", rows: 4, theme: "neon" }));
    renderDialog();
    expect(screen.getByRole("button", { name: "Download PNG" })).toBeInTheDocument();
  });

  it("survives a corrupt stored presentation instead of crashing", () => {
    localStorage.setItem("otari.share.presentation.v1", "{ not json");
    renderDialog();
    expect(screen.getByRole("button", { name: "Download PNG" })).toBeInTheDocument();
  });
});
