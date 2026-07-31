import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { USAGE_PRESETS } from "@/lib/timeRange";
import { ActivityTimeline, type TimelinePoint } from "./ActivityTimeline";

const SERIES: TimelinePoint[] = [
  { bucketStart: "2026-07-10T00:00:00Z", requests: 12 },
  { bucketStart: "2026-07-11T00:00:00Z", requests: 40 },
  { bucketStart: "2026-07-12T00:00:00Z", requests: 8 },
];

function renderTimeline(overrides: Partial<React.ComponentProps<typeof ActivityTimeline>> = {}) {
  const onPreset = vi.fn();
  const onSelectRange = vi.fn();
  const onSelectFull = vi.fn();
  render(
    <ActivityTimeline
      presets={USAGE_PRESETS}
      extentKey="30d"
      onPreset={onPreset}
      onSelectRange={onSelectRange}
      onSelectFull={onSelectFull}
      series={SERIES}
      bucket="day"
      windowStart="2026-07-10T00:00:00.000Z"
      windowEnd="2026-07-13T00:00:00.000Z"
      {...overrides}
    />,
  );
  return { onPreset, onSelectRange, onSelectFull };
}

describe("ActivityTimeline", () => {
  it("renders presets and highlights the extent, reporting a picked preset", async () => {
    const user = userEvent.setup();
    const { onPreset } = renderTimeline();
    await user.click(screen.getByRole("button", { name: "7d" }));
    expect(onPreset).toHaveBeenCalledWith(expect.objectContaining({ key: "7d" }));
  });

  it("captions the active window (inclusive end, UTC)", () => {
    renderTimeline();
    // Assert on day numbers, the range separator, and the UTC marker rather than a
    // month abbreviation: the caption formats with the runtime locale, so "Jul"
    // would make this suite fail outside en-US.
    const caption = (screen.getByText(/Showing/).textContent ?? "").replace(/\s+/g, " ");
    expect(caption).toMatch(/\b10\b/);
    expect(caption).toMatch(/\b12\b/); // 2026-07-13 exclusive -> inclusive day 12
    expect(caption).toContain("–"); // start – end range
    expect(caption).toContain("UTC");
  });

  it("captions an unbounded window as All time", () => {
    renderTimeline({ windowStart: undefined, windowEnd: undefined });
    expect(screen.getByText(/Showing/)).toHaveTextContent("All time");
  });

  it("renders the bucket unit label", () => {
    renderTimeline();
    expect(screen.getByText("Requests / day")).toBeInTheDocument();
  });

  it("shows an empty state when there is no activity", () => {
    renderTimeline({ series: [] });
    expect(screen.getByText(/No activity in this range/)).toBeInTheDocument();
  });

  it("renders a dual-thumb range slider positioned on the active window", () => {
    renderTimeline();
    // Window covers the whole 3-bucket series: thumbs sit at the extremes of the
    // fractional bucket scale [0, 3]. Queried by their edge labels so the pan
    // strip (also role="slider") is excluded; DOM order is start then end.
    const thumbs = screen.getAllByRole("slider", { name: /^Window/ });
    expect(thumbs).toHaveLength(2);
    expect(thumbs[0]).toHaveValue("0");
    expect(thumbs[1]).toHaveValue("3");
  });

  it("positions the thumbs on a sub-window", () => {
    renderTimeline({
      // Jul 11 only (one bucket of the three).
      windowStart: "2026-07-11T00:00:00.000Z",
      windowEnd: "2026-07-12T00:00:00.000Z",
    });
    const thumbs = screen.getAllByRole("slider", { name: /^Window/ });
    expect(thumbs[0]).toHaveValue("1");
    expect(thumbs[1]).toHaveValue("2");
  });

  it("promotes to the next larger preset when zooming out at the full extent", async () => {
    const user = userEvent.setup();
    // Window covers the whole series, so the extent itself must widen: 30d -> 90d.
    const { onPreset, onSelectRange } = renderTimeline();
    await user.click(screen.getByRole("button", { name: "Zoom out" }));
    expect(onPreset).toHaveBeenCalledWith(expect.objectContaining({ key: "90d" }));
    expect(onSelectRange).not.toHaveBeenCalled();
  });

  it("doubles a sub-window around its center when zooming out", async () => {
    const user = userEvent.setup();
    const series = [
      { bucketStart: "2026-07-10T00:00:00Z", requests: 1 },
      { bucketStart: "2026-07-11T00:00:00Z", requests: 2 },
      { bucketStart: "2026-07-12T00:00:00Z", requests: 3 },
      { bucketStart: "2026-07-13T00:00:00Z", requests: 4 },
      { bucketStart: "2026-07-14T00:00:00Z", requests: 5 },
    ];
    const { onSelectRange } = renderTimeline({
      series,
      // Jul 12 only (the middle bucket of five).
      windowStart: "2026-07-12T00:00:00.000Z",
      windowEnd: "2026-07-13T00:00:00.000Z",
    });
    await user.click(screen.getByRole("button", { name: "Zoom out" }));
    // One bucket doubles to two, centered, then snaps outward to whole buckets:
    // Jul 11 .. Jul 14 (exclusive).
    expect(onSelectRange).toHaveBeenCalledWith("2026-07-11T00:00:00.000Z", "2026-07-14T00:00:00.000Z");
  });

  it("offers Reset only when zoomed, and it restores the full extent", async () => {
    const user = userEvent.setup();
    const { onSelectFull } = renderTimeline({
      windowStart: "2026-07-11T00:00:00.000Z",
      windowEnd: "2026-07-12T00:00:00.000Z",
    });
    await user.click(screen.getByRole("button", { name: "Reset" }));
    expect(onSelectFull).toHaveBeenCalledOnce();
  });

  it("hides Reset at the full extent", () => {
    renderTimeline();
    expect(screen.queryByRole("button", { name: "Reset" })).not.toBeInTheDocument();
  });

  it("falls back to the smallest broader preset when the extent is not a preset", async () => {
    const user = userEvent.setup();
    // A drill-down window: extentKey is the custom sentinel, series spans 3 days.
    const { onPreset } = renderTimeline({ extentKey: "custom" });
    await user.click(screen.getByRole("button", { name: "Zoom out" }));
    // Smallest USAGE preset broader than 3 days is 7d.
    expect(onPreset).toHaveBeenCalledWith(expect.objectContaining({ key: "7d" }));
  });

  it("moves an edge by one whole bucket per arrow key", async () => {
    const user = userEvent.setup();
    const { onSelectRange } = renderTimeline();
    // Focus the end thumb and nudge it left: the full 3-bucket window narrows to
    // the first two buckets (arrows are rerouted to whole-bucket steps; the
    // slider's own 0.1-bucket step would be snapped back by the commit).
    screen.getByRole("slider", { name: /Window end/ }).focus();
    await user.keyboard("{ArrowLeft}");
    expect(onSelectRange).toHaveBeenCalledWith("2026-07-10T00:00:00.000Z", "2026-07-12T00:00:00.000Z");
  });

  it("pans the window by whole buckets from the keyboard", async () => {
    const user = userEvent.setup();
    const series = [
      { bucketStart: "2026-07-10T00:00:00Z", requests: 1 },
      { bucketStart: "2026-07-11T00:00:00Z", requests: 2 },
      { bucketStart: "2026-07-12T00:00:00Z", requests: 3 },
      { bucketStart: "2026-07-13T00:00:00Z", requests: 4 },
      { bucketStart: "2026-07-14T00:00:00Z", requests: 5 },
    ];
    // A two-bucket window (Jul 11 .. Jul 12); the pan strip slides it right by one
    // bucket without resizing it, so the keyboard reaches a mid-extent window.
    const { onSelectRange } = renderTimeline({
      series,
      windowStart: "2026-07-11T00:00:00.000Z",
      windowEnd: "2026-07-13T00:00:00.000Z",
    });
    screen.getByRole("slider", { name: "Pan the selected window" }).focus();
    await user.keyboard("{ArrowRight}");
    expect(onSelectRange).toHaveBeenCalledWith("2026-07-12T00:00:00.000Z", "2026-07-14T00:00:00.000Z");
  });

  it("pages the window by its own width", async () => {
    const user = userEvent.setup();
    const series = [
      { bucketStart: "2026-07-10T00:00:00Z", requests: 1 },
      { bucketStart: "2026-07-11T00:00:00Z", requests: 2 },
      { bucketStart: "2026-07-12T00:00:00Z", requests: 3 },
      { bucketStart: "2026-07-13T00:00:00Z", requests: 4 },
      { bucketStart: "2026-07-14T00:00:00Z", requests: 5 },
    ];
    // A two-bucket window at the left edge; Page Up pages right by one full span.
    const { onSelectRange } = renderTimeline({
      series,
      windowStart: "2026-07-10T00:00:00.000Z",
      windowEnd: "2026-07-12T00:00:00.000Z",
    });
    screen.getByRole("slider", { name: "Pan the selected window" }).focus();
    await user.keyboard("{PageUp}");
    expect(onSelectRange).toHaveBeenCalledWith("2026-07-12T00:00:00.000Z", "2026-07-14T00:00:00.000Z");
  });

  it("reports the reachable pan range on the strip, not the full extent", () => {
    // A one-bucket window in the three-bucket series: the left edge can only travel
    // 0..(n - span) = 0..2, and currently sits at bucket 1.
    renderTimeline({
      windowStart: "2026-07-11T00:00:00.000Z",
      windowEnd: "2026-07-12T00:00:00.000Z",
    });
    const pan = screen.getByRole("slider", { name: "Pan the selected window" });
    expect(pan).toHaveAttribute("aria-valuemin", "0");
    expect(pan).toHaveAttribute("aria-valuemax", "2");
    expect(pan).toHaveAttribute("aria-valuenow", "1");
    expect(pan).toHaveAttribute("aria-disabled", "false");
  });

  it("announces the pan strip disabled at the full extent", () => {
    // Full-window default: nothing to pan, but the control stays in the tab order.
    renderTimeline();
    const pan = screen.getByRole("slider", { name: "Pan the selected window" });
    expect(pan).toHaveAttribute("aria-disabled", "true");
    expect(pan).toHaveAttribute("tabindex", "0");
  });
});
