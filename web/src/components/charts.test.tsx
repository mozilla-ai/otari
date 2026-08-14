import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { ChartLegend, ChartTooltip, Sparkline, TrendChart, type SeriesDef } from "@/components/charts";

const COST_SERIES: SeriesDef[] = [{ key: "cost", label: "Cost", color: "var(--otari-brand)" }];
const STACK_SERIES: SeriesDef[] = [
  { key: "success", label: "Succeeded", color: "var(--otari-brand)" },
  { key: "errors", label: "Failed", color: "var(--otari-danger)" },
];

describe("TrendChart", () => {
  it("renders a single-series recharts bar chart with one bar per point", () => {
    const { container } = render(
      <TrendChart
        data={[
          { x: "2025-07-19T00:00:00Z", cost: 400 },
          { x: "2025-07-20T00:00:00Z", cost: 840.5 },
        ]}
        series={COST_SERIES}
        formatValue={(v: number) => `$${v}`}
        ariaLabel="cost per day"
      />,
    );

    // The chart is labeled for screen readers and backed by recharts (the SVG
    // surface only exists when the library mounted and measured a size).
    expect(screen.getByRole("img", { name: "cost per day" })).toBeInTheDocument();
    expect(container.querySelector(".recharts-surface")).not.toBeNull();
    // One bar per data point, single series: nothing encoded by hue alone.
    expect(container.querySelectorAll(".recharts-bar-rectangle")).toHaveLength(2);
  });

  it("renders one stacked rectangle per non-zero (series, point)", () => {
    const { container } = render(
      <TrendChart
        data={[
          { x: "2025-07-19T00:00:00Z", success: 10, errors: 2 },
          { x: "2025-07-20T00:00:00Z", success: 7, errors: 0 },
        ]}
        series={STACK_SERIES}
        formatValue={(v: number) => String(v)}
        ariaLabel="requests per day"
      />,
    );
    // The zero-height error segment on the second bucket draws nothing.
    expect(container.querySelectorAll(".recharts-bar-rectangle")).toHaveLength(3);
  });

  it("is only brush-selectable when a handler and 2+ buckets exist", () => {
    const onSelect = vi.fn();
    const { container, rerender } = render(
      <TrendChart
        data={[
          { x: "a", cost: 1 },
          { x: "b", cost: 2 },
        ]}
        series={COST_SERIES}
        formatValue={String}
        ariaLabel="c"
        onSelectRange={onSelect}
      />,
    );
    expect(container.querySelector(".cursor-crosshair")).not.toBeNull();
    rerender(
      <TrendChart data={[{ x: "a", cost: 1 }]} series={COST_SERIES} formatValue={String} ariaLabel="c" onSelectRange={onSelect} />,
    );
    expect(container.querySelector(".cursor-crosshair")).toBeNull();
  });
});

describe("Sparkline", () => {
  it("renders a recharts sparkline line for KPI tiles", () => {
    const { container } = render(<Sparkline values={[1, 3, 2, 5]} ariaLabel="Spend trend" />);

    expect(screen.getByRole("img", { name: "Spend trend" })).toBeInTheDocument();
    expect(container.querySelector(".recharts-line")).not.toBeNull();
  });
});

describe("ChartLegend", () => {
  it("labels every series of a multi-series chart", () => {
    render(<ChartLegend series={STACK_SERIES} />);
    expect(screen.getByText("Succeeded")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("renders nothing for a single series (the title names it)", () => {
    const { container } = render(<ChartLegend series={COST_SERIES} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("ChartTooltip", () => {
  const fmt = (v: number) => `$${v.toFixed(2)}`;

  it("formats the active value with its label", () => {
    render(<ChartTooltip active label="Jul 20" payload={[{ value: 840.5 }]} formatValue={fmt} />);
    expect(screen.getByText("Jul 20")).toBeInTheDocument();
    expect(screen.getByText("$840.50")).toBeInTheDocument();
  });

  it("renders a zero value rather than treating it as empty", () => {
    render(<ChartTooltip active label="Jul 20" payload={[{ value: 0 }]} formatValue={fmt} />);
    expect(screen.getByText("$0.00")).toBeInTheDocument();
  });

  it("renders nothing when inactive, empty, or non-numeric", () => {
    const { container: inactive } = render(<ChartTooltip payload={[{ value: 5 }]} formatValue={fmt} />);
    expect(inactive).toBeEmptyDOMElement();
    const { container: empty } = render(<ChartTooltip active payload={[]} formatValue={fmt} />);
    expect(empty).toBeEmptyDOMElement();
    const { container: nonNumeric } = render(<ChartTooltip active payload={[{ value: "n/a" }]} formatValue={fmt} />);
    expect(nonNumeric).toBeEmptyDOMElement();
  });

  it("lists per-series rows, hides zero rows, and totals a stack", () => {
    render(
      <ChartTooltip
        active
        label="2025-07-20T00:00:00Z"
        formatLabel={(iso) => iso.slice(0, 10)}
        payload={[
          { value: 10, name: "Succeeded", color: "#111" },
          { value: 0, name: "Failed", color: "#222" },
        ]}
        formatValue={(v) => String(v)}
        showTotal
      />,
    );
    expect(screen.getByText("2025-07-20")).toBeInTheDocument();
    expect(screen.getByText("Succeeded")).toBeInTheDocument();
    // The zero series row is hidden in a stack; the total still sums all rows.
    expect(screen.queryByText("Failed")).toBeNull();
    expect(screen.getByText("Total")).toBeInTheDocument();
  });
});
