import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { BarTrendChart, ChartTooltip, Sparkline } from "@/components/charts";

describe("charts", () => {
  it("renders a single-series recharts bar chart with one bar per point", () => {
    const { container } = render(
      <BarTrendChart
        data={[
          { label: "Jul 19", value: 400 },
          { label: "Jul 20", value: 840.5 },
        ]}
        formatValue={(v) => `$${v}`}
        ariaLabel="cost per day"
      />,
    );

    // The chart is labelled for screen readers and backed by recharts (the SVG
    // surface only exists when the library mounted and measured a size).
    expect(screen.getByRole("img", { name: "cost per day" })).toBeInTheDocument();
    expect(container.querySelector(".recharts-surface")).not.toBeNull();
    // One bar per data point, single series: nothing encoded by hue alone.
    expect(container.querySelectorAll(".recharts-bar-rectangle")).toHaveLength(2);
  });

  it("renders a recharts sparkline line for KPI tiles", () => {
    const { container } = render(<Sparkline values={[1, 3, 2, 5]} ariaLabel="Spend trend" />);

    expect(screen.getByRole("img", { name: "Spend trend" })).toBeInTheDocument();
    expect(container.querySelector(".recharts-line")).not.toBeNull();
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
});
