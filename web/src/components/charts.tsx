import type { ReactNode } from "react";
import { Bar, BarChart, Line, LineChart, ResponsiveContainer, Tooltip, XAxis } from "recharts";

// Shared chart primitives for the dashboard, built on recharts. Pages compose
// these instead of hand-rolling SVG, so tooltips, responsive sizing, and axis
// handling come from one place. Every chart here is single-series and single
// color (the brand token): nothing is encoded by hue alone, and the surrounding
// data tables / captions stay the accessible source of truth.

const BRAND = "var(--otari-brand)";

// One point in a single-series trend: an x-axis/tooltip label and its value.
export interface ChartPoint {
  label: string;
  value: number;
}

// Tooltip body. recharts clones this element and injects `active`, `payload`,
// and `label` at render time, so only `formatValue` is passed by the caller.
function ChartTooltip({
  active,
  label,
  payload,
  formatValue,
}: {
  active?: boolean;
  label?: ReactNode;
  payload?: readonly { value?: number }[];
  formatValue: (value: number) => string;
}) {
  const value = payload?.[0]?.value;
  if (!active || value == null) {
    return null;
  }
  return (
    <div className="rounded-md border border-[var(--otari-line)] bg-[var(--otari-surface)] px-2 py-1 text-xs shadow-sm">
      <div className="text-[var(--otari-muted)]">{label}</div>
      <div className="font-medium tabular-nums text-[var(--otari-ink)]">{formatValue(value)}</div>
    </div>
  );
}

// A single-series bar chart of one metric over time. Responsive width, fixed
// height; the y-axis is omitted (the caller's caption carries the peak) and the
// x-axis auto-thins its ticks so labels never collide. The tooltip shows the
// formatted value for the hovered bucket.
export function BarTrendChart({
  data,
  formatValue,
  ariaLabel,
  height = 200,
}: {
  data: ChartPoint[];
  formatValue: (value: number) => string;
  ariaLabel: string;
  height?: number;
}) {
  return (
    <div role="img" aria-label={ariaLabel} className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
          <XAxis
            dataKey="label"
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
            minTickGap={24}
            tick={{ fontSize: 10, fill: "var(--otari-muted)" }}
          />
          <Tooltip cursor={{ fill: "var(--otari-line)", opacity: 0.35 }} content={<ChartTooltip formatValue={formatValue} />} />
          <Bar dataKey="value" fill={BRAND} radius={[2, 2, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// A compact, axis-free trend line for KPI tiles. Conveys shape only: no ticks,
// no tooltip, one color. `ariaLabel` should describe what the trend is (e.g.
// "Spend trend over the selected window") so it is legible without the visual.
export function Sparkline({
  values,
  ariaLabel,
  height = 32,
}: {
  values: number[];
  ariaLabel: string;
  height?: number;
}) {
  const data = values.map((value, index) => ({ index, value }));
  return (
    <div role="img" aria-label={ariaLabel} className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
          <Line type="monotone" dataKey="value" stroke={BRAND} strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
