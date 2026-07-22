export function formatNumber(value: number | null | undefined): string {
  if (value == null) {
    return "0";
  }
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatCost(value: number | null | undefined): string {
  if (value == null) {
    return "$0.00";
  }
  // Show more precision for tiny per-request costs so they don't read as $0.00.
  const fractionDigits = value !== 0 && Math.abs(value) < 0.01 ? 4 : 2;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: fractionDigits,
  }).format(value);
}

// Compact token counts for context windows: 128000 -> "128K", 1000000 -> "1M".
// Returns an em-dash placeholder when unknown so table cells stay aligned.
export function formatContext(value: number | null | undefined): string {
  if (value == null) {
    return "—";
  }
  if (value >= 1_000_000) {
    const millions = value / 1_000_000;
    return `${Number.isInteger(millions) ? millions : millions.toFixed(1)}M`;
  }
  if (value >= 1000) {
    // Promote to "1M" rather than "1000K" when rounding lands on a thousand-K
    // (e.g. 999999 rounds to 1000K).
    const thousands = Math.round(value / 1000);
    return thousands >= 1000 ? "1M" : `${thousands}K`;
  }
  return String(value);
}

const MONTH_ABBREVIATIONS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

// models.dev release dates arrive as "YYYY-MM-DD" (occasionally just "YYYY-MM").
// Render a compact "Mon YYYY" for the table without pulling the value through a
// timezone-shifting Date parse. Returns an em-dash placeholder when unknown.
export function formatReleaseDate(value: string | null | undefined): string {
  if (!value) {
    return "—";
  }
  const match = /^(\d{4})-(\d{2})/.exec(value);
  if (!match) {
    return value;
  }
  const monthIndex = Number(match[2]) - 1;
  if (monthIndex < 0 || monthIndex > 11) {
    return match[1];
  }
  return `${MONTH_ABBREVIATIONS[monthIndex]} ${match[1]}`;
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) {
    return "—";
  }
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  return date.toLocaleString();
}

// Compact USD for aggregate tiles: cents precision (not the per-request 4dp that
// formatCost uses), so four+ figure totals stay readable. Non-null: callers guard
// nullable per-request costs (e.g. `cost === null ? "—" : formatUsd(cost)`).
const usdCompact = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 2,
});

export function formatUsd(value: number): string {
  return usdCompact.format(value);
}

// Compact token counts for aggregate tiles: 12.4M / 84.2k / 512.
export function formatTokens(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

export function formatPct(fraction: number): string {
  return `${(fraction * 100).toFixed(1)}%`;
}

// Period-over-period change. null when there is no comparable previous value
// (unbounded range, or a previous value of zero which would divide by zero).
export function deltaFraction(current: number, previous: number | undefined): number | null {
  if (previous === undefined || previous === 0) return null;
  return (current - previous) / previous;
}

export function formatRelative(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) {
    return "never";
  }
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  const seconds = Math.round((now - date.getTime()) / 1000);
  const future = seconds < 0;
  const abs = Math.abs(seconds);

  const units: [Intl.RelativeTimeFormatUnit, number][] = [
    ["second", 60],
    ["minute", 60],
    ["hour", 24],
    ["day", 30],
    ["month", 12],
    ["year", Number.POSITIVE_INFINITY],
  ];

  let value = abs;
  let unit: Intl.RelativeTimeFormatUnit = "second";
  for (const [candidate, divisor] of units) {
    unit = candidate;
    if (value < divisor) {
      break;
    }
    value = Math.floor(value / divisor);
  }

  const rtf = new Intl.RelativeTimeFormat("en-US", { numeric: "auto" });
  return rtf.format(future ? value : -value, unit);
}
