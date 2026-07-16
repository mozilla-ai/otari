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
    return `${Math.round(value / 1000)}K`;
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
