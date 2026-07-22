import { describe, expect, it } from "vitest";

import {
  deltaFraction,
  formatCost,
  formatNumber,
  formatPct,
  formatRelative,
  formatReleaseDate,
  formatTokens,
  formatUsd,
} from "@/lib/format";

describe("formatNumber", () => {
  it("groups thousands and handles nullish values", () => {
    expect(formatNumber(1234567)).toBe("1,234,567");
    expect(formatNumber(null)).toBe("0");
    expect(formatNumber(undefined)).toBe("0");
  });
});

describe("formatUsd", () => {
  it("renders compact cents for aggregate tiles", () => {
    expect(formatUsd(0)).toBe("$0.00");
    expect(formatUsd(1240.5)).toBe("$1,240.50");
    // No sub-cent precision here (unlike formatCost): tiles stay readable.
    expect(formatUsd(0.004)).toBe("$0.00");
  });
});

describe("formatTokens", () => {
  it("compacts at the k and M boundaries", () => {
    expect(formatTokens(512)).toBe("512");
    expect(formatTokens(999)).toBe("999");
    expect(formatTokens(1000)).toBe("1.0k");
    expect(formatTokens(84_200)).toBe("84.2k");
    expect(formatTokens(999_999)).toBe("1000.0k");
    expect(formatTokens(1_000_000)).toBe("1.0M");
    expect(formatTokens(12_400_000)).toBe("12.4M");
  });
});

describe("formatPct", () => {
  it("renders one decimal place", () => {
    expect(formatPct(0)).toBe("0.0%");
    expect(formatPct(0.021)).toBe("2.1%");
    expect(formatPct(1.25)).toBe("125.0%");
  });
});

describe("deltaFraction", () => {
  it("guards divide-by-zero and the unknown-previous case", () => {
    // No comparable baseline -> null (hides the delta), never Infinity/NaN.
    expect(deltaFraction(10, 0)).toBeNull();
    expect(deltaFraction(10, undefined)).toBeNull();
  });
  it("computes signed period-over-period change", () => {
    expect(deltaFraction(150, 100)).toBeCloseTo(0.5);
    expect(deltaFraction(80, 100)).toBeCloseTo(-0.2);
    expect(deltaFraction(100, 100)).toBe(0);
  });
});

describe("formatCost", () => {
  it("uses extra precision for sub-cent amounts", () => {
    expect(formatCost(0.0001)).toBe("$0.0001");
  });

  it("uses two decimals for normal amounts", () => {
    expect(formatCost(12.5)).toBe("$12.50");
    expect(formatCost(null)).toBe("$0.00");
  });
});

describe("formatReleaseDate", () => {
  it("renders a compact month and year without timezone drift", () => {
    expect(formatReleaseDate("2024-05-13")).toBe("May 2024");
    expect(formatReleaseDate("2025-01-01")).toBe("Jan 2025");
    expect(formatReleaseDate("2023-12")).toBe("Dec 2023");
  });

  it("falls back gracefully for missing or unparseable values", () => {
    expect(formatReleaseDate(null)).toBe("—");
    expect(formatReleaseDate(undefined)).toBe("—");
    expect(formatReleaseDate("someday")).toBe("someday");
    expect(formatReleaseDate("2024-13")).toBe("2024");
  });
});

describe("formatRelative", () => {
  const now = Date.parse("2026-01-01T12:00:00Z");

  it("describes past timestamps", () => {
    expect(formatRelative("2026-01-01T11:59:30Z", now)).toContain("seconds ago");
    expect(formatRelative("2026-01-01T10:00:00Z", now)).toContain("hours ago");
  });

  it("returns 'never' for missing timestamps", () => {
    expect(formatRelative(null, now)).toBe("never");
  });
});
