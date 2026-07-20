import { describe, expect, it } from "vitest";

import { formatCost, formatNumber, formatRelative, formatReleaseDate } from "@/lib/format";

describe("formatNumber", () => {
  it("groups thousands and handles nullish values", () => {
    expect(formatNumber(1234567)).toBe("1,234,567");
    expect(formatNumber(null)).toBe("0");
    expect(formatNumber(undefined)).toBe("0");
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
