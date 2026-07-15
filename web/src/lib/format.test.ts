import { describe, expect, it } from "vitest";

import { formatCost, formatNumber, formatRelative } from "@/lib/format";

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
