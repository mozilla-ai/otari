import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { resolveSelectedIds, selectionCount, useTableSelection } from "./tableSelection";

describe("resolveSelectedIds", () => {
  it("expands the 'all' sentinel to the page's selectable keys", () => {
    expect(resolveSelectedIds("all", ["a", "b", "c"])).toEqual(["a", "b", "c"]);
  });

  it("keeps only selected keys, in page order", () => {
    expect(resolveSelectedIds(new Set(["c", "a"]), ["a", "b", "c"])).toEqual(["a", "c"]);
  });

  it("drops stale keys not on the current page", () => {
    expect(resolveSelectedIds(new Set(["a", "gone"]), ["a", "b"])).toEqual(["a"]);
  });
});

describe("selectionCount", () => {
  it("counts an explicit selection", () => {
    expect(selectionCount(new Set(["a", "b"]), ["a", "b", "c"])).toBe(2);
  });
  it("counts 'all' as every selectable key", () => {
    expect(selectionCount("all", ["a", "b", "c"])).toBe(3);
  });
});

describe("useTableSelection", () => {
  it("drops all-matching when the selection changes manually", () => {
    const { result } = renderHook(() => useTableSelection());

    act(() => result.current.enableAllMatching());
    expect(result.current.allMatching).toBe(true);

    act(() => result.current.onSelectionChange(new Set(["a"])));
    expect(result.current.allMatching).toBe(false);
    expect(result.current.selectedKeys).toEqual(new Set(["a"]));
  });

  it("clear resets both the selection and all-matching", () => {
    const { result } = renderHook(() => useTableSelection());
    act(() => result.current.onSelectionChange(new Set(["a", "b"])));
    act(() => result.current.enableAllMatching());
    act(() => result.current.clear());
    expect(result.current.allMatching).toBe(false);
    expect(result.current.selectedKeys).toEqual(new Set());
  });
});
