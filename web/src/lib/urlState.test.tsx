import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { useUrlState } from "./urlState";

const DEFAULTS = { page: "0", size: "50", status: "" } as const;

function wrapperFor(url: string) {
  return ({ children }: { children: ReactNode }) => <MemoryRouter initialEntries={[url]}>{children}</MemoryRouter>;
}

describe("useUrlState.getNumber", () => {
  it("reads a numeric param", () => {
    const { result } = renderHook(() => useUrlState(DEFAULTS), { wrapper: wrapperFor("/?size=250") });
    expect(result.current.getNumber("size")).toBe(250);
  });

  it("falls back to the key's default when the param is absent", () => {
    const { result } = renderHook(() => useUrlState(DEFAULTS), { wrapper: wrapperFor("/") });
    expect(result.current.getNumber("size")).toBe(50);
    expect(result.current.getNumber("page")).toBe(0);
  });

  it("falls back to the default (not 0) when the param is present but non-numeric", () => {
    // A hand-edited `?size=abc` must not become pageSize=0 (which would send limit=0 → 422).
    const { result } = renderHook(() => useUrlState(DEFAULTS), { wrapper: wrapperFor("/?size=abc") });
    expect(result.current.getNumber("size")).toBe(50);
  });
});

describe("useUrlState.get", () => {
  it("returns the param value or the default", () => {
    const { result } = renderHook(() => useUrlState(DEFAULTS), { wrapper: wrapperFor("/?status=error") });
    expect(result.current.get("status")).toBe("error");
    expect(result.current.get("size")).toBe("50");
  });
});

const MULTI_DEFAULTS = { model: "", user_id: "" } as const;

describe("useUrlState.getAll", () => {
  it("reads every value of a repeated param, in order", () => {
    const { result } = renderHook(() => useUrlState(MULTI_DEFAULTS), {
      wrapper: wrapperFor("/?model=gpt-4o&model=claude-sonnet-5"),
    });
    expect(result.current.getAll("model")).toEqual(["gpt-4o", "claude-sonnet-5"]);
  });

  it("reads a single value as a one-element set", () => {
    const { result } = renderHook(() => useUrlState(MULTI_DEFAULTS), { wrapper: wrapperFor("/?model=gpt-4o") });
    expect(result.current.getAll("model")).toEqual(["gpt-4o"]);
  });

  it("treats an absent, blank, or whitespace param as no filter", () => {
    // `?model=` is what a cleared filter can leave behind; it must not become a
    // filter on the empty string, which would match nothing and read as "no rows".
    const { result } = renderHook(() => useUrlState(MULTI_DEFAULTS), { wrapper: wrapperFor("/?model=") });
    expect(result.current.getAll("model")).toEqual([]);
    expect(result.current.getAll("user_id")).toEqual([]);

    // A hand-edited whitespace value is the same thing, and trimming it keeps a chip
    // from rendering with a blank label.
    const spaced = renderHook(() => useUrlState(MULTI_DEFAULTS), { wrapper: wrapperFor("/?model=%20&model=gpt-4o") });
    expect(spaced.result.current.getAll("model")).toEqual(["gpt-4o"]);
  });

  it("applies a non-empty default only when the key is absent", () => {
    // Present-but-blank means the operator cleared the filter, so the default must
    // not resurrect it. Matches how `get` reads the same URL.
    const withDefault = { source: "gateway" } as const;
    const absent = renderHook(() => useUrlState(withDefault), { wrapper: wrapperFor("/") });
    expect(absent.result.current.getAll("source")).toEqual(["gateway"]);

    const cleared = renderHook(() => useUrlState(withDefault), { wrapper: wrapperFor("/?source=") });
    expect(cleared.result.current.getAll("source")).toEqual([]);
  });
});

describe("useUrlState.patch with arrays", () => {
  it("writes one param per value and drops the key when the set is empty", () => {
    const { result } = renderHook(() => useUrlState(MULTI_DEFAULTS), {
      wrapper: wrapperFor("/?model=gpt-4o&user_id=alice"),
    });

    act(() => result.current.patch({ model: ["gpt-4o", "claude-sonnet-5"] }));
    expect(result.current.getAll("model")).toEqual(["gpt-4o", "claude-sonnet-5"]);
    // The untouched key is left alone.
    expect(result.current.getAll("user_id")).toEqual(["alice"]);

    // A shorter set replaces the old one wholesale rather than appending, so a
    // removed value actually disappears.
    act(() => result.current.patch({ model: ["claude-sonnet-5"] }));
    expect(result.current.getAll("model")).toEqual(["claude-sonnet-5"]);

    act(() => result.current.patch({ model: [] }));
    expect(result.current.getAll("model")).toEqual([]);
  });
});
