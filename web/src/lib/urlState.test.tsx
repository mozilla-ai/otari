import { renderHook } from "@testing-library/react";
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
