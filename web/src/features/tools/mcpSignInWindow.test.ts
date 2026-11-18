import { afterEach, expect, it, vi } from "vitest"

import {
  isSignInWindow,
  openSignInWindow,
} from "@/features/tools/mcpSignInWindow"

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

it("does not take a state named like an object's own machinery for a popup's", () => {
  expect(isSignInWindow("constructor")).toBe(false)
})

it("opens no popup where no channel could bring its result back", () => {
  vi.stubGlobal("BroadcastChannel", undefined)
  const open = vi.spyOn(window, "open")

  expect(openSignInWindow()).toBeUndefined()
  expect(open).not.toHaveBeenCalled()
})

it("treats a window a blocker already closed as blocked", () => {
  vi.spyOn(window, "open").mockReturnValue({ closed: true } as Window)

  expect(openSignInWindow()).toBeUndefined()
})
