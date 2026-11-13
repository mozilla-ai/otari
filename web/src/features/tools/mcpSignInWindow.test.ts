import { expect, it } from "vitest"

import { isSignInWindow } from "@/features/tools/mcpSignInWindow"

it("does not take a state named like an object's own machinery for a popup's", () => {
  expect(isSignInWindow("constructor")).toBe(false)
})
