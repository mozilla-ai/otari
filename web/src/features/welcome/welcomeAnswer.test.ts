import { afterEach, expect, it, vi } from "vitest"

import {
  OWN_KEYS,
  readWelcomeAnswer,
  writeWelcomeAnswer,
} from "@/features/welcome/welcomeAnswer"

afterEach(() => {
  vi.restoreAllMocks()
})

it("keeps an answer storage refused for the rest of the tab", () => {
  const refused = () => {
    throw new DOMException("The quota has been exceeded.", "QuotaExceededError")
  }
  vi.spyOn(Storage.prototype, "setItem").mockImplementation(refused)
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(refused)

  writeWelcomeAnswer(OWN_KEYS)

  expect(readWelcomeAnswer()).toBe(OWN_KEYS)
})
