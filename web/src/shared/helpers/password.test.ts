import { describe, expect, it } from "vitest"

import {
  MAX_PASSWORD_BYTES,
  MIN_PASSWORD_LENGTH,
  newPasswordProblem,
  passwordByteLength,
  passwordLength,
} from "@/shared/helpers/password"

describe("password policy", () => {
  it("counts code points, not UTF-16 units, so the minimum matches the server's", () => {
    // Seven emoji: 14 to String.length, 7 to Python's len(), which is what the
    // gateway measures. A length count here would pass a password the server
    // then refuses as too short.
    const sevenEmoji = "😀😀😀😀😀😀😀"
    expect(sevenEmoji.length).toBe(14)
    expect(passwordLength(sevenEmoji)).toBe(7)
    expect(newPasswordProblem(sevenEmoji, sevenEmoji)).toBe(
      `At least ${MIN_PASSWORD_LENGTH} characters.`,
    )
  })

  it("counts bytes for the ceiling, because bcrypt's is bytes", () => {
    const accented = "é".repeat(40)
    expect(passwordLength(accented)).toBe(40)
    expect(passwordByteLength(accented)).toBe(80)
    expect(newPasswordProblem(accented, accented)).toBe(
      `At most ${MAX_PASSWORD_BYTES} bytes; accented characters count for more than one.`,
    )
  })

  it("treats an untouched form as unfinished rather than wrong", () => {
    expect(newPasswordProblem("", "")).toBeNull()
  })

  it("waits for the confirmation to be typed before complaining about it", () => {
    expect(newPasswordProblem("correct-horse", "")).toBeNull()
    expect(newPasswordProblem("correct-horse", "correct-hors")).toBe(
      "The two passwords do not match.",
    )
    expect(newPasswordProblem("correct-horse", "correct-horse")).toBeNull()
  })
})
