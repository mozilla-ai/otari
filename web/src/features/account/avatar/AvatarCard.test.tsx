import { act, render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { AvatarCard } from "./AvatarCard"
import { readAvatarChoice } from "./avatarChoice"
import { defaultFaceFor } from "./faces"

const ADA = "33333333-3333-3333-3333-333333333333"

vi.mock("@/shared/api/auth", () => ({
  useSessionCaller: () => ({ user_id: ADA }),
}))

// jsdom has no Web Animations; the bloom, ripple and pulse are decoration.
beforeEach(() => {
  Element.prototype.animate = vi.fn() as unknown as Element["animate"]
})

afterEach(() => {
  window.localStorage.clear()
})

function cardFace(): number {
  const svg = screen
    .getByRole("button", { name: /Change avatar/ })
    .parentElement?.querySelector("[data-face]")
  return Number(svg?.getAttribute("data-face"))
}

// Opens the picker and moves the frame one face to the right, which is a
// different face by construction: one period never holds a face twice.
async function openAndMove(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: /Change avatar/ }))
  const dialog = within(screen.getByRole("dialog"))
  const field = dialog.getByRole("application")
  act(() => field.focus())
  await user.keyboard("{ArrowRight}")
  return dialog
}

describe("AvatarCard", () => {
  it("starts on the face the user id gives", () => {
    render(<AvatarCard />)
    expect(cardFace()).toBe(defaultFaceFor(ADA))
  })

  it("keeps the saved face when the picker is cancelled", async () => {
    const user = userEvent.setup()
    render(<AvatarCard />)

    const dialog = await openAndMove(user)
    await user.click(dialog.getByRole("button", { name: "Cancel" }))

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(readAvatarChoice(ADA)).toBeNull()
    expect(cardFace()).toBe(defaultFaceFor(ADA))
  })

  it("saves the face in the frame", async () => {
    const user = userEvent.setup()
    render(<AvatarCard />)

    const dialog = await openAndMove(user)
    await user.click(dialog.getByRole("button", { name: "Save" }))

    const saved = readAvatarChoice(ADA)
    expect(saved).not.toBeNull()
    expect(saved).not.toBe(defaultFaceFor(ADA))
    expect(cardFace()).toBe(saved)
  })
})
