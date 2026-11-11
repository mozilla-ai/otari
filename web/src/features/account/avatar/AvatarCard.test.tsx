import { act, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { CallerIdentity } from "@/client"
import * as apiClient from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

import { AvatarCard } from "./AvatarCard"
import { defaultFaceFor } from "./faces"

const ADA = "33333333-3333-3333-3333-333333333333"

/**
 * Spies on the transport rather than the hooks, so the real cache seeding runs:
 * `written` is every face the card sent, and what the context reports after.
 */
function mockApi({
  stored = null,
  failWrite = false,
}: {
  stored?: number | null
  failWrite?: boolean
} = {}) {
  const written: Array<number | null> = []
  let caller: CallerIdentity = {
    user_id: ADA,
    email: "ada@example.com",
    full_name: "Ada",
    has_password: true,
    claims_deployment: false,
    avatar_face: stored,
  }
  vi.spyOn(apiClient, "apiFetch").mockImplementation(
    async (path: string, init?: RequestInit) => {
      if (path === "/organizations/me") {
        return organizationContext({ caller }) as never
      }
      if (path === "/auth/profile/avatar") {
        const { face } = JSON.parse(String(init?.body)) as {
          face: number | null
        }
        written.push(face)
        if (failWrite) throw new Error("The avatar could not be saved.")
        caller = { ...caller, avatar_face: face }
        return caller as never
      }
      throw new Error(`unexpected request: ${path}`)
    },
  )
  return written
}

// jsdom has no Web Animations; the bloom, ripple and pulse are decoration.
beforeEach(() => {
  Element.prototype.animate = vi.fn() as unknown as Element["animate"]
})

afterEach(() => {
  vi.restoreAllMocks()
})

function renderCard() {
  return render(
    <AppProviders>
      <AvatarCard />
    </AppProviders>,
  )
}

async function cardFace(): Promise<number> {
  const button = await screen.findByRole("button", { name: /currently/ })
  const svg = button.parentElement?.querySelector("[data-face]")
  return Number(svg?.getAttribute("data-face"))
}

// Opens the picker and moves the frame one face to the right, which is a
// different face by construction: one period never holds a face twice.
async function openAndMove(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("button", { name: /currently/ }))
  const dialog = within(screen.getByRole("dialog"))
  const field = dialog.getByRole("application")
  act(() => field.focus())
  await user.keyboard("{ArrowRight}")
  return dialog
}

describe("AvatarCard", () => {
  it("starts on the face the user id gives", async () => {
    mockApi()
    renderCard()
    expect(await cardFace()).toBe(defaultFaceFor(ADA))
  })

  it("starts on the face the profile holds", async () => {
    mockApi({ stored: 1234 })
    renderCard()
    expect(await cardFace()).toBe(1234)
  })

  it("keeps the saved face when the picker is cancelled", async () => {
    const written = mockApi()
    const user = userEvent.setup()
    renderCard()

    const dialog = await openAndMove(user)
    await user.click(dialog.getByRole("button", { name: "Cancel" }))

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    expect(written).toEqual([])
    expect(await cardFace()).toBe(defaultFaceFor(ADA))
  })

  it("saves the face in the frame to the profile", async () => {
    const written = mockApi()
    const user = userEvent.setup()
    renderCard()

    const dialog = await openAndMove(user)
    await user.click(dialog.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(written).toHaveLength(1))
    const [saved] = written
    expect(saved).not.toBe(defaultFaceFor(ADA))
    expect(await cardFace()).toBe(saved)
  })

  it("says so and puts the old face back when the save fails", async () => {
    const written = mockApi({ stored: 1234, failWrite: true })
    const user = userEvent.setup()
    renderCard()

    const dialog = await openAndMove(user)
    await user.click(dialog.getByRole("button", { name: "Save" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The avatar could not be saved.",
    )
    expect(written).toHaveLength(1)
    expect(await cardFace()).toBe(1234)
  })
})
