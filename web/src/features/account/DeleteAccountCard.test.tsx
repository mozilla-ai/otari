import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { CallerIdentity } from "@/client"
import { DeleteAccountCard } from "@/features/account/DeleteAccountCard"
import * as apiClient from "@/shared/api/client"
import { ApiError } from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

const logout = vi.fn(async () => {})
vi.mock("@/features/auth/AuthContext", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/features/auth/AuthContext")>()),
  useAuth: () => ({ logout }),
}))

const CALLER: CallerIdentity = {
  user_id: "33333333-3333-3333-3333-333333333333",
  email: "ada@example.com",
  full_name: "Ada Lovelace",
  has_password: true,
  claims_deployment: false,
}

function mockApi(
  caller: CallerIdentity,
  {
    refusal,
    organization = true,
  }: { refusal?: Error; organization?: boolean } = {},
) {
  const sent: unknown[] = []
  vi.spyOn(apiClient, "apiFetch").mockImplementation(
    async (path: string, init?: RequestInit) => {
      if (path === "/organizations/me") {
        if (!organization) {
          throw new ApiError(404, "No organization")
        }
        return organizationContext({ caller }) as never
      }
      if (path === "/auth/profile") {
        return caller as never
      }
      if (path === "/auth/account" && init?.method === "DELETE") {
        sent.push(JSON.parse(String(init.body)))
        if (refusal) {
          throw refusal
        }
        return undefined as never
      }
      throw new Error(`unexpected request: ${path}`)
    },
  )
  return sent
}

async function openDialog() {
  const user = userEvent.setup()
  render(
    <AppProviders>
      <DeleteAccountCard />
    </AppProviders>,
  )
  await user.click(
    await screen.findByRole("button", { name: "Delete account" }),
  )
  return user
}

describe("DeleteAccountCard", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("holds the deletion until the address and password are typed", async () => {
    mockApi(CALLER)
    const user = await openDialog()
    const dialog = await screen.findByRole("alertdialog")
    const confirm = () =>
      screen.getAllByRole("button", { name: "Delete account" }).at(-1)

    expect(dialog).toBeInTheDocument()
    expect(confirm()).toBeDisabled()
    await user.type(
      screen.getByLabelText(/Type ada@example.com to confirm/),
      "ada@example.com",
    )
    expect(confirm()).toBeDisabled()
    await user.type(screen.getByLabelText(/^Password/), "secret")
    expect(confirm()).toBeEnabled()
  })

  it("deletes the account and signs out", async () => {
    const sent = mockApi(CALLER)
    const user = await openDialog()
    await user.type(
      await screen.findByLabelText(/Type ada@example.com to confirm/),
      "ada@example.com",
    )
    await user.type(screen.getByLabelText(/^Password/), "secret")
    await user.click(
      screen.getAllByRole("button", { name: "Delete account" }).at(-1)!,
    )

    await waitFor(() => expect(logout).toHaveBeenCalledOnce())
    expect(sent).toEqual([
      { confirm_email: "ada@example.com", current_password: "secret" },
    ])
  })

  it("is offered to an account with no organization", async () => {
    const sent = mockApi(CALLER, { organization: false })
    const user = await openDialog()
    await user.type(
      await screen.findByLabelText(/Type ada@example.com to confirm/),
      "ada@example.com",
    )
    await user.type(screen.getByLabelText(/^Password/), "secret")
    await user.click(
      screen.getAllByRole("button", { name: "Delete account" }).at(-1)!,
    )

    await waitFor(() => expect(logout).toHaveBeenCalledOnce())
    expect(sent).toEqual([
      { confirm_email: "ada@example.com", current_password: "secret" },
    ])
  })

  it("asks no password of an account that has none", async () => {
    const sent = mockApi({ ...CALLER, has_password: false })
    const user = await openDialog()
    await user.type(
      await screen.findByLabelText(/Type ada@example.com to confirm/),
      "ada@example.com",
    )

    expect(screen.queryByLabelText(/^Password/)).not.toBeInTheDocument()
    await user.click(
      screen.getAllByRole("button", { name: "Delete account" }).at(-1)!,
    )
    await waitFor(() =>
      expect(sent).toEqual([{ confirm_email: "ada@example.com" }]),
    )
  })

  it("shows a refusal and stays signed in", async () => {
    mockApi(CALLER, {
      refusal: new ApiError(
        409,
        "Make another member an owner of Acme before deleting your account",
      ),
    })
    const user = await openDialog()
    await user.type(
      await screen.findByLabelText(/Type ada@example.com to confirm/),
      "ada@example.com",
    )
    await user.type(screen.getByLabelText(/^Password/), "secret")
    await user.click(
      screen.getAllByRole("button", { name: "Delete account" }).at(-1)!,
    )

    expect(
      await screen.findByText(/Make another member an owner of Acme/),
    ).toBeInTheDocument()
    expect(logout).not.toHaveBeenCalled()
  })
})
