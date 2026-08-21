import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { PasswordCard } from "@/features/account/PasswordCard"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

// Which of the two forms this card renders comes from the bootstrap, so every
// render goes through a DeploymentProvider. `master_key` is the unclaimed
// deployment a fixture describes by default (the claim form); `password` is one
// an operator has already claimed (the change form).
function renderCard(signInMethods: ("master_key" | "password")[]) {
  return render(
    <AppProviders>
      <DeploymentProvider value={bootstrap({ sign_in_methods: signInMethods })}>
        <PasswordCard />
      </DeploymentProvider>
    </AppProviders>,
  )
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function mockPut(body: unknown, status = 200) {
  return vi
    .spyOn(globalThis, "fetch")
    .mockResolvedValue(jsonResponse(body, status))
}

const CLAIMED = {
  email: "operator@example.com",
  master_key_sign_in_retired: true,
}

describe("PasswordCard on an unclaimed deployment", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("asks for an address and a password, and for no current one", () => {
    renderCard(["master_key"])

    expect(screen.getByLabelText("Email")).toBeInTheDocument()
    expect(screen.getByLabelText("New password")).toBeInTheDocument()
    expect(screen.queryByLabelText("Current password")).not.toBeInTheDocument()
  })

  it("claims the deployment and says the master key no longer signs in", async () => {
    const fetchMock = mockPut(CLAIMED)
    const user = userEvent.setup()
    renderCard(["master_key"])

    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("New password"), "a-real-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "a-real-password",
    )
    await user.click(screen.getByRole("button", { name: "Set password" }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe("/v1/auth/password")
    expect(init?.method).toBe("PUT")
    // No `current_password`: there is none to prove, and sending null would be
    // a different request from the one the claim documents.
    expect(init?.body).toBe(
      JSON.stringify({
        email: "operator@example.com",
        new_password: "a-real-password",
      }),
    )
    expect(
      await screen.findByText(/master key no longer signs in/i),
    ).toBeInTheDocument()
  })

  it("becomes the change form once the claim succeeds, without a reload", async () => {
    mockPut(CLAIMED)
    const user = userEvent.setup()
    renderCard(["master_key"])

    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("New password"), "a-real-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "a-real-password",
    )
    await user.click(screen.getByRole("button", { name: "Set password" }))

    // The bootstrap is read once per page load and still says `master_key`, so
    // this proves the card believes the response over the stale context.
    expect(await screen.findByLabelText("Current password")).toBeInTheDocument()
    expect(screen.queryByLabelText("Email")).not.toBeInTheDocument()
  })
})

describe("PasswordCard on a claimed deployment", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("requires the current password and sends it", async () => {
    const fetchMock = mockPut({
      email: "operator@example.com",
      master_key_sign_in_retired: true,
    })
    const user = userEvent.setup()
    renderCard(["password"])

    expect(screen.queryByLabelText("Email")).not.toBeInTheDocument()

    await user.type(screen.getByLabelText("Current password"), "old-password")
    await user.type(screen.getByLabelText("New password"), "new-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "new-password",
    )
    await user.click(screen.getByRole("button", { name: "Change password" }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(fetchMock.mock.calls[0][1]?.body).toBe(
      JSON.stringify({
        current_password: "old-password",
        new_password: "new-password",
      }),
    )
  })

  it("refuses a new password that is the current one, before asking the gateway", async () => {
    const fetchMock = mockPut(CLAIMED)
    const user = userEvent.setup()
    renderCard(["password"])

    await user.type(screen.getByLabelText("Current password"), "same-password")
    await user.type(screen.getByLabelText("New password"), "same-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "same-password",
    )

    expect(
      screen.getByText(/cannot be the one you already use/i),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Change password" }),
    ).toBeDisabled()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("renders the gateway's own refusal rather than a guess", async () => {
    mockPut({ detail: "Current password is incorrect" }, 400)
    const user = userEvent.setup()
    renderCard(["password"])

    await user.type(screen.getByLabelText("Current password"), "wrong-password")
    await user.type(screen.getByLabelText("New password"), "a-real-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "a-real-password",
    )
    await user.click(screen.getByRole("button", { name: "Change password" }))

    expect(
      await screen.findByText("Current password is incorrect"),
    ).toBeInTheDocument()
    // A 400 and not a 401, so the session survives a mistyped field: the card
    // is still on screen with the form the operator filled in.
    expect(screen.getByLabelText("Current password")).toBeInTheDocument()
  })
})

describe("PasswordCard policy checks", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("holds back a password under the minimum length", async () => {
    const fetchMock = mockPut(CLAIMED)
    const user = userEvent.setup()
    renderCard(["master_key"])

    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("New password"), "short")
    await user.type(screen.getByLabelText("Confirm new password"), "short")

    expect(screen.getByText("At least 8 characters.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Set password" })).toBeDisabled()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("counts the ceiling in bytes, not characters", async () => {
    const fetchMock = mockPut(CLAIMED)
    const user = userEvent.setup()
    renderCard(["master_key"])

    // 40 characters, and 80 bytes in UTF-8: under any character count bcrypt
    // would be described by, over the 72 bytes it actually hashes.
    const accented = "é".repeat(40)
    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("New password"), accented)
    await user.type(screen.getByLabelText("Confirm new password"), accented)

    // Targeted at the alert, not the text: the field's own description names
    // the same ceiling, so a loose match would pass without the check firing.
    expect(screen.getByRole("alert")).toHaveTextContent(
      "At most 72 bytes; accented characters count for more than one.",
    )
    expect(screen.getByRole("button", { name: "Set password" })).toBeDisabled()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("holds back a confirmation that does not match", async () => {
    const fetchMock = mockPut(CLAIMED)
    const user = userEvent.setup()
    renderCard(["master_key"])

    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("New password"), "a-real-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "a-real-passwor",
    )

    expect(
      screen.getByText("The two passwords do not match."),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Set password" })).toBeDisabled()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
