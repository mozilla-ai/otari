import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { CallerIdentity } from "@/client"
import { PasswordCard } from "@/features/account/PasswordCard"
import { API_ROOT } from "@/shared/api/client"
import { useOrganizationMembers } from "@/shared/api/organizations"
import { DeploymentProvider, useDeployment } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

// The three identities this card has to tell apart, and none of them is a fact
// about the deployment: which form applies is `has_password` plus whether there
// is an address, exactly as `PUT /v1/auth/password` branches on them.
const OPERATOR_UNCLAIMED: CallerIdentity = {
  user_id: "33333333-3333-3333-3333-333333333333",
  email: null,
  full_name: "Operator",
  has_password: false,
}
// Signed in through GitHub or Google, or added to the roster and never given a
// password: an address, and nothing to prove (mozilla-ai/otari-ai#2099).
const PASSWORDLESS: CallerIdentity = {
  user_id: "44444444-4444-4444-4444-444444444444",
  email: "ada@example.com",
  full_name: "Ada Lovelace",
  has_password: false,
}
const WITH_PASSWORD: CallerIdentity = { ...PASSWORDLESS, has_password: true }

const CLAIMED = {
  email: "operator@example.com",
  master_key_sign_in_retired: true,
}
const CHANGED = {
  email: "ada@example.com",
  master_key_sign_in_retired: false,
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/**
 * Stubs the two calls the card makes, at the transport rather than at the
 * hooks, so the real query keys and the cache seeding both run. The context is
 * served from a variable the PUT rewrites, which is how a test can tell a card
 * that re-read its own shape from one that kept a mode of its own.
 */
function mockApi(caller: CallerIdentity, put: unknown = CHANGED, status = 200) {
  let current = caller
  const fetchMock = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith("/organizations/me")) {
        return jsonResponse(organizationContext({ caller: current }))
      }
      if (url.endsWith("/auth/password")) {
        if (status === 200) {
          const body = JSON.parse(String(init?.body)) as { email?: string }
          current = {
            ...current,
            email: body.email ?? current.email,
            has_password: true,
          }
        }
        return jsonResponse(put, status)
      }
      return jsonResponse({ count: 0, data: [] })
    })
  return fetchMock
}

// Reports what the sign-in screen would offer after a claim. A span rather than
// an `output`, whose implicit role is `status` and would be a second match for
// the card's own saved line.
function SignInMethodsProbe() {
  return <span>offers {useDeployment().sign_in_methods.join(",")}</span>
}

function renderCard(
  signInMethods: ("master_key" | "password")[] = ["password"],
) {
  return render(
    <AppProviders>
      <DeploymentProvider value={bootstrap({ sign_in_methods: signInMethods })}>
        <PasswordCard />
        <SignInMethodsProbe />
      </DeploymentProvider>
    </AppProviders>,
  )
}

/** Opens the dialog from the card's own action and returns the frame. */
async function openDialog(
  user: ReturnType<typeof userEvent.setup>,
  label: string,
) {
  await user.click(await screen.findByRole("button", { name: label }))
  return await screen.findByRole("dialog")
}

async function fillNewPassword(
  user: ReturnType<typeof userEvent.setup>,
  dialog: HTMLElement,
  password: string,
  confirm = password,
) {
  await user.type(within(dialog).getByLabelText("New password"), password)
  await user.type(
    within(dialog).getByLabelText("Confirm new password"),
    confirm,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("PasswordCard for an identity that holds no password", () => {
  // The bug: `sign_in_methods` is the deployment's answer, so a claimed
  // deployment showed everybody the change form. Somebody who signed in with
  // GitHub or Google has no current password to type into it and no other way
  // through, which is the lock-out this card now cannot produce.
  it("asks for no current password, and for no address it already holds", async () => {
    mockApi(PASSWORDLESS)
    const user = userEvent.setup()
    renderCard(["password"])

    const dialog = await openDialog(user, "Set a password")

    expect(within(dialog).getByLabelText("New password")).toBeInTheDocument()
    expect(within(dialog).queryByLabelText("Current password")).toBeNull()
    expect(within(dialog).queryByLabelText("Email")).toBeNull()
  })

  it("sends the new password alone", async () => {
    const fetchMock = mockApi(PASSWORDLESS)
    const user = userEvent.setup()
    renderCard(["password"])

    const dialog = await openDialog(user, "Set a password")
    await fillNewPassword(user, dialog, "a-real-password")
    await user.click(
      within(dialog).getByRole("button", { name: "Set a password" }),
    )

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).endsWith("/auth/password"),
        ),
      ).toBe(true),
    )
    const [url, init] = fetchMock.mock.calls.find(([candidate]) =>
      String(candidate).endsWith("/auth/password"),
    ) as [string, RequestInit]
    expect(url).toBe(`${API_ROOT}/auth/password`)
    expect(init.method).toBe("PUT")
    // Neither `current_password` nor `email`: there is no old password to
    // prove, and resending the address the identity already holds is the one
    // shape `EmailChangeNotSupportedError` is about.
    expect(init.body).toBe(JSON.stringify({ new_password: "a-real-password" }))
  })

  it("becomes the change reading once the password lands, without a reload", async () => {
    mockApi(PASSWORDLESS)
    const user = userEvent.setup()
    renderCard(["password"])

    const dialog = await openDialog(user, "Set a password")
    await fillNewPassword(user, dialog, "a-real-password")
    await user.click(
      within(dialog).getByRole("button", { name: "Set a password" }),
    )

    // The card re-reads `caller.has_password`, which `useSetPassword` seats
    // back onto the membership context from the response.
    expect(
      await screen.findByRole("button", { name: "Change password" }),
    ).toBeInTheDocument()
    expect(screen.queryByRole("dialog")).toBeNull()
  })
})

describe("PasswordCard on an unclaimed deployment", () => {
  it("asks for an address and a password, and for no current one", async () => {
    mockApi(OPERATOR_UNCLAIMED, CLAIMED)
    const user = userEvent.setup()
    renderCard(["master_key"])

    const dialog = await openDialog(user, "Claim this deployment")

    expect(within(dialog).getByLabelText("Email")).toBeInTheDocument()
    expect(within(dialog).getByLabelText("New password")).toBeInTheDocument()
    expect(within(dialog).queryByLabelText("Current password")).toBeNull()
  })

  it("claims the deployment and stops the sign-in screen offering the master key", async () => {
    const fetchMock = mockApi(OPERATOR_UNCLAIMED, CLAIMED)
    const user = userEvent.setup()
    renderCard(["master_key"])

    const dialog = await openDialog(user, "Claim this deployment")
    await user.type(
      within(dialog).getByLabelText("Email"),
      "operator@example.com",
    )
    await fillNewPassword(user, dialog, "a-real-password")
    await user.click(
      within(dialog).getByRole("button", { name: "Claim this deployment" }),
    )

    expect(
      await screen.findByText(/master key no longer signs in/i),
    ).toBeInTheDocument()
    const put = fetchMock.mock.calls.find(([candidate]) =>
      String(candidate).endsWith("/auth/password"),
    ) as [string, RequestInit]
    expect(put[1].body).toBe(
      JSON.stringify({
        new_password: "a-real-password",
        email: "operator@example.com",
      }),
    )
    // The bootstrap is read once per load and still says `master_key`, so this
    // proves the tab believes the response over the stale context.
    expect(screen.getByText("offers password")).toBeInTheDocument()
  })
})

describe("PasswordCard for an identity that holds one", () => {
  it("requires the current password and sends it", async () => {
    const fetchMock = mockApi(WITH_PASSWORD)
    const user = userEvent.setup()
    renderCard()

    const dialog = await openDialog(user, "Change password")
    expect(within(dialog).queryByLabelText("Email")).toBeNull()

    await user.type(
      within(dialog).getByLabelText("Current password"),
      "old-password",
    )
    await fillNewPassword(user, dialog, "new-password")
    await user.click(
      within(dialog).getByRole("button", { name: "Change password" }),
    )

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).endsWith("/auth/password"),
        ),
      ).toBe(true),
    )
    const put = fetchMock.mock.calls.find(([candidate]) =>
      String(candidate).endsWith("/auth/password"),
    ) as [string, RequestInit]
    expect(put[1].body).toBe(
      JSON.stringify({
        new_password: "new-password",
        current_password: "old-password",
      }),
    )
  })

  it("refuses a new password that is the current one, before asking the gateway", async () => {
    const fetchMock = mockApi(WITH_PASSWORD)
    const user = userEvent.setup()
    renderCard()

    const dialog = await openDialog(user, "Change password")
    await user.type(
      within(dialog).getByLabelText("Current password"),
      "same-password",
    )
    await fillNewPassword(user, dialog, "same-password")

    expect(
      within(dialog).getByText(/cannot be the one you already use/i),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "Change password" }),
    ).toBeDisabled()
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).endsWith("/auth/password"),
      ),
    ).toBe(false)
  })

  it("renders the gateway's own refusal rather than a guess, and keeps the form", async () => {
    mockApi(WITH_PASSWORD, { detail: "Current password is incorrect" }, 400)
    const user = userEvent.setup()
    renderCard()

    const dialog = await openDialog(user, "Change password")
    await user.type(
      within(dialog).getByLabelText("Current password"),
      "wrong-password",
    )
    await fillNewPassword(user, dialog, "a-real-password")
    await user.click(
      within(dialog).getByRole("button", { name: "Change password" }),
    )

    expect(
      await within(dialog).findByText("Current password is incorrect"),
    ).toBeInTheDocument()
    // A 400 and not a 401, so the session survives a mistyped field: the frame
    // is still open with what was filled in.
    expect(within(dialog).getByLabelText("Current password")).toHaveValue(
      "wrong-password",
    )
  })

  it("reports the save on the card once the dialog has closed", async () => {
    mockApi(WITH_PASSWORD)
    const user = userEvent.setup()
    renderCard()

    const dialog = await openDialog(user, "Change password")
    await user.type(
      within(dialog).getByLabelText("Current password"),
      "old-password",
    )
    await fillNewPassword(user, dialog, "new-password")
    await user.click(
      within(dialog).getByRole("button", { name: "Change password" }),
    )

    const status = await screen.findByRole("status")
    expect(status).toHaveTextContent(/Your other sessions have ended/i)
  })
})

describe("PasswordCard policy checks", () => {
  async function typeAndOpen(password: string, confirm = password) {
    mockApi(PASSWORDLESS)
    const user = userEvent.setup()
    renderCard()
    const dialog = await openDialog(user, "Set a password")
    await fillNewPassword(user, dialog, password, confirm)
    return dialog
  }

  it("holds back a password under the minimum length", async () => {
    const dialog = await typeAndOpen("short")

    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      "At least 8 characters.",
    )
    expect(
      within(dialog).getByRole("button", { name: "Set a password" }),
    ).toBeDisabled()
  })

  it("counts the ceiling in bytes, not characters", async () => {
    // 40 characters, and 80 bytes in UTF-8: under any character count bcrypt
    // would be described by, over the 72 bytes it actually hashes.
    const dialog = await typeAndOpen("é".repeat(40))

    // Targeted at the alert, not the text: the field's own description names
    // the same ceiling, so a loose match would pass without the check firing.
    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      "At most 72 bytes; accented characters count for more than one.",
    )
  })

  it("counts the minimum in code points, as the gateway does", async () => {
    // Seven emoji: 14 to JavaScript's `String.length` and 7 to Python's `len`,
    // so a UTF-16 count would enable Save and hand the gateway a password its
    // own eight-character minimum refuses.
    const dialog = await typeAndOpen("🔒".repeat(7))

    expect(within(dialog).getByRole("alert")).toHaveTextContent(
      "At least 8 characters.",
    )
  })

  it("holds back a confirmation that does not match", async () => {
    const dialog = await typeAndOpen("a-real-password", "a-real-passwor")

    expect(
      within(dialog).getByText("The two passwords do not match."),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "Set a password" }),
    ).toBeDisabled()
  })
})

describe("PasswordCard and the identity it cannot read", () => {
  it("offers no form rather than a guessed one when the context fails", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
      String(input).endsWith("/organizations/me")
        ? jsonResponse({ detail: "boom" }, 403)
        : jsonResponse({ count: 0, data: [] }),
    )
    renderCard()

    expect(await screen.findByText(/could not be read/i)).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /password/i })).toBeNull()
  })
})

describe("PasswordCard and the member roster", () => {
  it("refreshes the roster, whose address a claim is what fills in", async () => {
    // The operator's roster row carries `email: null` until the claim writes
    // one, and `useOrganizationMembers` caches for a minute, so without an
    // invalidation the Members page would show the pre-claim row for the rest
    // of that minute.
    let memberFetches = 0
    let caller = OPERATOR_UNCLAIMED
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes(`${API_ROOT}/organizations/me/members`)) {
        memberFetches += 1
        return jsonResponse({ count: 0, data: [] })
      }
      if (url.endsWith("/organizations/me")) {
        return jsonResponse(organizationContext({ caller }))
      }
      caller = { ...caller, email: CLAIMED.email, has_password: true }
      return jsonResponse(CLAIMED)
    })
    const user = userEvent.setup()

    // Mounted beside the card, so the roster is an active query the way it is
    // on the Members page. An inactive one would not refetch on invalidation
    // and the test would pass without proving anything.
    function Harness() {
      useOrganizationMembers()
      return <PasswordCard />
    }
    render(
      <AppProviders>
        <DeploymentProvider
          value={bootstrap({ sign_in_methods: ["master_key"] })}
        >
          <Harness />
        </DeploymentProvider>
      </AppProviders>,
    )
    await waitFor(() => expect(memberFetches).toBe(1))

    const dialog = await openDialog(user, "Claim this deployment")
    await user.type(
      within(dialog).getByLabelText("Email"),
      "operator@example.com",
    )
    await fillNewPassword(user, dialog, "a-real-password")
    await user.click(
      within(dialog).getByRole("button", { name: "Claim this deployment" }),
    )

    await waitFor(() => expect(memberFetches).toBe(2))
  })
})
