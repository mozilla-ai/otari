import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext } from "@/client"
import { PasswordCard } from "@/features/account/PasswordCard"
import { useOrganizationMembers } from "@/shared/api/organizations"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
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

const CLAIMED = {
  email: "operator@example.com",
  master_key_sign_in_retired: true,
}

// The card reads GET /v1/organizations/me unconditionally, the way the account
// menu already does, so every test stubs it rather than leaving it failing in
// the background of a test that is not about it. `caller: null` (the default
// below, matching `organizationContext`'s own default) models the identity a
// first boot leaves behind, with no address of its own; a test about the
// migrated-identity gap overrides `caller.email`.
function mockRequests({
  caller = organizationContext().caller,
  put = CLAIMED,
  putStatus = 200,
}: {
  caller?: OrganizationContext["caller"]
  put?: unknown
  putStatus?: number
} = {}) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url === "/v1/organizations/me") {
      return jsonResponse(organizationContext({ caller }))
    }
    if (url === "/v1/auth/password") {
      return jsonResponse(put, putStatus)
    }
    return jsonResponse({ count: 0, data: [] })
  })
}

/** The call this card made to `PUT /v1/auth/password`, if any. */
function putCall(fetchMock: ReturnType<typeof vi.spyOn>) {
  return fetchMock.mock.calls.find(
    ([url]: [unknown]) => String(url) === "/v1/auth/password",
  )
}

/**
 * Holds `GET /v1/organizations/me` pending forever, for the loading-window
 * gap (otari#992): while it holds, `existingEmail` cannot yet distinguish a
 * migrated identity from a bare one.
 */
function mockOrganizationContextPending() {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url === "/v1/organizations/me") {
      return new Promise<Response>(() => {})
    }
    return jsonResponse({ count: 0, data: [] })
  })
}

/**
 * Fails `GET /v1/organizations/me`, everything else answers normally. 403,
 * not 500: the app's `QueryClient` retries a failed query up to twice by
 * default and only skips that for 401/403, and a status this test would
 * retry through is a status this test would also have to wait through.
 */
function mockOrganizationContextFailing(put: unknown = CLAIMED) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url === "/v1/organizations/me") {
      return jsonResponse({ detail: "boom" }, 403)
    }
    if (url === "/v1/auth/password") {
      return jsonResponse(put)
    }
    return jsonResponse({ count: 0, data: [] })
  })
}

/**
 * Wait past the organization-context query the claim form gates on: the
 * email field is disabled while this identity's address is still unknown
 * (otari#992's loading-window gap), so a test that types and submits before
 * it settles would exercise a state real typing speed never reaches rather
 * than the form itself.
 */
async function identityResolved() {
  await waitFor(() => expect(screen.getByLabelText("Email")).not.toBeDisabled())
}

describe("PasswordCard on an unclaimed deployment", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("asks for an address and a password, and for no current one", () => {
    mockRequests()
    renderCard(["master_key"])

    expect(screen.getByLabelText("Email")).toBeInTheDocument()
    expect(screen.getByLabelText("New password")).toBeInTheDocument()
    expect(screen.queryByLabelText("Current password")).not.toBeInTheDocument()
  })

  it("claims the deployment and says the master key no longer signs in", async () => {
    const fetchMock = mockRequests()
    const user = userEvent.setup()
    renderCard(["master_key"])

    await identityResolved()
    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("New password"), "a-real-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "a-real-password",
    )
    await user.click(screen.getByRole("button", { name: "Set password" }))

    await waitFor(() => expect(putCall(fetchMock)).toBeDefined())
    const [url, init] = putCall(fetchMock) ?? []
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
    mockRequests()
    const user = userEvent.setup()
    renderCard(["master_key"])

    await identityResolved()
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

describe("PasswordCard on a migrated deployment (otari#992)", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  const MIGRATED_OPERATOR = {
    user_id: "44444444-4444-4444-4444-444444444444",
    email: "operator@example.com",
    full_name: null,
  }

  it("prefills and locks the field to the address the identity already holds", async () => {
    mockRequests({ caller: MIGRATED_OPERATOR })
    renderCard(["master_key"])

    const emailField = await screen.findByDisplayValue("operator@example.com")
    expect(emailField).toHaveAttribute("readonly")
    expect(
      screen.getByText(/already has a sign-in address/i),
    ).toBeInTheDocument()
  })

  it("claims without resending the address, and the gateway does not refuse it", async () => {
    const fetchMock = mockRequests({ caller: MIGRATED_OPERATOR })
    const user = userEvent.setup()
    renderCard(["master_key"])

    // The field is locked, so there is nothing to type into it; only the new
    // password is this operator's to choose. Waited on by its resolved value
    // rather than its mere presence, matching the loading-window gap: the
    // field exists (and is briefly editable) before the query settles.
    await screen.findByDisplayValue("operator@example.com")
    await user.type(screen.getByLabelText("New password"), "a-real-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "a-real-password",
    )
    await user.click(screen.getByRole("button", { name: "Set password" }))

    await waitFor(() => expect(putCall(fetchMock)).toBeDefined())
    const [, init] = putCall(fetchMock) ?? []
    // No `email` at all: resending it risks a normalization mismatch the
    // gateway would read as a change (EmailChangeNotSupportedError), and the
    // identity already has this address, so nothing here needs to claim one.
    expect(init?.body).toBe(JSON.stringify({ new_password: "a-real-password" }))
  })

  it("names the address in the claim copy instead of asking for one", async () => {
    mockRequests({ caller: MIGRATED_OPERATOR })
    renderCard(["master_key"])

    expect(
      await screen.findByText(
        /Set a password to sign in as operator@example\.com/i,
      ),
    ).toBeInTheDocument()
  })

  it("blocks submission while the identity's address is still unknown", async () => {
    const fetchMock = mockOrganizationContextPending()
    const user = userEvent.setup()
    renderCard(["master_key"])

    // Not readonly and not the migrated address either: this is the gap
    // between "resolved, no address" and "hasn't resolved yet" that a bare
    // `existingEmail == null` check cannot tell apart on its own.
    const emailField = screen.getByLabelText("Email")
    expect(emailField).toBeDisabled()
    expect(emailField).not.toHaveAttribute("readonly")
    expect(
      screen.getByText(/Checking whether this identity already has/i),
    ).toBeInTheDocument()

    await user.type(screen.getByLabelText("New password"), "a-real-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "a-real-password",
    )

    // Filling in both passwords is not enough while the identity is unknown:
    // submitting here is exactly the failure otari#992 removed for the
    // resolved case.
    expect(screen.getByRole("button", { name: "Set password" })).toBeDisabled()
    await user.click(screen.getByRole("button", { name: "Set password" }))
    expect(putCall(fetchMock)).toBeUndefined()
  })

  it("falls open, with a visible note, when the identity's address can't be confirmed", async () => {
    const fetchMock = mockOrganizationContextFailing()
    const user = userEvent.setup()
    renderCard(["master_key"])

    // Unlike the pending case, a failed lookup must not make the deployment
    // permanently unclaimable through its only UI: the field stays editable
    // and says why, rather than refusing quietly or blocking forever.
    const emailField = await screen.findByLabelText("Email")
    await waitFor(() =>
      expect(
        screen.getByText(/Could not confirm whether this identity/i),
      ).toBeInTheDocument(),
    )
    expect(emailField).not.toBeDisabled()
    expect(emailField).not.toHaveAttribute("readonly")

    await user.type(emailField, "operator@example.com")
    await user.type(screen.getByLabelText("New password"), "a-real-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "a-real-password",
    )
    await user.click(screen.getByRole("button", { name: "Set password" }))

    await waitFor(() => expect(putCall(fetchMock)).toBeDefined())
    const [, init] = putCall(fetchMock) ?? []
    expect(init?.body).toBe(
      JSON.stringify({
        email: "operator@example.com",
        new_password: "a-real-password",
      }),
    )
  })
})

describe("PasswordCard on a claimed deployment", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("requires the current password and sends it", async () => {
    const fetchMock = mockRequests()
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

    await waitFor(() => expect(putCall(fetchMock)).toBeDefined())
    const [, init] = putCall(fetchMock) ?? []
    expect(init?.body).toBe(
      JSON.stringify({
        current_password: "old-password",
        new_password: "new-password",
      }),
    )
  })

  it("refuses a new password that is the current one, before asking the gateway", async () => {
    const fetchMock = mockRequests()
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
    expect(putCall(fetchMock)).toBeUndefined()
  })

  it("renders the gateway's own refusal rather than a guess", async () => {
    mockRequests({
      put: { detail: "Current password is incorrect" },
      putStatus: 400,
    })
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

  it("drops the saved line as soon as any field is retyped", async () => {
    mockRequests()
    const user = userEvent.setup()
    renderCard(["password"])

    await user.type(screen.getByLabelText("Current password"), "old-password")
    await user.type(screen.getByLabelText("New password"), "new-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "new-password",
    )
    await user.click(screen.getByRole("button", { name: "Change password" }))
    expect(await screen.findByRole("status")).toBeInTheDocument()

    // The password fields count, not only the first one: a line reporting a
    // call that has been superseded is worse beside a half-filled form than no
    // line at all.
    await user.type(screen.getByLabelText("New password"), "a")

    expect(screen.queryByRole("status")).not.toBeInTheDocument()
  })
})

describe("PasswordCard policy checks", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("holds back a password under the minimum length", async () => {
    const fetchMock = mockRequests()
    const user = userEvent.setup()
    renderCard(["master_key"])

    await identityResolved()
    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("New password"), "short")
    await user.type(screen.getByLabelText("Confirm new password"), "short")

    expect(screen.getByText("At least 8 characters.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Set password" })).toBeDisabled()
    expect(putCall(fetchMock)).toBeUndefined()
  })

  it("counts the ceiling in bytes, not characters", async () => {
    const fetchMock = mockRequests()
    const user = userEvent.setup()
    renderCard(["master_key"])
    await identityResolved()

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
    expect(putCall(fetchMock)).toBeUndefined()
  })

  it("counts the minimum in code points, as the gateway does", async () => {
    const fetchMock = mockRequests()
    const user = userEvent.setup()
    renderCard(["master_key"])
    await identityResolved()

    // Seven emoji: 14 to JavaScript's `String.length` and 7 to Python's `len`,
    // so a UTF-16 count would enable Save and hand the gateway a password its
    // own eight-character minimum refuses.
    const emoji = "🔒".repeat(7)
    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("New password"), emoji)
    await user.type(screen.getByLabelText("Confirm new password"), emoji)

    expect(screen.getByRole("alert")).toHaveTextContent(
      "At least 8 characters.",
    )
    expect(screen.getByRole("button", { name: "Set password" })).toBeDisabled()
    expect(putCall(fetchMock)).toBeUndefined()
  })

  it("holds back a confirmation that does not match", async () => {
    const fetchMock = mockRequests()
    const user = userEvent.setup()
    renderCard(["master_key"])

    await identityResolved()
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
    expect(putCall(fetchMock)).toBeUndefined()
  })
})

describe("PasswordCard and the member roster", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("refreshes the roster, whose address a claim is what fills in", async () => {
    // The operator's roster row carries `email: null` until the claim writes
    // one, and `useOrganizationMembers` caches for a minute, so without an
    // invalidation the Members page would show the pre-claim row for the rest
    // of that minute.
    let memberFetches = 0
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/v1/organizations/me/members")) {
        memberFetches += 1
        return jsonResponse({ count: 0, data: [] })
      }
      if (url === "/v1/organizations/me") {
        return jsonResponse(organizationContext())
      }
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
    await identityResolved()

    await user.type(screen.getByLabelText("Email"), "operator@example.com")
    await user.type(screen.getByLabelText("New password"), "a-real-password")
    await user.type(
      screen.getByLabelText("Confirm new password"),
      "a-real-password",
    )
    await user.click(screen.getByRole("button", { name: "Set password" }))

    await waitFor(() => expect(memberFetches).toBe(2))
  })
})
