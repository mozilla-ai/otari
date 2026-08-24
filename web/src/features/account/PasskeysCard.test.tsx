import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { DeploymentBootstrap } from "@/client"
import { PasskeysCard } from "@/features/account/PasskeysCard"
import { DeploymentProvider, useDeployment } from "@/shared/hooks/useDeployment"
import { bootstrap } from "@/tests/fixtures"
import { AppProviders } from "@/tests/providers"

const PASSKEY = {
  id: "11111111-1111-1111-1111-111111111111",
  name: "Work laptop",
  credential_id: "Y3JlZA",
  rp_id: "otari.example.com",
  transports: ["internal"],
  backed_up: true,
  is_usable: true,
  created_at: "2026-08-01T10:00:00Z",
  last_used_at: null,
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/**
 * Route by path, because this card's flows are two or three calls each and the
 * order they arrive in is not the point of any test here.
 */
function mockApi(routes: Record<string, () => Response>) {
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : String(input)
      const method = init?.method ?? "GET"
      const handler = routes[`${method} ${url}`]
      if (!handler) {
        throw new Error(`unexpected request: ${method} ${url}`)
      }
      return Promise.resolve(handler())
    })
}

const LIST = "GET /v1/auth/webauthn/credentials"
const OPTIONS = "POST /v1/auth/webauthn/register/options"
const REGISTER = "POST /v1/auth/webauthn/register"

function renderCard(passkeysReady = true) {
  return render(
    <AppProviders>
      <DeploymentProvider value={bootstrap({ passkeys_ready: passkeysReady })}>
        <PasskeysCard />
      </DeploymentProvider>
    </AppProviders>,
  )
}

/**
 * Renders the card beside a readout of `sign_in_methods`.
 *
 * The card reports a first registration or a last deletion into the deployment
 * context rather than refetching it, so the only way to assert it did the right
 * thing is to read the context back. Kept out of `renderCard` because every
 * other test here is about the card's own DOM.
 */
function SignInMethodsProbe() {
  const { sign_in_methods } = useDeployment()
  return <output>{sign_in_methods.join(",")}</output>
}

function renderCardWithProbe(methods: DeploymentBootstrap["sign_in_methods"]) {
  return render(
    <AppProviders>
      <DeploymentProvider
        value={bootstrap({ passkeys_ready: true, sign_in_methods: methods })}
      >
        <PasskeysCard />
        <SignInMethodsProbe />
      </DeploymentProvider>
    </AppProviders>,
  )
}

async function deleteFirstPasskey(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getAllByRole("button", { name: "Delete" })[0])
  await user.click(
    within(screen.getByRole("alertdialog")).getByRole("button", {
      name: "Delete",
    }),
  )
}

// A browser that can run ceremonies, with the authenticator stubbed. Every test
// that reaches the register button needs this; `supportsPasskeys` is read on
// render, so it is stubbed before one.
function stubAuthenticator(create: ReturnType<typeof vi.fn>) {
  // A secure context as well as the API: `supportsPasskeys` gates on both, and
  // jsdom reports `isSecureContext` false by default.
  vi.stubGlobal("isSecureContext", true)
  vi.stubGlobal("PublicKeyCredential", function PublicKeyCredential() {})
  vi.stubGlobal("navigator", {
    ...globalThis.navigator,
    credentials: { create },
  })
}

function fakeCreated() {
  return {
    id: "Y3JlZA",
    rawId: new Uint8Array([1, 2, 3]).buffer,
    type: "public-key",
    response: {
      clientDataJSON: new Uint8Array([4, 5]).buffer,
      attestationObject: new Uint8Array([6, 7]).buffer,
      getTransports: () => ["internal"],
    },
    getClientExtensionResults: () => ({}),
  }
}

describe("PasskeysCard", () => {
  beforeEach(() => {
    stubAuthenticator(vi.fn())
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it("lists the registered passkeys", async () => {
    mockApi({ [LIST]: () => jsonResponse({ data: [PASSKEY], count: 1 }) })
    renderCard()

    expect(await screen.findByText("Work laptop")).toBeInTheDocument()
    // The sync hint is the fact that decides whether losing a device loses the
    // passkey, so it is on the row rather than behind a tooltip.
    expect(
      screen.getByText(/Synced to your credential manager/),
    ).toBeInTheDocument()
    expect(screen.getByText(/Never used/)).toBeInTheDocument()
  })

  it("says so when there are none yet", async () => {
    mockApi({ [LIST]: () => jsonResponse({ data: [], count: 0 }) })
    renderCard()

    expect(
      await screen.findByText("You have no passkeys yet."),
    ).toBeInTheDocument()
  })

  it("marks a passkey the deployment can no longer assert, and still offers to delete it", async () => {
    // Registered before the relying-party ID moved. Hiding it would leave the
    // operator with an unexplained empty list and no way to clean up.
    mockApi({
      [LIST]: () =>
        jsonResponse({ data: [{ ...PASSKEY, is_usable: false }], count: 1 }),
    })
    renderCard()

    expect(await screen.findByText("Work laptop")).toBeInTheDocument()
    expect(screen.getByText(/can no longer sign you in/)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument()
  })

  it("says what to configure, and offers nothing, on a deployment with no relying party", async () => {
    // otari#648's rule: a surface with no fallback is absent rather than
    // offered and then refused. The reason is still named, because this is the
    // operator's own settings page and the fix is one setting away.
    mockApi({ [LIST]: () => jsonResponse({ data: [], count: 0 }) })
    renderCard(false)

    expect(
      await screen.findByText(/not set up for passkeys yet/),
    ).toBeInTheDocument()
    expect(screen.getByText("public_base_url")).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Add a passkey" }),
    ).not.toBeInTheDocument()
  })

  it("still lists and deletes an orphan on a deployment with no relying party", async () => {
    // The case that makes the gate safe to apply: registration is off, but
    // whatever was registered before the ID went away stays manageable.
    mockApi({
      [LIST]: () =>
        jsonResponse({ data: [{ ...PASSKEY, is_usable: false }], count: 1 }),
    })
    renderCard(false)

    expect(await screen.findByText("Work laptop")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete" })).toBeInTheDocument()
  })

  it("registers a passkey through the browser ceremony", async () => {
    const create = vi.fn().mockResolvedValue(fakeCreated())
    stubAuthenticator(create)
    const fetchMock = mockApi({
      [LIST]: () => jsonResponse({ data: [], count: 0 }),
      [OPTIONS]: () =>
        jsonResponse({
          challenge: "Y2hhbGxlbmdl",
          rp: { id: "otari.example.com", name: "otari" },
          user: { id: "dXNlcg", name: "op", displayName: "Op" },
        }),
      [REGISTER]: () => jsonResponse(PASSKEY, 201),
    })
    const user = userEvent.setup()
    renderCard()

    await screen.findByText("You have no passkeys yet.")
    await user.type(await screen.findByLabelText("Name"), "Work laptop")
    await user.click(screen.getByRole("button", { name: "Add a passkey" }))

    await waitFor(() => expect(create).toHaveBeenCalled())
    await waitFor(() => {
      const registered = fetchMock.mock.calls.find(
        ([url, init]) =>
          String(url) === "/v1/auth/webauthn/register" &&
          (init as RequestInit)?.method === "POST",
      )
      if (!registered) {
        throw new Error("the register call was never made")
      }
      const body = JSON.parse(String((registered[1] as RequestInit).body))
      expect(body.name).toBe("Work laptop")
      // The serialized ceremony, not the raw browser object: a `rawId` still
      // holding an ArrayBuffer would reach the gateway as `{}`.
      expect(body.credential.rawId).toBe("AQID")
    })
  })

  it("says nothing when the passkey prompt is dismissed", async () => {
    const create = vi
      .fn()
      .mockRejectedValue(new DOMException("denied", "NotAllowedError"))
    stubAuthenticator(create)
    mockApi({
      [LIST]: () => jsonResponse({ data: [], count: 0 }),
      [OPTIONS]: () =>
        jsonResponse({
          challenge: "Y2hhbGxlbmdl",
          rp: { id: "otari.example.com", name: "otari" },
          user: { id: "dXNlcg", name: "op", displayName: "Op" },
        }),
    })
    const user = userEvent.setup()
    renderCard()

    await screen.findByText("You have no passkeys yet.")
    await user.click(screen.getByRole("button", { name: "Add a passkey" }))

    await waitFor(() => expect(create).toHaveBeenCalled())
    // Pressing Escape is a decision, not a failed registration, so no banner.
    await waitFor(() =>
      expect(screen.queryByRole("alert")).not.toBeInTheDocument(),
    )
  })

  it("shows the gateway's own reason when the deployment is not configured", async () => {
    mockApi({
      [LIST]: () =>
        jsonResponse(
          {
            detail:
              "Passkeys are unavailable on this deployment: it does not know its own address. Set public_base_url (or webauthn_rp_id) and restart.",
          },
          503,
        ),
    })
    renderCard()

    // Naming the setting is the point: a friendlier message would hide the fix.
    expect(await screen.findByText(/public_base_url/)).toBeInTheDocument()
    // And there is nothing to press, because a ceremony cannot start.
    expect(
      screen.queryByRole("button", { name: "Add a passkey" }),
    ).not.toBeInTheDocument()
  })

  it("does not offer registration in a browser that cannot do it", async () => {
    vi.unstubAllGlobals()
    vi.stubGlobal("PublicKeyCredential", undefined)
    mockApi({ [LIST]: () => jsonResponse({ data: [], count: 0 }) })
    renderCard()

    expect(
      await screen.findByText(/This browser cannot use passkeys/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Add a passkey" }),
    ).not.toBeInTheDocument()
  })

  it("renames a passkey", async () => {
    const fetchMock = mockApi({
      [LIST]: () => jsonResponse({ data: [PASSKEY], count: 1 }),
      [`PATCH /v1/auth/webauthn/credentials/${PASSKEY.id}`]: () =>
        jsonResponse({ ...PASSKEY, name: "Old laptop" }),
    })
    const user = userEvent.setup()
    renderCard()

    await screen.findByText("Work laptop")
    await user.click(screen.getByRole("button", { name: "Rename" }))
    // Scoped to the dialog: the register form below carries a "Name" field of
    // its own, and an unscoped query would match both.
    const dialog = within(await screen.findByRole("alertdialog"))
    const field = dialog.getByLabelText("Name")
    await user.clear(field)
    await user.type(field, "Old laptop")
    await user.click(dialog.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      const patched = fetchMock.mock.calls.find(
        ([, init]) => (init as RequestInit)?.method === "PATCH",
      )
      if (!patched) {
        throw new Error("the rename call was never made")
      }
      expect(JSON.parse(String((patched[1] as RequestInit).body))).toEqual({
        name: "Old laptop",
      })
    })
  })

  it("does not send a blank rename", async () => {
    const fetchMock = mockApi({
      [LIST]: () => jsonResponse({ data: [PASSKEY], count: 1 }),
    })
    const user = userEvent.setup()
    renderCard()

    await screen.findByText("Work laptop")
    await user.click(screen.getByRole("button", { name: "Rename" }))
    const dialog = within(await screen.findByRole("alertdialog"))
    await user.clear(dialog.getByLabelText("Name"))
    await user.click(dialog.getByRole("button", { name: "Save" }))

    // The dialog stays open and nothing is sent, rather than a 422 arriving
    // from inside a dialog that looked like it worked.
    expect(screen.getByRole("alertdialog")).toBeInTheDocument()
    expect(
      fetchMock.mock.calls.some(
        ([, init]) => (init as RequestInit)?.method === "PATCH",
      ),
    ).toBe(false)
  })

  it("deletes a passkey after confirming, and says the password still works", async () => {
    const fetchMock = mockApi({
      [LIST]: () => jsonResponse({ data: [PASSKEY], count: 1 }),
      [`DELETE /v1/auth/webauthn/credentials/${PASSKEY.id}`]: () =>
        new Response(null, { status: 204 }),
    })
    const user = userEvent.setup()
    renderCard()

    await screen.findByText("Work laptop")
    await user.click(screen.getByRole("button", { name: "Delete" }))

    // Removing the last passkey is not a lockout, and the dialog says why.
    expect(
      await screen.findByText(/You can still sign in with your password/),
    ).toBeInTheDocument()
    // Scoped for the same reason the rename is: the row behind the dialog has
    // its own Delete button.
    await user.click(
      within(screen.getByRole("alertdialog")).getByRole("button", {
        name: "Delete",
      }),
    )

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.some(
          ([, init]) => (init as RequestInit)?.method === "DELETE",
        ),
      ).toBe(true)
    })
  })

  it("stops offering passkey sign-in once the last usable one is deleted", async () => {
    mockApi({
      [LIST]: () => jsonResponse({ data: [PASSKEY], count: 1 }),
      [`DELETE /v1/auth/webauthn/credentials/${PASSKEY.id}`]: () =>
        new Response(null, { status: 204 }),
    })
    const user = userEvent.setup()
    renderCardWithProbe(["passkey", "password"])

    await screen.findByText("Work laptop")
    await deleteFirstPasskey(user)

    // The gateway would stop publishing `passkey`, and this tab has to agree
    // now rather than at the next reload, or the sign-in screen it lands on
    // offers a button nothing can answer.
    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("password"),
    )
    expect(screen.getByRole("status")).not.toHaveTextContent("passkey")
  })

  it("keeps offering passkey sign-in when an unusable one is all that goes", async () => {
    // An orphan and a working passkey. Deleting the orphan changes nothing the
    // gateway counts, because `has_any_credential` never counted it.
    const orphan = {
      ...PASSKEY,
      id: "22222222-2222-2222-2222-222222222222",
      name: "Old laptop",
      rp_id: "old.example.com",
      is_usable: false,
    }
    mockApi({
      // Newest first, so the orphan is the first Delete button on the page.
      [LIST]: () => jsonResponse({ data: [orphan, PASSKEY], count: 2 }),
      [`DELETE /v1/auth/webauthn/credentials/${orphan.id}`]: () =>
        new Response(null, { status: 204 }),
    })
    const user = userEvent.setup()
    renderCardWithProbe(["passkey", "password"])

    await screen.findByText("Old laptop")
    await deleteFirstPasskey(user)

    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("passkey"),
    )
  })

  it("stops offering passkey sign-in when the last usable one goes beside an orphan", async () => {
    // The case a row count gets wrong: two rows, one usable. Deleting the
    // usable one leaves nothing the gateway would count, so the offer has to be
    // withdrawn even though a row remains.
    const orphan = {
      ...PASSKEY,
      id: "22222222-2222-2222-2222-222222222222",
      name: "Old laptop",
      rp_id: "old.example.com",
      is_usable: false,
    }
    mockApi({
      // PASSKEY first this time, so the usable one is the button that is clicked.
      [LIST]: () => jsonResponse({ data: [PASSKEY, orphan], count: 2 }),
      [`DELETE /v1/auth/webauthn/credentials/${PASSKEY.id}`]: () =>
        new Response(null, { status: 204 }),
    })
    const user = userEvent.setup()
    renderCardWithProbe(["passkey", "password"])

    await screen.findByText("Work laptop")
    await deleteFirstPasskey(user)

    await waitFor(() =>
      expect(screen.getByRole("status")).toHaveTextContent("password"),
    )
    expect(screen.getByRole("status")).not.toHaveTextContent("passkey")
  })
})
