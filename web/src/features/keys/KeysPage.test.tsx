import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor, within } from "@testing-library/react"
import userEvent, { type UserEvent } from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { ApiKey, DeploymentBootstrap, User } from "@/client"
import { KeysPage } from "@/features/keys/KeysPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { apiKey, bootstrap, organizationMember } from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"
import { pickOption } from "@/tests/select"

function user(overrides: Partial<User> = {}): User {
  return {
    user_id: "alice",
    // An alias is what makes the owner option's label differ from its user_id,
    // which is the case the picker has to get right.
    alias: "Alice",
    spend: 0,
    reserved: 0,
    current_tokens: 0,
    reserved_tokens: 0,
    current_requests: 0,
    reserved_requests: 0,
    budget_id: null,
    allowed_models: null,
    budget_started_at: null,
    next_budget_reset_at: null,
    blocked: false,
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    metadata: {},
    ...overrides,
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

const NEW_SECRET = "gw-NEWSECRET0000000000000000000000000000000000000000000000"
const REGEN_SECRET =
  "gw-REGEN00000000000000000000000000000000000000000000000000"

// Both key surfaces answer identical shapes, so one handler serves them: the
// operator's /api/v1/keys and the member's /api/v1/organizations/me/keys
// (otari-ai#1941). Which one the page asked is what the member-view cases
// assert, off the spy's recorded URLs.
const KEYS_URL = /\/api\/v1\/(?:organizations\/me\/)?keys(?:\/|\?|$)/

function mockApi(
  opts: {
    keys?: ApiKey[]
    users?: User[]
    members?: ReturnType<typeof organizationMember>[]
    deploymentOperator?: boolean
  } = {},
) {
  let list = [...(opts.keys ?? [])]
  const users = opts.users ?? []
  const members = opts.members ?? []

  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()

      if (KEYS_URL.test(url)) {
        if (url.endsWith("/rotate") && method === "POST") {
          const id = url.split("/").slice(-2)[0]
          const prefix = REGEN_SECRET.slice(0, 10)
          list = list.map((k) =>
            k.id === id ? { ...k, key_prefix: prefix } : k,
          )
          const row = list.find((k) => k.id === id) ?? apiKey({ id })
          return jsonResponse({ ...row, key: REGEN_SECRET, key_prefix: prefix })
        }
        if (method === "POST") {
          const body = JSON.parse(String(init?.body)) as {
            key_name?: string | null
            user_id?: string | null
            allowed_models?: string[] | null
            reject_user_mismatch?: boolean | null
          }
          const row = apiKey({
            id: "key-new",
            key_prefix: NEW_SECRET.slice(0, 10),
            key_name: body.key_name ?? null,
            user_id: body.user_id ?? "apikey-key-new",
            allowed_models: body.allowed_models ?? null,
            reject_user_mismatch: body.reject_user_mismatch ?? null,
          })
          list = [...list, row]
          return jsonResponse({ ...row, key: NEW_SECRET })
        }
        if (method === "PATCH") {
          const id = decodeURIComponent(url.split("/").pop() ?? "")
          const body = JSON.parse(String(init?.body)) as Partial<ApiKey>
          list = list.map((k) => (k.id === id ? { ...k, ...body } : k))
          return jsonResponse(list.find((k) => k.id === id))
        }
        if (method === "DELETE") {
          const id = decodeURIComponent(url.split("/").pop() ?? "")
          list = list.filter((k) => k.id !== id)
          return new Response(null, { status: 204 })
        }
        return jsonResponse(list)
      }
      // Before /api/v1/users, and paged: the owner picker names members through
      // this, and `fetchAllPaged` reads `data`/`count` rather than a bare list.
      if (url.includes(`${API_ROOT}/organizations/me/members`)) {
        return jsonResponse({ data: members, count: members.length })
      }
      // Seeds the scope: `deployment_operator` is what routes the page onto the
      // operator surface or the member one. These suites default to the
      // operator's view, and the member cases flip it.
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse({
          organization_member_id: "om-1",
          role: "member",
          status: "active",
          organization: {
            id: "org-1",
            name: "Acme",
            slug: "acme",
            created_by_user_id: null,
            created_at: "2026-01-01T00:00:00+00:00",
            updated_at: null,
          },
          byo_provider_keys_allowed: true,
          deployment_operator: opts.deploymentOperator ?? true,
          provider_key_encryption_available: true,
          workspace_memberships: [],
        })
      }
      if (url.includes(`${API_ROOT}/users`)) {
        return jsonResponse(users)
      }
      if (url.includes(`${API_ROOT}/models/discoverable`)) {
        return jsonResponse({
          providers: [
            {
              provider: "openai",
              ok: true,
              error: null,
              models: [{ id: "gpt-4o", key: "openai:gpt-4o" }],
            },
          ],
        })
      }
      if (url.includes(`${API_ROOT}/providers`)) {
        return jsonResponse({ providers: [{ instance: "openai" }] })
      }
      if (url.includes(`${API_ROOT}/aliases`)) {
        return jsonResponse([])
      }
      return jsonResponse([])
    })
}

function renderPage(
  ui: ReactElement,
  deployment: DeploymentBootstrap = bootstrap(),
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  // The page reads the deployment's surfaces to decide whether it may name key
  // owners from the organization roster, and links to the pages that own a key's
  // budget, so it needs both the deployment context and a router around it.
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={deployment}>{ui}</DeploymentProvider>
    </QueryClientProvider>,
  )
}

/**
 * Press the create dialog's submit.
 *
 * Scoped to the dialog because "Create key" is deliberately on screen twice
 * while it is open: the labels rule makes the page's trigger and the dialog's
 * submit the same string, and the trigger no longer hides behind the form.
 */
async function submitTheCreateDialog(user: UserEvent) {
  const dialog = await screen.findByRole("dialog")
  await user.click(within(dialog).getByRole("button", { name: "Create key" }))
}

describe("KeysPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows a single empty state (onboarding panel, not also the table fallback)", async () => {
    mockApi({ keys: [] })
    renderPage(<KeysPage />)

    // The onboarding panel owns the empty state.
    expect(
      await screen.findByRole("heading", { name: "No API keys yet" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Create your first key" }),
    ).toBeInTheDocument()
    // The table (and its own "no rows" fallback) is suppressed, so the two empty
    // states are not stacked. "authenticate a caller" is unique to that fallback.
    expect(screen.queryByText(/authenticate a caller/)).not.toBeInTheDocument()
    expect(
      screen.queryByRole("grid", { name: "API keys" }),
    ).not.toBeInTheDocument()
  })

  it("lists keys with status and prefix, never the full secret", async () => {
    mockApi({
      keys: [
        apiKey({
          id: "key-1",
          key_name: "ci-bot",
          key_prefix: "gw-AbC3dE",
          is_active: true,
        }),
        apiKey({
          id: "key-2",
          key_name: "legacy",
          key_prefix: null,
          is_active: false,
        }),
      ],
    })
    renderPage(<KeysPage />)

    const activeRow = (await screen.findByText("ci-bot")).closest("tr")!
    expect(within(activeRow).getByText("Active")).toBeInTheDocument()
    expect(within(activeRow).getByText("gw-AbC3dE…")).toBeInTheDocument()

    // A key minted before the prefix existed renders "—", not a crash.
    const legacyRow = screen.getByText("legacy").closest("tr")!
    expect(within(legacyRow).getByText("Disabled")).toBeInTheDocument()
    expect(within(legacyRow).getByText("—")).toBeInTheDocument()

    expect(document.body.textContent).not.toContain(NEW_SECRET)
  })

  it("shows the plaintext + first-call snippet once, then only the prefix", async () => {
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByLabelText("Name"), "deploy-key")
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    // Open revealed: this screen exists to hand the key over, and there is no
    // second chance to read it.
    const reveal = await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    expect(within(reveal).getByLabelText("Secret key")).toHaveValue(NEW_SECRET)
    const curl = within(reveal).getByLabelText("curl") as HTMLTextAreaElement
    const python = within(reveal).getByLabelText(
      "Python (OpenAI SDK)",
    ) as HTMLTextAreaElement
    expect(curl.value).toContain(`Otari-Key: ${NEW_SECRET}`)
    expect(curl.value).toContain(
      `${window.location.origin}${API_ROOT}/chat/completions`,
    )
    expect(python.value).toContain(NEW_SECRET)

    // One credential, one reveal: concealing the key conceals the requests that
    // carry it, rather than leaving it in plain sight twice over.
    await user.click(
      within(reveal).getByRole("button", { name: "Hide Secret key" }),
    )
    expect(within(reveal).getByLabelText("Secret key")).not.toHaveValue(
      NEW_SECRET,
    )
    expect(curl.value).not.toContain(NEW_SECRET)
    expect(python.value).not.toContain(NEW_SECRET)

    // And the affordance is on each field, not only on the key: the snippet's
    // own toggle brings all three back (otari-ai#2111).
    await user.click(within(reveal).getByRole("button", { name: "Show curl" }))
    expect(within(reveal).getByLabelText("Secret key")).toHaveValue(NEW_SECRET)
    expect(curl.value).toContain(NEW_SECRET)
    expect(python.value).toContain(NEW_SECRET)

    // The acknowledgement is the dialog's footer action, so it is outside the
    // alert that holds the key. It is unique on screen either way.
    await user.click(
      screen.getByRole("button", { name: /I.?ve saved this key/ }),
    )

    // After closing, the list shows only the prefix and the secret is gone from the DOM.
    expect(
      screen.queryByRole("alert", { name: /API key created|New secret for/ }),
    ).not.toBeInTheDocument()
    expect(
      await screen.findByText(`${NEW_SECRET.slice(0, 10)}…`),
    ).toBeInTheDocument()
    expect(document.body.textContent).not.toContain(NEW_SECRET)
  })

  /** Mint a key and open the one-time reveal, which is where the snippets live. */
  async function revealANewKey(deployment?: DeploymentBootstrap) {
    mockApi({ keys: [] })
    const person = userEvent.setup()
    renderPage(<KeysPage />, deployment)

    await screen.findByText("No API keys yet")
    await person.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await person.type(screen.getByLabelText("Name"), "deploy-key")
    await person.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await person.keyboard("{Escape}")
    await person.click(screen.getByRole("button", { name: "Create key" }))

    return within(
      await screen.findByRole("alert", {
        name: /API key created|New secret for/,
      }),
    )
  }

  it("sends the snippet at the data plane a hosted deployment published", async () => {
    // The control plane serves this dashboard and is deliberately not where
    // inference belongs (otari#822), so the origin is the one address the
    // snippet must not name.
    const dialog = await revealANewKey(
      bootstrap({
        deployment_type: "hosted",
        data_plane_url: "https://gateway.otari.ai",
      }),
    )

    const curl = dialog.getByLabelText("curl") as HTMLTextAreaElement
    expect(curl.value).toContain(
      `https://gateway.otari.ai${API_ROOT}/chat/completions`,
    )
    expect(curl.value).not.toContain(window.location.origin)
  })

  it("shows no snippet when a hosted deployment published no data plane", async () => {
    // Withheld rather than aimed at this host: a placeholder would be a URL
    // nobody reading it could replace, and the origin would be the bug itself.
    const dialog = await revealANewKey(
      bootstrap({ deployment_type: "hosted", data_plane_url: null }),
    )

    expect(dialog.getByLabelText("Secret key")).toBeInTheDocument()
    expect(dialog.queryByLabelText("curl")).not.toBeInTheDocument()
    expect(
      dialog.getByText(/has not published the gateway address/),
    ).toBeInTheDocument()
  })

  it("keeps the reveal up through a stray Escape; only the save button dismisses it", async () => {
    // The reveal is a strip on the page now rather than a modal, so there is no
    // Esc handler to suppress and no backdrop to click. What still has to hold
    // is the thing the modal was protecting: a one-time secret cannot be lost
    // to a keystroke aimed at something else.
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    await user.keyboard("{Escape}")
    expect(
      screen.getByRole("alert", { name: /API key created|New secret for/ }),
    ).toBeInTheDocument()

    // The acknowledgement is the dialog's footer action, so it is outside the
    // alert that holds the key. It is unique on screen either way.
    await user.click(
      screen.getByRole("button", { name: /I.?ve saved this key/ }),
    )
    expect(
      screen.queryByRole("alert", { name: /API key created|New secret for/ }),
    ).not.toBeInTheDocument()
  })

  // The reveal is a modal again, and the objection its docstring used to raise
  // is answered rather than ignored: Escape and the backdrop work everywhere in
  // this dialog EXCEPT here, where the content cannot be recovered. So the press
  // that must not dismiss it is the backdrop, which is the only "elsewhere" a
  // modal has; the page behind is deliberately out of the accessibility tree now
  // and there is nothing on it to click.
  it("keeps the reveal up through a press on the backdrop", async () => {
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    const backdrop = document.querySelector('[class*="modal__backdrop"]')
    expect(backdrop).not.toBeNull()
    await user.click(backdrop as Element)
    expect(
      screen.getByRole("alert", { name: /API key created|New secret for/ }),
    ).toBeInTheDocument()

    // The acknowledgement is the dialog's footer action, so it is outside the
    // alert that holds the key. It is unique on screen either way.
    await user.click(
      screen.getByRole("button", { name: /I.?ve saved this key/ }),
    )
    await waitFor(() =>
      expect(
        screen.queryByRole("alert", {
          name: /API key created|New secret for/,
        }),
      ).toBeNull(),
    )
  })

  it("returns focus to the page's create action when the reveal closes", async () => {
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    // The acknowledgement is the dialog's footer action, so it is outside the
    // alert that holds the key. It is unique on screen either way.
    await user.click(
      screen.getByRole("button", { name: /I.?ve saved this key/ }),
    )

    // The form that opened the reveal is gone by now, so nothing else has a
    // claim on focus and it would otherwise be left on <body>.
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Create key" }),
      ),
    )
  })

  // Rewritten from "moves focus onto the confirm and back on cancel". That
  // version scoped every query to `within(row)` because the confirm used to
  // swap the trigger for a Confirm/Cancel pair in place, and React reused the
  // same <button> node, so focus rode onto Confirm with nothing managing it.
  // The confirmation is a sibling row now, which changes the facts: probing
  // document.activeElement shows the trigger is never unmounted and keeps
  // focus, so arming cannot strand it on <body>. That is what is asserted.
  it("keeps the armed row's trigger mounted and focused, so arming never drops focus", async () => {
    mockApi({ keys: [apiKey()] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    const trigger = within(row).getByRole("button", { name: "Regenerate" })
    await user.click(trigger)

    await screen.findByText(/stops working immediately/)
    // Same node, not a re-rendered replacement: the strip is added beside the
    // row rather than swapped into it.
    expect(within(row).getByRole("button", { name: "Regenerate" })).toBe(
      trigger,
    )
    expect(document.activeElement).not.toBe(document.body)
  })

  it("returns focus to the row action when an armed row is cancelled", async () => {
    // Cancelling unmounts the strip from under the focused Confirm, which the
    // browser answers by moving focus to <body>: the keyboard loses its place
    // on the most destructive control in the row. `useConfirmationFocus` hands
    // it back, and the trigger is identified by `lastArmed` rather than `armed`
    // so the ref is still attached when the hook's effect runs.
    mockApi({ keys: [apiKey()] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    const trigger = within(row).getByRole("button", { name: "Regenerate" })
    await user.click(trigger)
    await screen.findByText(/stops working immediately/)

    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await waitFor(() =>
      expect(screen.queryByText(/stops working immediately/)).toBeNull(),
    )

    expect(document.activeElement).not.toBe(document.body)
    expect(document.activeElement).toBe(
      within(row).getByRole("button", { name: "Regenerate" }),
    )
  })

  it("confirms Copied when the clipboard API is available", async () => {
    mockApi({ keys: [] })
    const user = userEvent.setup()
    // Install after userEvent.setup(), which otherwise replaces navigator.clipboard
    // with its own stub.
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    })
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    const reveal = await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    // Concealed first, because copying without reading is what has to keep
    // working once the operator puts the key away (otari-ai#2111). The step
    // opens revealed, so the toggle is how that state is reached now.
    await user.click(
      within(reveal).getByRole("button", { name: "Hide Secret key" }),
    )
    const copyButtons = within(reveal).getAllByRole("button", { name: "Copy" })
    await user.click(copyButtons[0])

    // The key reached the clipboard and never the screen.
    expect(writeText).toHaveBeenCalledWith(NEW_SECRET)
    expect(
      within(reveal).queryByDisplayValue(NEW_SECRET),
    ).not.toBeInTheDocument()
    expect(
      await within(reveal).findByText("Copied to clipboard."),
    ).toBeInTheDocument()
  })

  it("disables an active key via PATCH, then offers permanent delete", async () => {
    const fetchMock = mockApi({
      keys: [apiKey({ id: "key-1", key_name: "ci-bot", is_active: true })],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    // An active key offers no Delete (require-disable-first).
    expect(
      within(row).queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument()

    await user.click(within(row).getByRole("button", { name: "Disable" }))

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes(`${API_ROOT}/keys/key-1`) &&
        (init?.method ?? "") === "PATCH",
    )
    expect(JSON.parse(String(patch?.[1]?.body))).toEqual({ is_active: false })

    // Once disabled, a Delete action appears.
    const disabledRow = (await screen.findByText("Disabled")).closest("tr")!
    expect(
      within(disabledRow).getByRole("button", { name: "Delete" }),
    ).toBeInTheDocument()
  })

  it("regenerates a secret after an explicit confirm", async () => {
    mockApi({
      keys: [apiKey({ id: "key-1", key_name: "ci-bot", is_active: true })],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Regenerate" }))
    // Arming opens a strip in its own row directly under the key's, so the
    // message and the confirm are siblings of that row rather than inside it.
    const armed = screen
      .getByText(/stops working immediately/)
      .closest("tr") as HTMLTableRowElement
    expect(armed).not.toBe(row)
    expect(row.nextElementSibling).toBe(armed)
    // The message names the key it is about, which is the whole point of
    // confirming in place rather than in a dialog.
    expect(within(armed).getByText("ci-bot")).toBeInTheDocument()
    await user.click(within(armed).getByRole("button", { name: "Regenerate" }))

    const reveal = await screen.findByRole("alert", {
      name: /API key created|New secret for/,
    })
    // Revealed on arrival, the same as a created key: regenerate hands over the
    // same thing and hands it over the same way.
    expect(within(reveal).getByDisplayValue(REGEN_SECRET)).toBeInTheDocument()
  })

  it("keeps the page's create action visible while the dialog is open", async () => {
    // It used to hide itself while the inline form was on the page. The form is
    // over the page now, so hiding the control that opened it would take the
    // heading's action away mid-task for no reason, and it is also where focus
    // returns.
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    const trigger = screen.getByRole("button", { name: "Create key" })
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )

    await screen.findByRole("dialog")
    expect(trigger).toBeInTheDocument()
  })

  it("clears the owner and the budget exemption when it reopens", async () => {
    // The draft is fresh on every open, which is what the page's open counter
    // buys: the dialog stays mounted through the exit so its content is intact
    // while it animates out, and the remount on the way in is what clears it.
    // Before that, the reset ran on close and left the owner and the exemption
    // behind, so the next key inherited both and a reopened dialog was dirty on
    // arrival, arming the guard on a form nobody had touched.
    mockApi({ keys: [], users: [user({ user_id: "alice", alias: "Alice" })] })
    const usr = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await usr.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await usr.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    // Focus off the picker before reaching for anything else: its popover is
    // open and react-aria aria-hides the rest of the dialog while it is.
    await usr.click(screen.getByLabelText("Name"))
    await usr.click(screen.getByRole("button", { name: "Advanced" }))
    await usr.click(screen.getByLabelText("Exempt from budget"))

    // Out through the guard, which is the only way out of a dirty form.
    await usr.keyboard("{Escape}")
    await usr.click(screen.getByRole("button", { name: "Discard" }))

    await usr.click(screen.getByRole("button", { name: "Create key" }))
    expect(screen.getByPlaceholderText(/Pick a user/)).toHaveValue("")
    await usr.click(screen.getByRole("button", { name: "Advanced" }))
    expect(screen.getByLabelText("Exempt from budget")).not.toBeChecked()
    // And nothing is unsaved on arrival, so Escape closes rather than guarding.
    await usr.keyboard("{Escape}")
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("guards a field the old dirty check did not know about", async () => {
    // The check read three of the seven values the form owns, so a backdrop
    // press or Escape discarded the rest without asking.
    mockApi({ keys: [] })
    const usr = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await usr.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await usr.click(screen.getByRole("button", { name: "Advanced" }))
    await usr.click(screen.getByLabelText("Exempt from budget"))

    await usr.keyboard("{Escape}")
    expect(screen.getByRole("dialog")).toHaveTextContent("Unsaved changes")
  })

  it("does not walk /v1/users until the create dialog is opened", async () => {
    // The dialog stays mounted while closed so it can animate out, which left
    // its owner picker's roster fetching on every visit to the page.
    // `fetchAllUsers` walks up to 100 pages of 1000, so this is the page's
    // cost, not the dialog's. The same shape is waiting on every other page
    // this series migrates.
    const fetchMock = mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    const usersCalls = () =>
      fetchMock.mock.calls.filter(([u]) =>
        String(u).includes(`${API_ROOT}/users`),
      )
    expect(usersCalls()).toHaveLength(0)

    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await screen.findByRole("dialog")
    await waitFor(() => expect(usersCalls().length).toBeGreaterThan(0))
  })

  it("hands over the secret with no way for the form to skip it", async () => {
    // The one chance to read the key is not something the form can waive. The
    // only "Create another" is on the secret step, where it is reached by
    // having been shown the key first.
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).queryByLabelText("Create another"),
    ).not.toBeInTheDocument()
    await user.type(screen.getByLabelText("Name"), "first-key")
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    expect(
      await screen.findByRole("alert", { name: /API key created/ }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: /I.?ve saved this key/ }),
    ).toBeInTheDocument()
  })

  it("shows a regenerated secret in the same dialog a created one arrives in", async () => {
    // Create and regenerate hand over the same thing, so they hand it over the
    // same way. Before this they were a dialog and a full-width strip on one
    // page.
    mockApi({
      keys: [apiKey({ id: "key-1", key_name: "ci-bot", is_active: true })],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Regenerate" }))
    const armed = screen
      .getByText(/stops working immediately/)
      .closest("tr") as HTMLTableRowElement
    await user.click(within(armed).getByRole("button", { name: "Regenerate" }))

    const dialog = await screen.findByRole("dialog")
    expect(
      within(dialog).getByRole("alert", { name: /New secret for ci-bot/ }),
    ).toBeInTheDocument()
    // No form step and nothing to create again: the key already exists.
    expect(
      within(dialog).queryByRole("button", { name: "Create another" }),
    ).not.toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: /I.?ve saved this key/ }),
    ).toBeInTheDocument()
  })

  it("permanently deletes a disabled key after confirm", async () => {
    const fetchMock = mockApi({
      keys: [apiKey({ id: "key-1", key_name: "legacy", is_active: false })],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("legacy")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(/unlinks its usage history/),
    ).toBeInTheDocument()
    // The key is named in the dialog, so the operator is not confirming against
    // a row they can no longer see.
    expect(within(dialog).getByText("legacy")).toBeInTheDocument()
    await user.click(
      within(dialog).getByRole("button", { name: "Delete permanently" }),
    )

    const del = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes(`${API_ROOT}/keys/key-1`) &&
        (init?.method ?? "") === "DELETE",
    )
    expect(del).toBeDefined()
    expect(screen.queryByText("legacy")).not.toBeInTheDocument()
  })

  it("creates a key restricted to selected models", async () => {
    const fetchMock = mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Advanced" }))
    await user.click(screen.getByRole("button", { name: "Only selected" }))
    // The scope picker is a catalog combobox, not free text: type to filter, then
    // pick the discovered model.
    await user.type(screen.getByLabelText("Add a model"), "gpt-4o")
    await user.click(
      await screen.findByRole("option", { name: "openai:gpt-4o" }),
    )
    // Close the combobox popover, which otherwise aria-hides the submit button.
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith(`${API_ROOT}/keys`) &&
        (init?.method ?? "") === "POST",
    )
    expect(JSON.parse(String(post?.[1]?.body)).allowed_models).toEqual([
      "openai:gpt-4o",
    ])
    // User-first: the key names its owner rather than auto-creating a virtual user.
    expect(JSON.parse(String(post?.[1]?.body)).user_id).toBe("alice")
  })

  it("creates a budget-exempt key when the toggle is checked", async () => {
    const fetchMock = mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    // The exempt toggle lives under the Advanced disclosure.
    await user.click(screen.getByRole("button", { name: "Advanced" }))
    await user.click(screen.getByLabelText("Exempt from budget"))
    await user.click(screen.getByRole("button", { name: "Create key" }))

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith(`${API_ROOT}/keys`) &&
        (init?.method ?? "") === "POST",
    )
    expect(JSON.parse(String(post?.[1]?.body)).exclude_from_budget).toBe(true)
  })

  it("creates a key pinned to accept a mismatched user field", async () => {
    const fetchMock = mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Advanced" }))
    await pickOption(user, "Mismatched user field", "Always accept")
    await user.click(screen.getByRole("button", { name: "Create key" }))

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith(`${API_ROOT}/keys`) &&
        (init?.method ?? "") === "POST",
    )
    expect(JSON.parse(String(post?.[1]?.body)).reject_user_mismatch).toBe(false)
    // The created row carries the override back, so the list reflects it.
    expect(await screen.findByText("Lenient user")).toBeInTheDocument()
  })

  it("defaults a new key to inheriting the deployment setting", async () => {
    const fetchMock = mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await submitTheCreateDialog(user)

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith(`${API_ROOT}/keys`) &&
        (init?.method ?? "") === "POST",
    )
    expect(JSON.parse(String(post?.[1]?.body)).reject_user_mismatch).toBeNull()
  })

  it("renders a Budget-exempt chip for exempt keys", async () => {
    mockApi({
      keys: [
        apiKey({ id: "key-1", key_name: "ci-bot", exclude_from_budget: true }),
      ],
    })
    renderPage(<KeysPage />)
    expect(await screen.findByText("Budget-exempt")).toBeInTheDocument()
  })

  it("chips a key that overrides the deployment user-mismatch setting", async () => {
    mockApi({
      keys: [
        apiKey({
          id: "key-1",
          key_name: "claude-code",
          reject_user_mismatch: false,
        }),
        apiKey({
          id: "key-2",
          key_name: "pinned-strict",
          reject_user_mismatch: true,
        }),
        apiKey({
          id: "key-3",
          key_name: "inherits",
          reject_user_mismatch: null,
        }),
      ],
    })
    renderPage(<KeysPage />)

    expect(await screen.findByText("Lenient user")).toBeInTheDocument()
    expect(screen.getByText("Strict user")).toBeInTheDocument()
    // A key that inherits gets no chip: there is nothing unusual to flag.
    const inheritRow = screen.getByText("inherits").closest("tr")!
    expect(within(inheritRow).queryByText(/user$/)).not.toBeInTheDocument()
  })

  it("posts the picked owner's user_id, not the option's display label", async () => {
    // Regression: picking an existing owner used to submit the option's label
    // ("alice (Alice)") because selecting writes that text back into the input,
    // re-firing onInputChange. The keys API does not know that id, so it silently
    // created a second user aliased "User alice (Alice)" instead of reusing alice.
    const fetchMock = mockApi({
      keys: [],
      users: [user({ user_id: "alice", alias: "Alice" })],
    })
    const usr = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await usr.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await usr.click(screen.getByPlaceholderText(/Pick a user/))
    await usr.click(
      await screen.findByRole("option", { name: "alice (Alice)" }),
    )
    // Focus goes to another field rather than Escape putting the popover away.
    // The box is `menuTrigger="focus"`, so selecting an option hands focus back
    // to the input and the popover reopens, and react-aria marks the rest of the
    // page `aria-hidden` while it is open, which is what puts the submit out of
    // reach. Escape would close it and then reach the dialog, arming the
    // unsaved-changes guard and taking the submit out of the footer instead.
    await usr.click(screen.getByLabelText("Name"))
    await submitTheCreateDialog(usr)

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith(`${API_ROOT}/keys`) &&
        (init?.method ?? "") === "POST",
    )
    expect(JSON.parse(String(post?.[1]?.body)).user_id).toBe("alice")
  })

  it("blocks all models by posting an empty list", async () => {
    const fetchMock = mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Advanced" }))
    await user.click(screen.getByRole("button", { name: "Block all" }))
    await user.click(screen.getByRole("button", { name: "Create key" }))

    const post = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).endsWith(`${API_ROOT}/keys`) &&
        (init?.method ?? "") === "POST",
    )
    expect(JSON.parse(String(post?.[1]?.body)).allowed_models).toEqual([])
  })

  it("disables Create when 'Only selected' has no models (never a silent deny-all)", async () => {
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    // Give it an owner so the only reason Create stays disabled is the empty scope.
    await user.type(screen.getByPlaceholderText(/Pick a user/), "alice")
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Advanced" }))
    await user.click(screen.getByRole("button", { name: "Only selected" }))

    expect(screen.getByRole("button", { name: "Create key" })).toBeDisabled()
  })

  it("requires an owner before a key can be created (no anonymous virtual users)", async () => {
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    // Owner is empty: Create is blocked.
    expect(screen.getByRole("button", { name: "Create key" })).toBeDisabled()

    await user.type(screen.getByPlaceholderText(/Pick a user/), "team-checkout")
    await user.keyboard("{Escape}")
    expect(screen.getByRole("button", { name: "Create key" })).toBeEnabled()
  })

  it("frames the per-key scope as narrowing within the owner's access", async () => {
    mockApi({ keys: [] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("No API keys yet")
    await user.click(
      screen.getByRole("button", { name: "Create your first key" }),
    )
    await user.type(screen.getByPlaceholderText(/Pick a user/), "team-checkout")
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Advanced" }))

    // The "any" mode is labeled as inheritance, not unrestricted, and the owner's
    // access is surfaced for context (a new id starts unrestricted).
    expect(
      screen.getByRole("button", { name: "Inherit owner access" }),
    ).toBeInTheDocument()
    expect(screen.getByText(/starts unrestricted/)).toBeInTheDocument()
  })

  it("opens the edit form in a dialog, naming the key it is about", async () => {
    mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    // A dialog rather than a band above the table: the row keeps its place, and
    // the page under it does not shift by the height of a form (otari-ai#2125).
    const dialog = await screen.findByRole("dialog", { name: "Edit key" })
    expect(within(dialog).getByText("ci-bot")).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "Save" }),
    ).toBeInTheDocument()
  })

  it("asks before discarding an edited field", async () => {
    mockApi({ keys: [apiKey({ id: "key-1", key_name: "ci-bot" })] })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await user.type(await screen.findByLabelText("Name"), "-2")
    await user.keyboard("{Escape}")

    // The guard, not the exit: an edit form seeds from the row, so dirty has to
    // mean "differs from the key" rather than "is not empty".
    expect(screen.getByRole("button", { name: "Keep editing" })).toBeVisible()
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
  })

  it("toggles exclude_from_budget on an existing key via PATCH", async () => {
    const fetchMock = mockApi({
      keys: [
        apiKey({ id: "key-1", key_name: "ci-bot", exclude_from_budget: false }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await user.click(await screen.findByLabelText("Exempt from budget"))
    await user.click(screen.getByRole("button", { name: "Save" }))

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes(`${API_ROOT}/keys/key-1`) &&
        (init?.method ?? "") === "PATCH",
    )
    expect(JSON.parse(String(patch?.[1]?.body)).exclude_from_budget).toBe(true)
  })

  it("sets reject_user_mismatch on an existing key via PATCH", async () => {
    const fetchMock = mockApi({
      keys: [
        apiKey({ id: "key-1", key_name: "ci-bot", reject_user_mismatch: null }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await pickOption(user, "Mismatched user field", "Always accept")
    await user.click(screen.getByRole("button", { name: "Save" }))

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes(`${API_ROOT}/keys/key-1`) &&
        (init?.method ?? "") === "PATCH",
    )
    expect(JSON.parse(String(patch?.[1]?.body)).reject_user_mismatch).toBe(
      false,
    )
  })

  it("clears a key's override back to inheriting via PATCH", async () => {
    const fetchMock = mockApi({
      keys: [
        apiKey({
          id: "key-1",
          key_name: "ci-bot",
          reject_user_mismatch: false,
        }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await pickOption(
      user,
      "Mismatched user field",
      "Use the deployment setting (default)",
    )
    await user.click(screen.getByRole("button", { name: "Save" }))

    const patch = fetchMock.mock.calls.find(
      ([u, init]) =>
        String(u).includes(`${API_ROOT}/keys/key-1`) &&
        (init?.method ?? "") === "PATCH",
    )
    // An explicit null is what clears the override; omitting it would leave it set.
    expect(JSON.parse(String(patch?.[1]?.body)).reject_user_mismatch).toBeNull()
  })

  it("resets the edit form when switching to a different key row", async () => {
    mockApi({
      keys: [
        apiKey({ id: "k1", key_name: "alpha" }),
        apiKey({ id: "k2", key_name: "bravo" }),
      ],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const alphaRow = (await screen.findByText("alpha")).closest("tr")!
    await user.click(within(alphaRow).getByRole("button", { name: "Edit" }))
    expect(await screen.findByLabelText("Name")).toHaveValue("alpha")
    await user.click(screen.getByRole("button", { name: "Cancel" }))

    // The next key must open on its own values; a form that survived the first
    // row would keep "alpha" and PATCH the wrong key.
    const bravoRow = screen.getByText("bravo").closest("tr")!
    await user.click(within(bravoRow).getByRole("button", { name: "Edit" }))
    expect(await screen.findByLabelText("Name")).toHaveValue("bravo")
  })

  it("clicking a row action does not also open the edit form", async () => {
    mockApi({
      keys: [apiKey({ id: "key-1", key_name: "ci-bot", is_active: true })],
    })
    const user = userEvent.setup()
    renderPage(<KeysPage />)

    const row = (await screen.findByText("ci-bot")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Disable" }))

    expect(
      screen.queryByRole("button", { name: "Save" }),
    ).not.toBeInTheDocument()
  })

  it("shows a key's access scope in its row without a misleading count", async () => {
    mockApi({
      keys: [
        apiKey({
          id: "k1",
          key_name: "scoped",
          allowed_models: ["openai:*", "openai:gpt-4o"],
        }),
        apiKey({ id: "k2", key_name: "open", allowed_models: null }),
        apiKey({ id: "k3", key_name: "locked", allowed_models: [] }),
      ],
    })
    renderPage(<KeysPage />)

    const scoped = (await screen.findByText("scoped")).closest("tr")!
    // A wildcard is many models, so the chip says "Selected models", not "2 models".
    expect(within(scoped).getByText("Selected models")).toBeInTheDocument()
    expect(
      within(screen.getByText("open").closest("tr")!).getByText("All models"),
    ).toBeInTheDocument()
    expect(
      within(screen.getByText("locked").closest("tr")!).getByText("No models"),
    ).toBeInTheDocument()
  })

  it("flags an expired key and marks a virtual owner", async () => {
    mockApi({
      keys: [
        apiKey({
          id: "key-1",
          key_name: "old",
          is_active: true,
          expires_at: "2020-01-01T00:00:00+00:00",
          user_id: "apikey-abcdef",
        }),
      ],
    })
    renderPage(<KeysPage />)

    const row = (await screen.findByText("old")).closest("tr")!
    expect(within(row).getByText("Expired")).toBeInTheDocument()
    expect(within(row).getByText("virtual")).toBeInTheDocument()
  })

  it("names a key's owner from the roster when that owner is a member", async () => {
    const uuid = "33333333-3333-3333-3333-333333333333"
    mockApi({
      keys: [apiKey({ id: "key-1", key_name: "alice-laptop", user_id: uuid })],
      users: [user({ user_id: uuid, alias: "alice@example.com" })],
      members: [
        organizationMember({
          attribution_user_id: uuid,
          full_name: null,
          email: "alice@example.com",
        }),
      ],
    })
    renderPage(<KeysPage />)

    const row = (await screen.findByText("alice-laptop")).closest("tr")!
    // The person, not the UUID their identity was minted under.
    expect(
      await within(row).findByText("alice@example.com"),
    ).toBeInTheDocument()
    expect(within(row).queryByText(uuid)).not.toBeInTheDocument()
  })

  it("leaves an owner nobody named as the id it always was", async () => {
    mockApi({
      keys: [apiKey({ id: "key-1", key_name: "ci", user_id: "ci-bot" })],
      users: [user({ user_id: "ci-bot", alias: null })],
      members: [],
    })
    renderPage(<KeysPage />)

    // No roster entry claims `ci-bot`, and a hand-made id is already the
    // readable form, so the column is unchanged from before members existed.
    const row = (await screen.findByText("ci")).closest("tr")!
    expect(within(row).getByText("ci-bot")).toBeInTheDocument()
  })

  // The member's view of the same page (otari-ai#1941): every hook reads and
  // writes /api/v1/organizations/me/keys, and the operator-only affordances (the
  // owner picker, the budget exemption, the Owner column, the links to pages a
  // member cannot open) are absent rather than present and refused.
  describe("as a member", () => {
    it("lists through the member surface, without the Owner column or operator links", async () => {
      const fetchMock = mockApi({
        deploymentOperator: false,
        keys: [apiKey({ id: "key-1", key_name: "mine" })],
      })
      renderPage(<KeysPage />)

      const row = (await screen.findByText("mine")).closest("tr")!
      expect(within(row).getByText("Active")).toBeInTheDocument()

      // The read went to the member surface, and never to the operator one.
      const listCalls = fetchMock.mock.calls
        .map(([u]) => String(u))
        .filter((u) => KEYS_URL.test(u))
      expect(listCalls.length).toBeGreaterThan(0)
      for (const u of listCalls) {
        expect(u).toContain(`${API_ROOT}/organizations/me/keys`)
      }

      // Every key here is the caller's own, so no Owner column; and the pages
      // the operator paragraph links to would refuse a member.
      expect(
        screen.queryByRole("columnheader", { name: "Owner" }),
      ).not.toBeInTheDocument()
      expect(
        screen.queryByRole("link", { name: /Spend & budgets/ }),
      ).not.toBeInTheDocument()
    })

    it("creates a key with no owner picker and no budget exemption", async () => {
      const fetchMock = mockApi({ deploymentOperator: false, keys: [] })
      const usr = userEvent.setup()
      renderPage(<KeysPage />)

      await screen.findByText("No API keys yet")
      await usr.click(
        screen.getByRole("button", { name: "Create your first key" }),
      )

      // No owner to pick: the key is the caller's own, and Create does not wait
      // for one.
      expect(
        screen.queryByPlaceholderText(/Pick a user/),
      ).not.toBeInTheDocument()
      expect(screen.getByRole("button", { name: "Create key" })).toBeEnabled()

      await usr.click(screen.getByRole("button", { name: "Advanced" }))
      expect(
        screen.queryByLabelText("Exempt from budget"),
      ).not.toBeInTheDocument()

      await usr.type(screen.getByPlaceholderText("ci-bot"), "my-key")
      await usr.click(screen.getByRole("button", { name: "Create key" }))
      const reveal = await screen.findByRole("alert", {
        name: /API key created|New secret for/,
      })
      expect(within(reveal).getByDisplayValue(NEW_SECRET)).toBeInTheDocument()

      const post = fetchMock.mock.calls.find(
        ([u, init]) =>
          String(u).endsWith(`${API_ROOT}/organizations/me/keys`) &&
          (init?.method ?? "") === "POST",
      )
      expect(post).toBeDefined()
      const body = JSON.parse(String(post?.[1]?.body))
      expect(body.key_name).toBe("my-key")
      // The member body carries neither escalation field.
      expect(body).not.toHaveProperty("user_id")
      expect(body).not.toHaveProperty("exclude_from_budget")
    })

    it("edits through the member surface, with no budget exemption to send", async () => {
      const fetchMock = mockApi({
        deploymentOperator: false,
        keys: [apiKey({ id: "key-1", key_name: "mine" })],
      })
      const usr = userEvent.setup()
      renderPage(<KeysPage />)

      const row = (await screen.findByText("mine")).closest("tr")!
      await usr.click(within(row).getByRole("button", { name: "Edit" }))
      expect(
        screen.queryByLabelText("Exempt from budget"),
      ).not.toBeInTheDocument()

      await usr.click(screen.getByRole("button", { name: "Save" }))

      const patch = fetchMock.mock.calls.find(
        ([u, init]) =>
          String(u).endsWith(`${API_ROOT}/organizations/me/keys/key-1`) &&
          (init?.method ?? "") === "PATCH",
      )
      expect(patch).toBeDefined()
      expect(JSON.parse(String(patch?.[1]?.body))).not.toHaveProperty(
        "exclude_from_budget",
      )
    })
  })
})
