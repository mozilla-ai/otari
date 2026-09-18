import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { KeysPage } from "@/features/keys/KeysPage"
import { API_ROOT } from "@/shared/api/client"
import { apiKey, organizationMember } from "@/tests/fixtures"
import { mockApi, renderPage, submitTheCreateDialog, user } from "@/tests/keys"
import { pickOption } from "@/tests/select"

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe("KeysPage creation", () => {
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

  it("posts the picked owner's user_id, not the option's display label", async () => {
    // Regression: picking an existing owner used to submit the option's label
    // ("alice (Alice)") because selecting writes that text back into the input,
    // re-firing onInputChange. The keys API does not know that id, so it silently
    // created a second user aliased "User alice (Alice)" instead of reusing alice.
    const fetchMock = mockApi({
      // The key is what puts alice in this organization, and so in the picker
      // at all (otari-ai#2108).
      keys: [apiKey({ id: "key-1", key_name: "existing", user_id: "alice" })],
      users: [user({ user_id: "alice", alias: "Alice" })],
    })
    const usr = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("existing")
    await usr.click(screen.getByRole("button", { name: "Create key" }))
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

  it("offers this organization's users as owners, not the deployment's", async () => {
    // `/api/v1/users` is deployment-wide, so on a deployment holding several
    // tenants it answered with every tenant's people and the picker offered
    // them all (otari-ai#2108).
    const member = "33333333-3333-3333-3333-333333333333"
    const otherTenant = "44444444-4444-4444-4444-444444444444"
    mockApi({
      keys: [apiKey({ id: "key-1", key_name: "ci", user_id: "ci-bot" })],
      users: [
        user({ user_id: member, alias: "alice@example.com" }),
        user({ user_id: "ci-bot", alias: null }),
        user({ user_id: otherTenant, alias: "someone@other.example" }),
      ],
      members: [
        organizationMember({
          attribution_user_id: member,
          full_name: "Alice Example",
        }),
      ],
    })
    const usr = userEvent.setup()
    renderPage(<KeysPage />)

    await screen.findByText("ci")
    await usr.click(screen.getByRole("button", { name: "Create key" }))
    await usr.click(screen.getByPlaceholderText(/Pick a user/))

    // On the roster, and the owner of a key in this organization.
    expect(
      await screen.findByRole("option", { name: /Alice Example/ }),
    ).toBeInTheDocument()
    expect(screen.getByRole("option", { name: "ci-bot" })).toBeInTheDocument()
    // Neither: another organization's person, and nothing here names them.
    expect(
      screen.queryByRole("option", { name: /other\.example/ }),
    ).not.toBeInTheDocument()
    expect(screen.queryByText(otherTenant)).not.toBeInTheDocument()
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
})
