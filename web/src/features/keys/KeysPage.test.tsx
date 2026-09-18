import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { KeysPage } from "@/features/keys/KeysPage"
import { apiKey, organizationMember } from "@/tests/fixtures"
import {
  KEYS_URL,
  mockApi,
  NEW_SECRET,
  renderPage,
  stubRegionWidth,
  user,
} from "@/tests/keys"

afterEach(() => {
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe("KeysPage", () => {
  it("keeps selection, full identity, and prefix copying available in the mobile list", async () => {
    stubRegionWidth(390)
    const name = "prod-gateway-eu-west-1-primary-ingress-router"
    const owner = "alexandra.constantinescu@platform-engineering.example.com"
    mockApi({
      keys: [
        apiKey({ key_name: name, user_id: owner, key_prefix: "gw-prefix" }),
      ],
    })
    const user = userEvent.setup()
    const copy = vi
      .spyOn(navigator.clipboard, "writeText")
      .mockResolvedValue(undefined)
    renderPage(<KeysPage />)
    await screen.findByRole("button", { name: `Actions for ${name}` })
    expect(screen.queryByRole("grid")).not.toBeInTheDocument()
    await user.click(screen.getByRole("checkbox", { name: "Select all keys" }))
    expect(
      screen.getByRole("checkbox", { name: `Select ${name}` }),
    ).toBeChecked()
    await user.click(
      screen.getByRole("button", { name: `Actions for ${name}` }),
    )
    const menu = await screen.findByRole("menu")
    expect(within(menu.parentElement!).getByText(owner)).toBeInTheDocument()
    expect(screen.getByText("Created:")).toBeInTheDocument()
    expect(screen.getByRole("menuitem", { name: /^Delete/ })).toHaveAttribute(
      "aria-disabled",
      "true",
    )
    await user.keyboard("{Escape}")
    expect(
      await screen.findByRole("checkbox", { name: `Select ${name}` }),
    ).toBeChecked()
    await user.click(
      screen.getByRole("button", { name: `Copy key prefix for ${name}` }),
    )
    expect(copy).toHaveBeenCalledWith("gw-prefix")
  })

  it("keeps two faces in Owner and folds the lanes into the menu only once they leave the row", async () => {
    const member = "33333333-3333-3333-3333-333333333333"
    stubRegionWidth(1440)
    mockApi({
      keys: [
        apiKey({ id: "key-1", key_name: "named", user_id: member }),
        apiKey({ id: "key-2", key_name: "raw", user_id: "ci-bot" }),
      ],
      users: [user({ user_id: member, alias: "alice@example.com" })],
      members: [
        organizationMember({
          attribution_user_id: member,
          full_name: "Alice Example",
        }),
      ],
    })
    const usr = userEvent.setup()
    renderPage(<KeysPage />)

    // A member is a person and takes the body face; a raw id is an identifier
    // and takes the mono one, which is the only thing telling the two apart.
    const named = await screen.findByText("Alice Example")
    expect(named).toHaveClass("text-sm", "text-foreground")
    expect(named).toHaveAttribute("title", member)
    expect(screen.getByText("ci-bot", { selector: "span" })).toHaveClass(
      "text-mono-caption",
    )

    // Created, Last used and Expires are lanes here, so the menu does not
    // repeat them.
    await usr.click(screen.getByRole("button", { name: "Actions for named" }))
    await screen.findByRole("menu")
    expect(screen.queryByText("Created:")).not.toBeInTheDocument()
  })

  it("loads inside the list on a phone rather than painting the table first", async () => {
    stubRegionWidth(390)
    let release: (() => void) | undefined
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    const base = mockApi({ keys: [apiKey({ key_name: "held" })] })
    const inner = base.getMockImplementation()!
    base.mockImplementation(async (input, init) => {
      if (KEYS_URL.test(String(input))) await held
      return inner(input, init)
    })
    renderPage(<KeysPage />)

    // The seven-column skeleton is what used to appear here for the whole load.
    expect(await screen.findByText("Loading…")).toBeInTheDocument()
    expect(screen.queryByRole("grid")).not.toBeInTheDocument()

    release?.()
    await screen.findByRole("button", { name: "Actions for held" })
    expect(screen.queryByRole("grid")).not.toBeInTheDocument()
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

  it("lists key fingerprints and supports keys without stored suffixes", async () => {
    mockApi({
      keys: [
        apiKey({
          id: "key-1",
          key_name: "ci-bot",
          key_prefix: "gw-AbC3dE",
          key_suffix: "1234",
          is_active: true,
        }),
        apiKey({
          id: "key-prefix-only",
          key_name: "prefix-only",
          key_prefix: "gw-Older",
          key_suffix: null,
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
    expect(within(activeRow).getByText("gw-AbC3dE…1234")).toBeInTheDocument()

    const prefixOnlyRow = screen.getByText("prefix-only").closest("tr")!
    expect(within(prefixOnlyRow).getByText("gw-Older…")).toBeInTheDocument()

    // A key minted before the prefix existed renders "—", not a crash.
    const legacyRow = screen.getByText("legacy").closest("tr")!
    expect(within(legacyRow).getByText("Disabled")).toBeInTheDocument()
    expect(within(legacyRow).getByText("—")).toBeInTheDocument()

    expect(document.body.textContent).not.toContain(NEW_SECRET)
  })

  it("keeps the action lane's slots the same on a live row and a disabled one", async () => {
    mockApi({
      keys: [
        apiKey({ id: "key-1", key_name: "ci-bot", is_active: true }),
        apiKey({ id: "key-2", key_name: "legacy", is_active: false }),
      ],
    })
    renderPage(<KeysPage />)

    const slots = async (name: string) => {
      const row = (await screen.findByText(name)).closest("tr")!
      const lane = row.lastElementChild!.firstElementChild!
      return lane.children.length
    }

    // Delete is only offered once a key is disabled, and the lane is
    // right-aligned, so a lane one control shorter slid every glyph beside it
    // along and put Edit in a different column on each row. The slot is held
    // open instead.
    expect(await slots("ci-bot")).toBe(await slots("legacy"))
    expect(
      within((await screen.findByText("ci-bot")).closest("tr")!).queryByRole(
        "button",
        { name: "Delete" },
      ),
    ).not.toBeInTheDocument()
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
})
