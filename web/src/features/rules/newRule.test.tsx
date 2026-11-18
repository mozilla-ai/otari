import { Outlet } from "@tanstack/react-router"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import { NewRuleDialog } from "@/features/rules/NewRuleDialog"
import {
  buildTargets,
  emptyListWords,
  limitStatus,
  matchTargets,
  searchPlaceholder,
  targetsOfKind,
} from "@/features/rules/ruleTargets"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  apiKey,
  bootstrap,
  organization,
  organizationContext,
  organizationMember,
  organizationSpendCeiling,
  workspace,
} from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"

const ORGANIZATION = organization({ name: "Northwind" })
const PLATFORM = workspace({ id: "ws-platform", name: "Platform" })
const RESEARCH = workspace({ id: "ws-research", name: "Research" })

const place = (one: typeof PLATFORM, memberId: string) => ({
  workspace_member_id: `${one.id}-${memberId}`,
  workspace_id: one.id,
  workspace_name: one.name,
  role: "member",
})

const WEI = organizationMember({
  organization_member_id: "m-wei",
  user_id: "u-wei",
  attribution_user_id: "u-wei",
  full_name: "Wei Zhang",
  email: "wei.zhang@northwind.example",
  role: "member",
  workspaces: [place(PLATFORM, "m-wei")],
})
const ZAINAB = organizationMember({
  organization_member_id: "m-zainab",
  user_id: "u-zainab",
  attribution_user_id: "u-zainab",
  full_name: "Zainab Yusuf",
  email: "zainab.yusuf@northwind.example",
  role: "admin",
  workspaces: [place(PLATFORM, "m-zainab"), place(RESEARCH, "m-zainab")],
})
const SUSPENDED = organizationMember({
  organization_member_id: "m-gone",
  user_id: "u-gone",
  attribution_user_id: "u-gone",
  full_name: "Gone Person",
  email: "gone@northwind.example",
  role: "member",
  status: "suspended",
})

const CI_KEY = apiKey({
  id: "key-ci",
  key_name: "ci-bot",
  user_id: "u-zainab",
  workspace_id: PLATFORM.id,
})
const LAPTOP_KEY = apiKey({
  id: "key-laptop",
  key_name: "laptop",
  user_id: "u-wei",
  workspace_id: RESEARCH.id,
})
const DEAD_KEY = apiKey({ id: "key-dead", key_name: "old", is_active: false })

const WEI_LIMIT = organizationSpendCeiling({
  id: "c-wei",
  scope_type: "org_member",
  scope_id: "m-wei",
  max_budget: 30,
  reset_alignment: "calendar_week",
})

const SOURCES = {
  organization: ORGANIZATION,
  workspaces: [PLATFORM, RESEARCH],
  members: [WEI, ZAINAB, SUSPENDED],
  keys: [CI_KEY, LAPTOP_KEY, DEAD_KEY],
  ceilings: [WEI_LIMIT],
  now: Date.parse("2026-11-17T12:00:00Z"),
}

describe("rule targets", () => {
  const targets = buildTargets(SOURCES)
  const names = (kind: Parameters<typeof targetsOfKind>[1], query = "") =>
    matchTargets(targetsOfKind(targets, kind), query).map((one) => one.name)

  it("lists each kind apart, with only who a limit can land on", () => {
    expect(names("everyone")).toEqual(["Northwind"])
    expect(names("workspace")).toEqual(["Platform", "Research"])
    expect(names("person")).toEqual(["Wei Zhang", "Zainab Yusuf"])
    expect(names("key")).toEqual(["ci-bot", "laptop"])
  })

  it("does not offer a key past its expiry", () => {
    const expired = apiKey({
      id: "k-expired",
      key_name: "old-bot",
      is_active: true,
      expires_at: "2026-11-10T00:00:00Z",
    })
    const withExpired = buildTargets({
      ...SOURCES,
      keys: [...SOURCES.keys, expired],
    })
    expect(
      targetsOfKind(withExpired, "key").map((one) => one.name),
    ).not.toContain("old-bot")
  })

  it("keeps the scopes the limit dialog writes to", () => {
    expect(targets.map((one) => [one.scope, one.scopeId])).toEqual([
      ["organization", ORGANIZATION.id],
      ["workspace", "ws-platform"],
      ["workspace", "ws-research"],
      ["org_member", "m-wei"],
      ["org_member", "m-zainab"],
      ["api_token", "key-ci"],
      ["api_token", "key-laptop"],
    ])
  })

  it("searches a person by name or email", () => {
    expect(names("person", "zai")).toEqual(["Zainab Yusuf"])
    expect(names("person", "WEI.ZHANG@")).toEqual(["Wei Zhang"])
    expect(names("person", "admin")).toEqual([])
  })

  it("searches an API key by name or owner", () => {
    expect(names("key", "lap")).toEqual(["laptop"])
    expect(names("key", "zainab")).toEqual(["ci-bot"])
    expect(names("key", "platform")).toEqual([])
  })

  it("searches a workspace by name", () => {
    expect(names("workspace", "sea")).toEqual(["Research"])
    expect(names("workspace", "2 people")).toEqual([])
  })

  it("says who each row is from what is already loaded", () => {
    const detail = Object.fromEntries(
      targets.map((one) => [one.name, one.detail]),
    )
    expect(detail["Wei Zhang"]).toBe("wei.zhang@northwind.example · Member")
    expect(detail["ci-bot"]).toBe("Zainab Yusuf · Platform")
    expect(detail.Platform).toBe("2 people")
    expect(detail.Research).toBe("1 person")
  })

  it("leaves out a join it does not have", () => {
    const [key] = targetsOfKind(
      buildTargets({ ...SOURCES, members: [], workspaces: [] }),
      "key",
    )
    expect(key?.detail).toBe("")
  })

  it("names the limit a target already has, or the lack of one", () => {
    const [wei, zainab] = targetsOfKind(targets, "person")
    expect(wei && limitStatus(wei)).toBe("Limit $30.00 a week")
    expect(zainab && limitStatus(zainab)).toBe("No limit yet")
    expect(wei && limitStatus(wei, "failed")).toBe("Could not load limits")
  })

  it("words the search and its empty list for the kind", () => {
    expect(searchPlaceholder("person", 12, "ready")).toBe(
      "Search 12 people by name or email",
    )
    expect(searchPlaceholder("key", 1, "ready")).toBe(
      "Search 1 API key by name or owner",
    )
    expect(searchPlaceholder("workspace", 0, "loading")).toBe(
      "Loading workspaces",
    )
    expect(searchPlaceholder("person", 0, "ready")).toBe("No people yet")
    expect(searchPlaceholder("person", 0, "failed")).toBe(
      "Could not load people",
    )
    expect(emptyListWords("person", " x ", "ready", false)).toBe(
      "No people match “x”",
    )
    expect(emptyListWords("key", "", "loading", true)).toBe("Loading API keys")
    expect(emptyListWords("person", "", "failed", true)).toBe(
      "Could not load people",
    )
  })
})

/**
 * The dialog over the sources above. `failing` answers that read with a 403,
 * which the client does not retry; `pending` never answers it.
 */
function renderDialog({
  failing,
  pending,
}: {
  failing?: string
  pending?: string
} = {}) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const path = String(input).slice(API_ROOT.length).split("?")[0]
    const paged = (data: unknown[]) =>
      Response.json({ data, count: data.length })
    if (path === failing) {
      return Response.json({ detail: `Refused ${failing}` }, { status: 403 })
    }
    if (path === pending) return new Promise<Response>(() => {})
    switch (path) {
      case "/organizations/me":
        return Response.json(
          organizationContext({
            organization: ORGANIZATION,
            deployment_operator: false,
          }),
        )
      case "/organizations/me/members":
        return paged(SOURCES.members)
      case "/workspaces":
        return paged(SOURCES.workspaces)
      case "/organizations/me/keys":
        return Response.json(SOURCES.keys)
      case "/organizations/me/spend-ceilings":
        return paged(SOURCES.ceilings)
      default:
        return paged([])
    }
  })
  return renderWithRouter(<NewRuleDialog onClose={() => {}} />, {
    url: "/rules",
    shell: (
      <Provider>
        <DeploymentProvider value={bootstrap()}>
          <SelectedWorkspaceProvider>
            <Outlet />
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </Provider>
    ),
  })
}

const kind = (name: string) => screen.getByRole("radio", { name })
const submit = (name: string) => screen.getByRole("button", { name })

describe("New rule, Applies to", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("opens with no kind picked and the second field held, disabled", async () => {
    await renderDialog()
    await screen.findByRole("radiogroup", { name: "Applies to" })
    for (const name of ["Everyone", "Workspace", "Person", "API key"]) {
      expect(kind(name)).not.toBeChecked()
    }
    expect(screen.getByRole("combobox", { name: "Who" })).toBeDisabled()
    expect(submit("Next")).toBeDisabled()
  })

  it("says where the rules it cannot make are made today", async () => {
    await renderDialog()
    await screen.findByRole("radiogroup", { name: "Applies to" })
    expect(
      screen.getByText(/Model access is set on Members & roles/),
    ).toBeInTheDocument()
    expect(screen.queryByText(/would be/)).toBeNull()
  })

  it("enters the empty kind group at Everyone and arrows on from there", async () => {
    const user = userEvent.setup()
    await renderDialog()
    await screen.findByRole("radiogroup", { name: "Applies to" })
    for (let step = 0; step < 6; step++) {
      if (document.activeElement?.getAttribute("type") === "radio") break
      await user.tab()
    }
    expect(kind("Everyone")).toHaveFocus()
    // Focus alone picks nothing.
    for (const name of ["Everyone", "Workspace", "Person", "API key"]) {
      expect(kind(name)).not.toBeChecked()
    }

    await user.keyboard("{ArrowRight}")
    expect(kind("Workspace")).toBeChecked()
    await user.keyboard("{ArrowDown}")
    expect(kind("Person")).toBeChecked()
    // Arrowing back onto Everyone picks it, as in any radio group.
    await user.keyboard("{ArrowLeft}{ArrowLeft}")
    expect(kind("Everyone")).toBeChecked()
    // Next waits for the limits already set.
    await waitFor(() => expect(submit("Next")).toBeEnabled())
  })

  it("picks Everyone with Space from the first stop", async () => {
    const user = userEvent.setup()
    await renderDialog()
    await screen.findByRole("radiogroup", { name: "Applies to" })
    kind("Everyone").focus()
    await user.keyboard(" ")
    expect(kind("Everyone")).toBeChecked()
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument()
  })

  it("lists only the picked kind, and finds a person by email", async () => {
    const user = userEvent.setup()
    await renderDialog()
    await user.click(await screen.findByRole("radio", { name: "Person" }))
    const field = screen.getByRole("combobox", { name: "Person" })
    await waitFor(() =>
      expect(field).toHaveAttribute(
        "placeholder",
        "Search 2 people by name or email",
      ),
    )
    await user.click(field)
    expect(await screen.findAllByRole("option")).toHaveLength(2)
    expect(
      screen.getByRole("option", {
        name: "Wei Zhang, wei.zhang@northwind.example · Member, Limit $30.00 a week",
      }),
    ).toBeInTheDocument()

    await user.type(field, "zainab.yusuf@")
    await waitFor(() => expect(screen.getAllByRole("option")).toHaveLength(1))
    expect(screen.getByRole("option")).toHaveTextContent("Zainab Yusuf")

    await user.clear(field)
    await user.type(field, "nobody")
    expect(
      await screen.findByText("No people match “nobody”"),
    ).toBeInTheDocument()
  })

  it("reads Edit limit once the chosen target already has a limit", async () => {
    const user = userEvent.setup()
    await renderDialog()
    await user.click(await screen.findByRole("radio", { name: "Person" }))
    const field = screen.getByRole("combobox", { name: "Person" })
    await user.click(field)
    await user.click(await screen.findByRole("option", { name: /Zainab/ }))
    expect(field).toHaveValue("Zainab Yusuf")
    await waitFor(() => expect(submit("Next")).toBeEnabled())
    // The limit is its own element, so it stays whole where the line is cut.
    expect(field).toHaveAccessibleDescription(
      /^zainab\.yusuf@northwind\.example · Admin\s*·\s*No limit yet$/,
    )

    await user.clear(field)
    await user.click(await screen.findByRole("option", { name: /Wei Zhang/ }))
    expect(submit("Edit limit")).toBeEnabled()
    expect(field).toHaveAccessibleDescription(
      /^wei\.zhang@northwind\.example · Member\s*·\s*Limit \$30\.00 a week$/,
    )
  })

  it("clears the choice when the kind changes", async () => {
    const user = userEvent.setup()
    await renderDialog()
    await user.click(await screen.findByRole("radio", { name: "Person" }))
    await user.click(screen.getByRole("combobox", { name: "Person" }))
    await user.click(await screen.findByRole("option", { name: /Wei Zhang/ }))
    expect(submit("Edit limit")).toBeEnabled()

    await user.click(kind("API key"))
    expect(screen.getByRole("combobox", { name: "API key" })).toHaveValue("")
    expect(submit("Next")).toBeDisabled()

    // Back on the first kind, the earlier pick has not been kept.
    await user.click(kind("Person"))
    expect(screen.getByRole("combobox", { name: "Person" })).toHaveValue("")
    expect(submit("Next")).toBeDisabled()
  })

  it("holds Next until the limits already set are read", async () => {
    const user = userEvent.setup()
    await renderDialog({ pending: "/organizations/me/spend-ceilings" })
    await user.click(await screen.findByRole("radio", { name: "Everyone" }))
    expect(screen.getByText("Covers everyone in Northwind.")).toBeVisible()
    expect(submit("Next")).toBeDisabled()
  })

  it("says the limits already set could not be read, and holds Next", async () => {
    const user = userEvent.setup()
    await renderDialog({ failing: "/organizations/me/spend-ceilings" })
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Refused /organizations/me/spend-ceilings",
    )
    await user.click(screen.getByRole("radio", { name: "Everyone" }))
    expect(submit("Next")).toBeDisabled()
  })

  it("says the people could not be read", async () => {
    await renderDialog({ failing: "/organizations/me/members" })
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Refused /organizations/me/members",
    )
  })

  it("does not call the people none when they could not be read", async () => {
    const user = userEvent.setup()
    await renderDialog({ failing: "/organizations/me/members" })
    await screen.findByRole("alert")
    await user.click(screen.getByRole("radio", { name: "Person" }))
    const field = screen.getByRole("combobox", { name: "Person" })
    expect(field).toHaveAttribute("placeholder", "Could not load people")
    await user.click(field)
    expect(await screen.findByText("Could not load people")).toBeVisible()
    expect(screen.queryByText(/No people/)).toBeNull()
  })

  it("does not say a target has no limit when the limits could not be read", async () => {
    const user = userEvent.setup()
    await renderDialog({ failing: "/organizations/me/spend-ceilings" })
    await screen.findByRole("alert")
    await user.click(screen.getByRole("radio", { name: "Everyone" }))
    expect(screen.getByText("Could not load limits")).toBeVisible()

    await user.click(screen.getByRole("radio", { name: "Person" }))
    const field = screen.getByRole("combobox", { name: "Person" })
    await user.click(field)
    await user.click(await screen.findByRole("option", { name: /Zainab/ }))
    expect(field).toHaveAccessibleDescription(
      /^zainab\.yusuf@northwind\.example · Admin\s*·\s*Could not load limits$/,
    )
    expect(screen.queryByText(/No limit yet/)).toBeNull()
  })

  it("takes Everyone as the whole organization, with no list", async () => {
    const user = userEvent.setup()
    await renderDialog()
    await user.click(await screen.findByRole("radio", { name: "Everyone" }))
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument()
    expect(
      screen.getByText("Covers everyone in Northwind."),
    ).toBeInTheDocument()
    await waitFor(() => expect(submit("Next")).toBeEnabled())
  })

  it("hands the limit dialog the chosen target", async () => {
    const user = userEvent.setup()
    await renderDialog()
    await user.click(await screen.findByRole("radio", { name: "API key" }))
    await user.click(screen.getByRole("combobox", { name: "API key" }))
    await user.click(await screen.findByRole("option", { name: /ci-bot/ }))
    await user.click(submit("Next"))
    expect(
      await screen.findByRole("heading", { name: "Spend limit on ci-bot" }),
    ).toBeInTheDocument()
  })
})
