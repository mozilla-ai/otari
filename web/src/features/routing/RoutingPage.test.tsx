import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useParams, useSearch } from "@tanstack/react-router"
import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext } from "@/client"
import { RoutingPage } from "@/features/routing/RoutingPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, organizationContext } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"
import {
  ADMIN_WORKSPACE,
  adminContext,
  CHAIN,
  LEARNED,
  MEMBERS,
  mockApi,
  policy,
  renderInWorkspace,
  renderPage,
  WEIGHTED,
} from "@/tests/routing"

// The list. Creating a policy is in RoutingPage.create.test.tsx, and a
// policy's own page in PolicyDetailPage.test.tsx.

afterEach(() => {
  vi.restoreAllMocks()
})

/**
 * The page's own create action, scoped to the heading's header: once the
 * dialog is open its submit says "Create policy" too.
 */
const createTrigger = async () => {
  const heading = await screen.findByRole("heading", { name: "Routing" })
  const header = heading.closest("header")
  if (!header) throw new Error("PageIntro's header is gone")
  return within(header).findByRole("button", { name: "Create policy" })
}

/** Where a row's Edit or View lands: the policy's page, with its scope. */
function PolicyProbe() {
  const { name } = useParams({ strict: false }) as { name?: string }
  const search = useSearch({ strict: false }) as { user?: string }
  return (
    <p>
      Policy page for {name}
      {search.user ? ` as ${search.user}` : ""}
    </p>
  )
}

const POLICY_PAGE = [{ path: "/routing/$name", element: <PolicyProbe /> }]

describe("RoutingPage", () => {
  it("lists each policy's main model, its backups and who it applies to", async () => {
    mockApi()
    renderPage(<RoutingPage />)

    const fastRow = (await screen.findByText("fast")).closest("tr")!
    // By model and provider rather than as a selector, read off the selector
    // itself so the table makes no catalog read per row.
    expect(within(fastRow).getByText("gpt-5-mini")).toBeInTheDocument()
    expect(within(fastRow).getByText("OpenAI")).toBeInTheDocument()
    // The backups are spelled out in order, so a fallback is visible without
    // opening the policy.
    const backups = within(fastRow).getByRole("list", { name: "Backups" })
    expect(within(backups).getAllByRole("listitem")).toHaveLength(1)
    expect(backups).toHaveTextContent("claude-haiku-4-5 · Anthropic")
    expect(within(fastRow).getByText("Every caller")).toBeInTheDocument()
  })

  it("says a policy has no backups rather than leaving the cell empty", async () => {
    mockApi([policy("solo", { select: [{ default: "openai:gpt-5" }] })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("solo")).closest("tr")!
    expect(within(row).getByText("None")).toBeInTheDocument()
    expect(
      within(row).queryByRole("list", { name: "Backups" }),
    ).not.toBeInTheDocument()
  })

  it("marks a policy that decides per request, since it has no single target", async () => {
    mockApi()
    renderPage(<RoutingPage />)

    const autoRow = (await screen.findByText("auto")).closest("tr")!
    expect(within(autoRow).getByText("DYNAMIC")).toBeInTheDocument()
    // The default target, and the condition that moves traffic off it.
    expect(within(autoRow).getByText("gpt-5-mini")).toBeInTheDocument()
    expect(
      within(autoRow).getByText(/From 80% of budget: gpt-5-nano · OpenAI/),
    ).toBeInTheDocument()
  })

  it("does not offer to edit or delete a policy that lives in config.yml", async () => {
    mockApi()
    renderPage(<RoutingPage />)

    const autoRow = (await screen.findByText("auto")).closest("tr")!
    expect(within(autoRow).getByText("set in config.yml")).toBeInTheDocument()
    expect(
      within(autoRow).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
    expect(
      within(autoRow).queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument()
    // Its page still opens, read-only.
    expect(
      within(autoRow).getByRole("button", { name: "View" }),
    ).toBeInTheDocument()
  })

  it("keeps the page's create action visible while the dialog is open", async () => {
    // It used to hide itself while the inline form was on the page. The form is
    // over the page now, so hiding the control that opened it would take the
    // heading's action away mid-task for no reason.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const trigger = await createTrigger()
    await user.click(trigger)
    await screen.findByRole("dialog")
    expect(trigger).toBeInTheDocument()
  })

  it("lists a row per user for a policy written to several of them", async () => {
    mockApi(
      [
        policy("cheap", CHAIN, { user_id: "u-bob" }),
        policy("cheap", CHAIN, { user_id: "u-carol" }),
      ],
      null,
      [],
      { members: MEMBERS },
    )
    renderPage(<RoutingPage />)

    // Two rows under one name: the scope is half the row's identity, so they do
    // not collapse into one.
    expect(
      await screen.findAllByRole("rowheader", { name: "cheap" }),
    ).toHaveLength(2)
  })

  it("opens a policy's own page from Edit", async () => {
    mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />, "/routing", POLICY_PAGE)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    expect(await screen.findByText("Policy page for fast")).toBeInTheDocument()
  })

  it("carries a scoped row's user to its page", async () => {
    mockApi([policy("cheap", CHAIN, { user_id: "u-bob" })], null, [], {
      members: MEMBERS,
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />, "/routing", POLICY_PAGE)

    const row = (await screen.findByText("cheap")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    expect(
      await screen.findByText("Policy page for cheap as u-bob"),
    ).toBeInTheDocument()
  })

  it("does not offer Edit for a policy the form would silently truncate", async () => {
    // The editor models only a `budget_used_pct.gte` condition. Offering Edit on a
    // policy built through the API with anything else would drop it on save, which
    // is worse than not offering the button.
    mockApi([
      policy("api-authored", {
        select: [
          { when: { key_id: "k-1" }, target: "openai:gpt-5-nano" },
          { default: "openai:gpt-5-mini" },
        ],
      }),
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("api-authored")).closest("tr")!
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
    expect(
      within(row).getByRole("button", { name: "View" }),
    ).toBeInTheDocument()
    // Delete stays available: removing a policy is never lossy.
    expect(
      within(row).getByRole("button", { name: "Delete" }),
    ).toBeInTheDocument()
  })

  it("lists stored aliases alongside policies, so nothing is unmanageable", async () => {
    // Aliases were folded into this page. If they were not listed here they would
    // be invisible and undeletable from the dashboard, since the Aliases tab is gone.
    mockApi([], "http://guardrails:8000", [
      {
        name: "legacy",
        target: "openai:gpt-4o-mini",
        source: "stored",
        user_id: null,
      },
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("legacy")).closest("tr")!
    expect(within(row).getByText("gpt-4o-mini")).toBeInTheDocument()
    expect(within(row).getByText("alias")).toBeInTheDocument()
    expect(
      within(row).getByRole("button", { name: "Delete" }),
    ).toBeInTheDocument()
  })

  it("names the policy in a confirm dialog before deleting it", async () => {
    // otari-ai#2110: the confirmation used to arm inside the row, where it read
    // as part of the table rather than as a decision. It is a modal now, and
    // the policy it is about has to be named in it: the row is behind the
    // backdrop, so the name on the row is no longer the operator's reference.
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))

    const dialog = await screen.findByRole("alertdialog")
    expect(within(dialog).getByText(/^fast stops resolving/)).toBeVisible()
    // Nothing is sent by opening it.
    expect(calls.some((call) => call.method === "DELETE")).toBe(false)

    await user.click(
      within(dialog).getByRole("button", { name: "Delete policy" }),
    )

    const deletes = calls.filter((call) => call.method === "DELETE")
    expect(deletes).toHaveLength(1)
    expect(deletes[0].url).toContain(`${API_ROOT}/routing/policies/fast`)
  })

  it("deletes nothing when the confirm dialog is cancelled", async () => {
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))
    const dialog = await screen.findByRole("alertdialog")
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))

    expect(calls.some((call) => call.method === "DELETE")).toBe(false)
    expect(screen.getByText("fast")).toBeInTheDocument()
  })

  it("reports a refused delete inside the dialog, leaving the row", async () => {
    // The page banner no longer carries this: the operator is looking at the
    // modal, and a message behind the backdrop is a message they do not read.
    mockApi([policy("fast", CHAIN)], "http://guardrails:8000", [], {
      deleteBody: { status: 409, detail: "fast is referenced by an alias" },
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))
    const dialog = await screen.findByRole("alertdialog")
    await user.click(
      within(dialog).getByRole("button", { name: "Delete policy" }),
    )

    expect(
      await within(dialog).findByText(/referenced by an alias/),
    ).toBeVisible()
    // Still open, so the operator can retry or back out rather than being
    // returned to a table that looks unchanged for no stated reason.
    expect(screen.getByRole("alertdialog")).toBeInTheDocument()
    expect(screen.getByText("fast")).toBeInTheDocument()
  })

  it("does not greet the next row's confirm with the last row's refusal", async () => {
    // The mutation holds its error until the next call, and the dialog reads it,
    // so without clearing it on close the second row opens already reporting a
    // failure that was about the first.
    mockApi([policy("fast", CHAIN), policy("smart", LEARNED)], undefined, [], {
      deleteBody: { status: 409, detail: "fast is referenced by an alias" },
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const fast = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(fast).getByRole("button", { name: "Delete" }))
    const first = await screen.findByRole("alertdialog")
    await user.click(
      within(first).getByRole("button", { name: "Delete policy" }),
    )
    await within(first).findByText(/referenced by an alias/)
    await user.click(within(first).getByRole("button", { name: "Cancel" }))

    const smart = screen.getByText("smart").closest("tr")!
    await user.click(within(smart).getByRole("button", { name: "Delete" }))

    const second = await screen.findByRole("alertdialog")
    expect(within(second).getByText(/^smart stops resolving/)).toBeVisible()
    expect(within(second).queryByText(/referenced by an alias/)).toBeNull()
  })

  it("deletes an alias through the alias endpoint, not the policy one", async () => {
    const { calls } = mockApi([], "http://guardrails:8000", [
      {
        name: "legacy",
        target: "openai:gpt-4o-mini",
        source: "stored",
        user_id: null,
      },
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("legacy")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Delete" }))
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Delete alias",
      }),
    )

    const deletes = calls.filter((call) => call.method === "DELETE")
    expect(deletes).toHaveLength(1)
    // An alias still lives in model_aliases; deleting it as a policy would 404 and
    // leave the row in place.
    expect(deletes[0].url).toContain(`${API_ROOT}/aliases/legacy`)
  })

  it("summarises a learned policy by its pool rather than as an opaque dynamic row", async () => {
    // "Chosen per request" is true of a tier-down too. What an operator needs to
    // see here is that a router picks between named models, and which one serves
    // when it declines.
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    expect(
      within(row).getByText(
        /Learned . 2 candidates, gpt-5 · OpenAI by default/,
      ),
    ).toBeInTheDocument()
  })

  it("summarises a weighted policy by its split, not by its pool size", async () => {
    // Two provider:model strings do not fit the cell, and the shares are what tells
    // one weighted policy from another at a glance.
    mockApi([policy("balanced", WEIGHTED, { is_dynamic: true })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    expect(within(row).getByText("WEIGHTED")).toBeInTheDocument()
    expect(
      within(row).getByText(/70% \/ 30% across 2 models/),
    ).toBeInTheDocument()
  })

  it("names a router backend it does not know without claiming it learns", async () => {
    // Only "knn" learns. Labelling every other backend "Learned" would make the table
    // lie about the first backend added after this line was written.
    mockApi([
      policy("future", {
        select: [
          {
            router: "cheapest",
            candidates: ["openai:gpt-5-nano", "openai:gpt-5"],
          },
          { default: "openai:gpt-5" },
        ],
      }),
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("future")).closest("tr")!
    expect(within(row).getByText("ROUTED")).toBeInTheDocument()
    expect(within(row).queryByText("Learned")).not.toBeInTheDocument()
  })

  it("does not offer the examples panel for a weighted policy, which learns nothing", async () => {
    mockApi([policy("balanced", WEIGHTED, { is_dynamic: true })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    expect(
      within(row).queryByRole("button", { name: "Examples" }),
    ).not.toBeInTheDocument()
    expect(
      within(row).getByRole("button", { name: "Edit" }),
    ).toBeInTheDocument()
  })

  it("does not offer Edit for a weighted policy with no split to show", async () => {
    // The form would have to invent the shares, and the API refuses such a spec
    // anyway, so this one is only reachable as an older or hand-written document.
    mockApi([
      policy("legacy", {
        select: [
          {
            router: "weighted",
            candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
          },
          { default: "openai:gpt-5" },
        ],
      }),
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("legacy")).closest("tr")!
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
  })

  it("does not offer Edit for a router backend the form cannot write", async () => {
    mockApi([
      policy("future", {
        select: [
          {
            router: "cheapest",
            candidates: ["openai:gpt-5-nano", "openai:gpt-5"],
          },
          { default: "openai:gpt-5" },
        ],
      }),
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("future")).closest("tr")!
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
  })

  it("offers the examples panel only on a policy that actually uses a router", async () => {
    // Readiness is a per-policy question, so it belongs on the row like Edit does
    // rather than in a panel that is always on the page.
    mockApi([
      policy("smart", LEARNED, { is_dynamic: true }),
      policy("fast", CHAIN),
    ])
    renderPage(<RoutingPage />)

    const learnedRow = (await screen.findByText("smart")).closest("tr")!
    const plainRow = (await screen.findByText("fast")).closest("tr")!
    expect(
      within(learnedRow).getByRole("button", { name: "Examples" }),
    ).toBeInTheDocument()
    expect(
      within(plainRow).queryByRole("button", { name: "Examples" }),
    ).not.toBeInTheDocument()
    // Nothing about learned routing is on the page until asked for.
    expect(screen.queryByText(/Whose memory/)).not.toBeInTheDocument()
  })

  it("offers the examples panel for a config.yml policy, which cannot be edited", async () => {
    // Reading readiness is safe for a policy this page cannot change, and without it
    // a config-defined learned policy would be entirely opaque here.
    mockApi([policy("smart", LEARNED, { is_dynamic: true, source: "config" })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    expect(
      within(row).getByRole("button", { name: "Examples" }),
    ).toBeInTheDocument()
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
    expect(within(row).getByText("set in config.yml")).toBeInTheDocument()
  })

  it("names the pool and what serves when the router declines", async () => {
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))

    expect(
      await screen.findByText(/ranks openai:gpt-5-nano, openai:gpt-5/),
    ).toBeInTheDocument()
    expect(screen.getByText(/serves whenever it declines/)).toBeInTheDocument()
    // The honest empty state: no user picked yet, so no warmth claim.
    expect(screen.getByText(/Pick a user to see how warm/)).toBeInTheDocument()
  })

  it("reports each pool's warmth for the chosen user, since memory is per user", async () => {
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))
    await user.type(
      screen.getByRole("combobox", { name: /whose memory/i }),
      "alice",
    )
    await user.keyboard("{Escape}")

    expect(await screen.findByText("6 / 20 examples")).toBeInTheDocument()
    expect(screen.getByText("WARMING UP")).toBeInTheDocument()
    // A task partition warms on its own, so it gets its own line.
    expect(screen.getByText("summaries")).toBeInTheDocument()
    expect(screen.getByText("21 / 20 examples")).toBeInTheDocument()
    expect(screen.getByText("ROUTING")).toBeInTheDocument()
  })

  it("says where examples come from instead of offering to collect them", async () => {
    // Recording examples is an API job in this release. The panel has to say so, or
    // an operator reads "0 examples" as a bug with no next step.
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))

    expect(
      await screen.findByText(/POST \/api\/v1\/routing\/preferences\/rank/),
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /teach it/i })).toBeInTheDocument()
    // No write affordance anywhere in it.
    expect(
      screen.queryByRole("button", { name: /ask all/i }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /record these scores/i }),
    ).not.toBeInTheDocument()
  })

  it("does not ask whose memory for a user-scoped policy", async () => {
    // A policy scoped to one user can only use that user's memory, so asking would
    // be a question with one answer, and a wrong answer would be accepted.
    mockApi([policy("smart", LEARNED, { is_dynamic: true, user_id: "alice" })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Examples" }))

    expect(
      screen.queryByRole("combobox", { name: /whose memory/i }),
    ).not.toBeInTheDocument()
    expect(await screen.findByText("6 / 20 examples")).toBeInTheDocument()
  })

  it("refuses to edit a policy whose router sits before its conditions", async () => {
    // Selection is order-sensitive server-side and the form always re-emits
    // conditions first, so editing this spec would silently change what it does.
    mockApi([
      policy("api-authored", {
        select: [
          { router: "knn", candidates: ["openai:gpt-5-nano", "openai:gpt-5"] },
          {
            when: { budget_used_pct: { gte: 80 } },
            target: "openai:gpt-5-nano",
          },
          { default: "openai:gpt-5" },
        ],
      }),
    ])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("api-authored")).closest("tr")!
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
    expect(
      within(row).getByRole("button", { name: "View" }),
    ).toBeInTheDocument()
    // Reading its readiness is still fine.
    expect(
      within(row).getByRole("button", { name: "Examples" }),
    ).toBeInTheDocument()
  })

  it("lists the tenant-scoped policies read-only for a non-operator", async () => {
    // The member half of otari-ai#1942: the page reads
    // /v1/organizations/me/routing-policies and offers nothing that writes.
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
      memberPolicies: [policy("fast", CHAIN)],
    })
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    expect(within(row).getByText("gpt-5-mini")).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "New policy" }),
    ).not.toBeInTheDocument()
    expect(
      within(row).queryByRole("button", { name: "Edit" }),
    ).not.toBeInTheDocument()
    expect(
      within(row).queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument()
    // The Examples panel reads the operator-only /v1/routing/status, so its
    // opener goes with the rest of the actions column.
    expect(
      within(row).queryByRole("button", { name: "Examples" }),
    ).not.toBeInTheDocument()
  })

  it("keeps cached operator rows out of a member's table", async () => {
    // A disabled query still serves whatever sits under its key, so a caller
    // demoted mid-session would otherwise keep seeing the deployment-wide
    // policies and aliases they fetched as an operator. The pre-seeded client
    // stands in for that cache.
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    client.setQueryData(["routing-policies"], [policy("operator-only", CHAIN)])
    client.setQueryData(
      ["aliases"],
      [
        {
          name: "operator-alias",
          target: "openai:gpt-5",
          source: "stored",
          user_id: null,
        },
      ],
    )
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
      memberPolicies: [policy("mine", CHAIN)],
    })
    render(
      <DeploymentProvider value={bootstrap()}>
        <QueryClientProvider client={client}>
          <RoutingPage />
        </QueryClientProvider>
      </DeploymentProvider>,
      { wrapper: withRouter({ url: "/" }) },
    )

    expect(await screen.findByText("mine")).toBeInTheDocument()
    expect(screen.queryByText("operator-only")).not.toBeInTheDocument()
    expect(screen.queryByText("operator-alias")).not.toBeInTheDocument()
  })

  it("never fires the operator reads for a non-operator", async () => {
    const { calls } = mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
      memberPolicies: [policy("fast", CHAIN)],
    })
    renderPage(<RoutingPage />)
    await screen.findByText("fast")

    const urls = calls.map((call) => call.url)
    expect(
      urls.some((url) => url.endsWith(`${API_ROOT}/routing/policies`)),
    ).toBe(false)
    expect(urls.some((url) => url.includes(`${API_ROOT}/aliases`))).toBe(false)
    expect(urls.some((url) => url.includes(`${API_ROOT}/tool-settings`))).toBe(
      false,
    )
    expect(urls.some((url) => url.includes(`${API_ROOT}/users`))).toBe(false)
  })

  it("withholds the deep-linked add form from a member", async () => {
    // ?target= seeds `adding` before the membership context settles, so the
    // role has to be applied at render time. Without that a member on this URL
    // gets a create form whose only outcome is a refusal, and the read-only
    // empty state is suppressed behind it.
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
      memberPolicies: [],
    })
    renderPage(<RoutingPage />, "/routing?target=openai:gpt-4o")

    expect(
      await screen.findByText(/once your organization's admins define them/),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /Create policy/i }),
    ).not.toBeInTheDocument()
  })

  it("tells a member with no policies who defines them", async () => {
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
      }),
      memberPolicies: [],
    })
    renderPage(<RoutingPage />)

    expect(
      await screen.findByText(/once your organization's admins define them/),
    ).toBeInTheDocument()
    // The operator's numbered getting-started walkthrough is not for them.
    expect(screen.queryByText(/Create a policy/)).not.toBeInTheDocument()
  })
})

// otari-ai#2087: the page showed an operator every tenant's stored rows and an
// admin every workspace of their organization's, while resolution is scoped to
// one workspace. Both reads and both writes name the selected workspace now,
// and the surface they land on is still the caller's role.
describe("RoutingPage scoped to the selected workspace", () => {
  const OPERATOR_WORKSPACE = ADMIN_WORKSPACE

  function operatorInWorkspace(): OrganizationContext {
    return organizationContext({
      workspace_memberships: [
        { workspace_id: OPERATOR_WORKSPACE, name: "Alpha one", role: "owner" },
      ],
    })
  }

  it("names the selected workspace on the operator's list read", async () => {
    const { calls } = mockApi([policy("fast", CHAIN)], null, [], {
      context: operatorInWorkspace(),
    })
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("fast")
    const listed = calls.filter(
      (call) =>
        call.method === "GET" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    expect(listed.length).toBeGreaterThan(0)
    for (const call of listed) {
      expect(call.url).toContain(`workspace_id=${OPERATOR_WORKSPACE}`)
    }
  })

  it("names the selected workspace on an admin's list read", async () => {
    const { calls } = mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [policy("tenant-fast", CHAIN)],
    })
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("tenant-fast")
    const listed = calls.filter(
      (call) =>
        call.method === "GET" &&
        call.url.includes(`${API_ROOT}/organizations/me/routing-policies`),
    )
    expect(listed.length).toBeGreaterThan(0)
    for (const call of listed) {
      expect(call.url).toContain(`workspace_id=${ADMIN_WORKSPACE}`)
    }
  })

  it("names the row's workspace on an operator's delete", async () => {
    const OTHER_WORKSPACE = "66666666-6666-6666-6666-666666666666"
    const { calls } = mockApi(
      [policy("doomed", CHAIN, { workspace_id: OTHER_WORKSPACE })],
      null,
      [],
      { context: operatorInWorkspace() },
    )
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("doomed")
    await user.click(screen.getByRole("button", { name: "Delete" }))
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Delete policy",
      }),
    )

    const deleted = calls.find((call) => call.method === "DELETE")
    expect(deleted?.url).toContain(`${API_ROOT}/routing/policies/doomed`)
    expect(deleted?.url).toContain(`workspace_id=${OTHER_WORKSPACE}`)
  })
})

// otari-ai#1969: the Build pages are Edit for admins. An organization admin
// writes the tenant-scoped routers, which name the workspace and take no user
// scope; an operator keeps the deployment-wide ones, unchanged above.
describe("RoutingPage for an organization admin", () => {
  it("deletes through the tenant-scoped router, naming the workspace", async () => {
    const { calls } = mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [policy("doomed", CHAIN)],
    })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("doomed")
    await user.click(screen.getByRole("button", { name: "Delete" }))
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Delete policy",
      }),
    )

    const deleted = calls.find((call) => call.method === "DELETE")
    expect(deleted?.url).toContain(
      `${API_ROOT}/organizations/me/routing-policies/`,
    )
    expect(deleted?.url).toContain(`workspace_id=${ADMIN_WORKSPACE}`)
  })

  it("lists the tenant-scoped aliases beside the policies", async () => {
    mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [policy("mine", CHAIN)],
      memberAliases: [
        {
          name: "tenant-alias",
          target: "openai:gpt-5",
          source: "stored",
          user_id: null,
        },
      ],
    })
    renderInWorkspace(<RoutingPage />)

    expect(await screen.findByText("mine")).toBeInTheDocument()
    expect(screen.getByText("tenant-alias")).toBeInTheDocument()
  })

  it("deletes an alias from the alias's own workspace", async () => {
    const OTHER_WORKSPACE = "77777777-7777-7777-7777-777777777777"
    const { calls } = mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [],
      memberAliases: [
        {
          name: "doomed-alias",
          target: "openai:gpt-5",
          source: "stored",
          user_id: null,
          workspace_id: OTHER_WORKSPACE,
        },
      ],
    })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("doomed-alias")
    await user.click(screen.getByRole("button", { name: "Delete" }))
    await user.click(
      within(await screen.findByRole("alertdialog")).getByRole("button", {
        name: "Delete alias",
      }),
    )

    const deleted = calls.find((call) => call.method === "DELETE")
    expect(deleted?.url).toContain(`${API_ROOT}/organizations/me/aliases/`)
    expect(deleted?.url).toContain(`workspace_id=${OTHER_WORKSPACE}`)
  })

  it("withholds the write affordances from an admin in no workspace", async () => {
    // The switcher is seeded from the caller's own memberships, so an admin who
    // joined none has no workspace to scope a write to and gets the read-only
    // page rather than a form whose only outcome is a 422.
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "admin",
      }),
      memberPolicies: [policy("mine", CHAIN)],
    })
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("mine")
    expect(
      screen.queryByRole("button", { name: "New policy" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument()
  })

  it("still withholds the write affordances from a member", async () => {
    mockApi([], null, [], {
      context: organizationContext({
        deployment_operator: false,
        role: "member",
        workspace_memberships: [
          { workspace_id: ADMIN_WORKSPACE, name: "Alpha one", role: "member" },
        ],
      }),
      memberPolicies: [policy("mine", CHAIN)],
    })
    renderInWorkspace(<RoutingPage />)

    await screen.findByText("mine")
    expect(
      screen.queryByRole("button", { name: "New policy" }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument()
  })
})
