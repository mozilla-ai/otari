import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { OrganizationContext, PolicySpec } from "@/client"
import { RoutingPage } from "@/features/routing/RoutingPage"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"
import {
  ADMIN_WORKSPACE,
  adminContext,
  MEMBERS,
  mockApi,
  policyWrites,
  renderInWorkspace,
  renderPage,
} from "@/tests/routing"

// Creating a policy: what it is for first, then a short form for that alone.
// The list is in RoutingPage.test.tsx and a policy's own page in
// PolicyDetailPage.test.tsx.

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

type TestUser = ReturnType<typeof userEvent.setup>

const dialog = () => within(screen.getByRole("dialog"))

/** Open the dialog from the heading and pick what the policy is for. */
async function startCreate(user: TestUser, kind = "Fallback") {
  await user.click(await createTrigger())
  await user.click(dialog().getByRole("radio", { name: kind }))
  await user.click(dialog().getByRole("button", { name: "Continue" }))
}

/** The two fields a fallback needs, with the model popover dismissed after,
 *  since an open popover aria-hides the submit. */
async function nameAndServe(
  user: TestUser,
  name: string,
  target = "openai:gpt-5-nano",
) {
  await user.type(screen.getByRole("textbox", { name: /^name$/i }), name)
  await user.type(
    screen.getByRole("combobox", { name: /^main model$/i }),
    target,
  )
  await user.keyboard("{Escape}")
}

async function typeModel(user: TestUser, name: RegExp, text: string) {
  await user.type(screen.getByRole("combobox", { name }), text)
  await user.keyboard("{Escape}")
}

/** Everything a fallback needs: a name, a main model and its one backup. */
async function fillFallback(
  user: TestUser,
  name: string,
  target = "openai:gpt-5-nano",
) {
  await nameAndServe(user, name, target)
  await typeModel(user, /^backup 1$/i, "anthropic:claude-haiku-4-5")
}

/** Choose people under "Applies to", the way an operator does. */
async function pickUsers(user: TestUser, queries: string[]) {
  await user.click(dialog().getByRole("button", { name: /^Applies to/ }))
  await user.click(dialog().getByRole("button", { name: "Specific users" }))
  for (const query of queries) {
    const search = dialog().getByLabelText("Users")
    // The primitive keeps the query after a pick, so clear it first.
    await user.clear(search)
    await user.type(search, query)
    await user.click(await screen.findByRole("option"))
  }
  await user.keyboard("{Escape}")
}

async function submitDialog(user: TestUser, label = "Create policy") {
  await user.click(dialog().getByRole("button", { name: label }))
}

function lastSpec(calls: { url: string; method: string; body: unknown }[]) {
  const writes = policyWrites(calls)
  expect(writes.length).toBeGreaterThan(0)
  return writes[writes.length - 1].spec as PolicySpec
}

describe("Creating a routing policy", () => {
  it("asks what the policy is for first, one line per choice", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    expect(screen.getByRole("dialog")).toHaveAccessibleName("New policy")
    expect(dialog().getAllByRole("radio")).toHaveLength(5)
    for (const choice of [
      "Fallback",
      "Split traffic",
      "Smart router",
      "Spill over at rate limits",
      "Tier down on budget",
    ]) {
      expect(dialog().getByRole("radio", { name: choice })).toBeInTheDocument()
    }
    expect(dialog().getByRole("radio", { name: "Fallback" })).toBeChecked()
    expect(
      dialog().getByText(/backups tried in order when it fails/),
    ).toBeInTheDocument()
  })

  it("opens a fallback with its own fields and nothing else", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    expect(screen.getByRole("dialog")).toHaveAccessibleName("New fallback")
    expect(dialog().getByText("If unavailable, use")).toBeInTheDocument()
    expect(
      dialog().getByRole("combobox", { name: /^backup 1$/i }),
    ).toBeRequired()
    expect(
      dialog().getByRole("button", { name: "Add a backup" }),
    ).toBeInTheDocument()
    expect(
      dialog().getByRole("button", { name: /^Applies to/ }),
    ).toHaveAttribute("aria-expanded", "false")
    // No section of another kind, and no fold to hide them in.
    for (const absent of [/Advanced/, /Add a guardrail/, /Add a tier-down/]) {
      expect(
        dialog().queryByRole("button", { name: absent }),
      ).not.toBeInTheDocument()
    }
    expect(
      dialog().queryByRole("radio", { name: "Split traffic" }),
    ).not.toBeInTheDocument()
  })

  it("goes back to the choice, keeping the name typed", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "cheap")
    await user.click(dialog().getByRole("button", { name: "Back" }))
    await user.click(dialog().getByRole("radio", { name: "Smart router" }))
    await user.click(dialog().getByRole("button", { name: "Continue" }))

    expect(screen.getByRole("dialog")).toHaveAccessibleName("New smart router")
    expect(screen.getByRole("textbox", { name: /^name$/i })).toHaveValue(
      "cheap",
    )
  })

  it("guards a half-built policy against a stray Escape", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "cheap")
    await user.keyboard("{Escape}")

    expect(screen.getByRole("dialog")).toHaveTextContent("Unsaved changes")
    await user.click(dialog().getByRole("button", { name: "Keep editing" }))
    expect(
      dialog().getByRole("button", { name: "Create policy" }),
    ).toBeInTheDocument()
  })

  it("creates a fallback from a name, a main model and a backup", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await nameAndServe(user, "cheap")
    // A fallback with nothing to fall back to is not one.
    expect(
      dialog().getByRole("button", { name: "Create policy" }),
    ).toBeDisabled()
    await typeModel(user, /^backup 1$/i, "anthropic:claude-haiku-4-5")
    await submitDialog(user)

    const writes = policyWrites(calls)
    expect(writes).toHaveLength(1)
    expect(writes[0].name).toBe("cheap")
    // The fallthrough is explicit and last, which is what the schema requires.
    expect(writes[0].spec).toEqual({
      select: [{ default: "openai:gpt-5-nano" }],
      on_failure: ["anthropic:claude-haiku-4-5"],
    })
  })

  it("writes exactly one unscoped policy for every caller", async () => {
    const { calls } = mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await fillFallback(user, "cheap")
    await submitDialog(user)

    const writes = policyWrites(calls)
    expect(writes).toHaveLength(1)
    expect(writes[0].user_id ?? null).toBeNull()
  })

  it("writes one policy per chosen user, each carrying its own scope", async () => {
    // A policy's key is its name plus its user, so N people are N rows.
    const { calls } = mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await fillFallback(user, "cheap")
    await pickUsers(user, ["bob", "carol"])
    await submitDialog(user)

    const writes = policyWrites(calls)
    expect(writes.map((write) => write.user_id)).toEqual(["u-bob", "u-carol"])
    for (const write of writes) {
      expect(write.name).toBe("cheap")
      expect(write.spec.select).toEqual([{ default: "openai:gpt-5-nano" }])
    }
  })

  it("shows the chosen people by name, not as bare owner ids", async () => {
    mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await fillFallback(user, "cheap")
    await pickUsers(user, ["bob"])

    expect(
      dialog().getByRole("button", { name: "Remove Bob Builder" }),
    ).toBeInTheDocument()
  })

  it("will not submit a scoped policy with nobody chosen", async () => {
    const { calls } = mockApi([], null, [], { members: MEMBERS })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await fillFallback(user, "cheap")
    await user.click(dialog().getByRole("button", { name: /^Applies to/ }))
    await user.click(dialog().getByRole("button", { name: "Specific users" }))

    expect(
      dialog().getByRole("button", { name: "Create policy" }),
    ).toBeDisabled()
    await submitDialog(user)
    expect(policyWrites(calls)).toHaveLength(0)
  })

  it("keeps the dialog open on a part-written save, naming who landed and who did not", async () => {
    const { calls } = mockApi([], null, [], {
      members: MEMBERS,
      refuseFirstWriteFor: ["u-carol"],
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await fillFallback(user, "cheap")
    await pickUsers(user, ["bob", "carol", "alice"])
    await submitDialog(user)

    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("Created for Bob Builder, alice (alice).")
    expect(alert).toHaveTextContent("Not created for u-carol (carol)")
    expect(alert).toHaveTextContent("No room for u-carol")

    // Pressing again rewrites nothing that already landed.
    await submitDialog(user)
    expect(policyWrites(calls).map((write) => write.user_id)).toEqual([
      "u-bob",
      "u-carol",
      "alice",
      "u-carol",
    ])
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("fixes who a policy applies to once a write for somebody has landed", async () => {
    mockApi([], null, [], {
      members: MEMBERS,
      refuseFirstWriteFor: ["u-carol"],
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await fillFallback(user, "cheap")
    await pickUsers(user, ["bob", "carol"])
    await submitDialog(user)
    await screen.findByRole("alert")

    expect(
      dialog().queryByRole("button", { name: "Specific users" }),
    ).not.toBeInTheDocument()
    expect(dialog().queryByLabelText("Users")).not.toBeInTheDocument()
    expect(
      dialog().getByText(/already been created/, { exact: false }),
    ).toHaveTextContent("delete it from the list")
  })

  it("writes every scope again when the payload changed after a part-written save", async () => {
    const { calls } = mockApi([], null, [], {
      members: MEMBERS,
      refuseFirstWriteFor: ["u-carol"],
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await fillFallback(user, "cheap")
    await pickUsers(user, ["bob", "carol"])
    await submitDialog(user)
    expect(await screen.findByRole("alert")).toBeInTheDocument()

    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "er")
    await submitDialog(user)

    expect(
      policyWrites(calls).map((write) => [write.name, write.user_id]),
    ).toEqual([
      ["cheap", "u-bob"],
      ["cheap", "u-carol"],
      ["cheaper", "u-bob"],
      ["cheaper", "u-carol"],
    ])
  })

  it("refuses a policy name that would shadow a real model selector", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await fillFallback(user, "openai:gpt-4o")

    expect(screen.getByText(/cannot contain/)).toBeInTheDocument()
    expect(
      dialog().getByRole("button", { name: "Create policy" }),
    ).toBeDisabled()
  })

  it("creates a budget tier-down from a main model and a cheaper one", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user, "Tier down on budget")
    expect(screen.getByRole("dialog")).toHaveAccessibleName(
      "New budget tier-down",
    )
    await nameAndServe(user, "thrifty", "openai:gpt-5")
    await typeModel(user, /^use instead$/i, "openai:gpt-5-nano")
    await submitDialog(user)

    expect(lastSpec(calls)).toEqual({
      select: [
        { when: { budget_used_pct: { gte: 80 } }, target: "openai:gpt-5-nano" },
        { default: "openai:gpt-5" },
      ],
    })
  })

  it("refuses a tier-down threshold that could never fire", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user, "Tier down on budget")
    await nameAndServe(user, "thrifty", "openai:gpt-5-mini")
    await typeModel(user, /^use instead$/i, "openai:gpt-5-nano")
    const threshold = screen.getByRole("textbox", {
      name: /budget used at least/i,
    })
    await user.clear(threshold)
    await user.type(threshold, "100")

    // The budget gate refuses a request before selection at 100%.
    expect(screen.getByText("Must be under 100.")).toBeInTheDocument()
    expect(
      dialog().getByRole("button", { name: "Create policy" }),
    ).toBeDisabled()
  })

  it("creates a weighted split", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user, "Split traffic")
    await user.type(
      screen.getByRole("textbox", { name: /^name$/i }),
      "balanced",
    )
    await typeModel(user, /model 1/i, "openai:gpt-5")
    await typeModel(user, /model 2/i, "anthropic:claude-sonnet-4-5")
    // Seeded as an even split, which the operator then skews.
    const shares = screen.getAllByRole("textbox", { name: /share/i })
    expect(shares.map((share) => (share as HTMLInputElement).value)).toEqual([
      "50",
      "50",
    ])
    await user.clear(shares[0])
    await user.type(shares[0], "70")
    await user.clear(shares[1])
    await user.type(shares[1], "30")
    expect(screen.getByText("70% of requests")).toBeInTheDocument()
    await submitDialog(user)

    expect(lastSpec(calls)).toEqual({
      select: [
        {
          router: "weighted",
          candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
          weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30 },
        },
        { default: "openai:gpt-5" },
      ],
    })
  })

  it("creates a spill-over policy in the order the operator sets", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user, "Spill over at rate limits")
    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "spill")
    await typeModel(user, /model 1/i, "openai:gpt-5")
    await typeModel(user, /model 2/i, "anthropic:claude-sonnet-4-5")
    // Order is the policy, and the opt-out mark moves with its model.
    await user.click(screen.getByRole("button", { name: "Move model 2 up" }))
    expect(
      screen.getByRole("button", { name: "Move model 1 down" }),
    ).toHaveFocus()
    await submitDialog(user)

    expect(lastSpec(calls)).toEqual({
      select: [
        {
          router: "priority",
          candidates: ["anthropic:claude-sonnet-4-5", "openai:gpt-5"],
        },
        { default: "openai:gpt-5" },
      ],
    })
  })

  it("creates a smart router with the marked model as its default", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user, "Smart router")
    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "smart")
    await typeModel(user, /model 1/i, "openai:gpt-5-nano")
    await typeModel(user, /model 2/i, "openai:gpt-5")
    await user.click(
      screen.getAllByRole("radio", { name: /serves when unsure/i })[1],
    )
    await submitDialog(user)

    expect(lastSpec(calls)).toEqual({
      select: [
        { router: "knn", candidates: ["openai:gpt-5-nano", "openai:gpt-5"] },
        { default: "openai:gpt-5" },
      ],
    })
  })

  it("will not save a pool of one, which is not a routing decision", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user, "Smart router")
    await user.type(screen.getByRole("textbox", { name: /^name$/i }), "smart")
    await typeModel(user, /model 1/i, "openai:gpt-5")
    await user.click(screen.getByRole("button", { name: "Remove model 2" }))

    expect(screen.getByText(/at least two models/i)).toBeInTheDocument()
    await submitDialog(user)
    expect(policyWrites(calls)).toHaveLength(0)
  })

  it("will not author a pool past the server's model cap", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user, "Smart router")
    // Seeded with two; three more reach the cap of 5.
    for (let i = 0; i < 3; i += 1) {
      await user.click(screen.getByRole("button", { name: "Add a model" }))
    }

    expect(screen.getByRole("button", { name: "Add a model" })).toBeDisabled()
    expect(
      screen.getByText(/dispatches at most 5 models, counting the backups/i),
    ).toBeInTheDocument()
    expect(policyWrites(calls)).toHaveLength(0)
  })

  it("reports a refused save inside the dialog, leaving the form filled", async () => {
    mockApi([], "http://guardrails:8000", [], {
      saveBody: { status: 400, detail: "cheap already names an alias" },
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await fillFallback(user, "cheap")
    await submitDialog(user)

    expect(await dialog().findByText(/already names an alias/)).toBeVisible()
    expect(screen.getByRole("textbox", { name: /^name$/i })).toHaveValue(
      "cheap",
    )
  })

  it("offers a fresh draft on each open of the create dialog", async () => {
    // Reset on the way in, not out: the frame keeps its content while it
    // animates out, so the page keys the dialog on an open counter.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await startCreate(user)
    await user.type(
      screen.getByRole("textbox", { name: /^name$/i }),
      "half-typed",
    )
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Discard" }))

    await startCreate(user)
    expect(screen.getByRole("textbox", { name: /^name$/i })).toHaveValue("")
  })

  it("does not fetch the roster until the dialog opens", async () => {
    // The dialog stays mounted while closed so its frame can play its exit.
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await screen.findByText("No routing policies yet")
    const rosterReads = () =>
      calls.filter((call) => call.url.includes(`${API_ROOT}/users`))
    expect(rosterReads()).toHaveLength(0)

    await user.click(await createTrigger())
    await waitFor(() => expect(rosterReads().length).toBeGreaterThan(0))
  })
})

describe("Creating from the empty state", () => {
  const emptyState = async () =>
    (
      await screen.findByRole("heading", { name: "No routing policies yet" })
    ).closest("div")!.parentElement!

  it("opens a fallback directly, and the other kinds one press away", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(
      within(await emptyState()).getByRole("button", {
        name: "Create fallback",
      }),
    )
    expect(await screen.findByRole("dialog")).toHaveAccessibleName(
      "New fallback",
    )
    await user.click(dialog().getByRole("button", { name: "Cancel" }))
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )

    await user.click(
      within(await emptyState()).getByRole("button", {
        name: "Other policy types",
      }),
    )
    expect(await screen.findByRole("dialog")).toHaveAccessibleName("New policy")
  })

  it("returns focus to the page's action when the empty state's dialog closes", async () => {
    // Creating the first policy unmounts the empty state, so focus lands on
    // the heading's action rather than on the body.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(
      within(await emptyState()).getByRole("button", {
        name: "Create fallback",
      }),
    )
    await fillFallback(user, "cheap")
    await submitDialog(user)

    const trigger = await createTrigger()
    await waitFor(() => expect(trigger).toHaveFocus())
  })

  it("opens a fallback on a deep link naming its main model", async () => {
    mockApi([], null, [], { context: organizationContext() })
    renderPage(<RoutingPage />, "/routing?target=openai:gpt-4o")

    expect(await screen.findByRole("dialog")).toHaveAccessibleName(
      "New fallback",
    )
  })
})

describe("Creating in a workspace", () => {
  function operatorInWorkspace(): OrganizationContext {
    return organizationContext({
      workspace_memberships: [
        { workspace_id: ADMIN_WORKSPACE, name: "Alpha one", role: "owner" },
      ],
    })
  }

  it("lands an operator's create in the workspace they are looking at", async () => {
    const { calls } = mockApi([], null, [], { context: operatorInWorkspace() })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await startCreate(user)
    await fillFallback(user, "scoped", "openai:gpt-5-mini")
    await submitDialog(user)

    expect(policyWrites(calls)[0]).toMatchObject({
      name: "scoped",
      workspace_id: ADMIN_WORKSPACE,
    })
  })

  it("writes an admin's policy through the tenant-scoped router", async () => {
    const { calls } = mockApi([], null, [], {
      context: adminContext(),
      memberPolicies: [],
    })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await startCreate(user)
    await fillFallback(user, "tenant-fast", "openai:gpt-5-mini")
    await submitDialog(user)

    const written = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/organizations/me/routing-policies`),
    )
    expect(written?.body).toMatchObject({
      name: "tenant-fast",
      workspace_id: ADMIN_WORKSPACE,
    })
    expect(policyWrites(calls)).toHaveLength(0)
  })

  it("offers an admin no user scope, which the tenant router refuses", async () => {
    mockApi([], null, [], { context: adminContext(), memberPolicies: [] })
    const user = userEvent.setup()
    renderInWorkspace(<RoutingPage />)

    await startCreate(user)
    expect(
      await screen.findByText(/applies to everyone in the selected workspace/i),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: /^Applies to/ }),
    ).not.toBeInTheDocument()
  })
})
