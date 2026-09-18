import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { PolicySpec } from "@/client"
import { RoutingPage } from "@/features/routing/RoutingPage"
import { API_ROOT } from "@/shared/api/client"
import {
  CHAIN,
  createTrigger,
  mockApi,
  policy,
  renderPage,
} from "@/tests/routing"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RoutingPage policy form", () => {
  it("keeps the failure chain and guardrails out of the way until asked for", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    // Naming one model must stay a short task, so neither section is present yet.
    expect(screen.queryByText("If that fails, try")).not.toBeInTheDocument()
    expect(screen.queryByText("Always check")).not.toBeInTheDocument()

    await user.click(
      screen.getByRole("button", { name: /Add a fallback chain/ }),
    )
    expect(screen.getByText("If that fails, try")).toBeInTheDocument()
    // Adding another one belongs inside the section it extends, not in the row of
    // links that start a section. Walked up to the bordered section rather than
    // a fixed number of parents: the heading now sits in a row of its own with
    // the section's Remove, so counting levels would pin the markup instead of
    // the rule.
    const section = screen
      .getByText("If that fails, try")
      .closest<HTMLElement>("div.border")!
    expect(
      within(section).getByRole("button", { name: /Another fallback/ }),
    ).toBeInTheDocument()
  })

  it("takes a whole section away in one press, whatever it holds", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.click(
      screen.getByRole("button", { name: /Let a router pick the cheapest/ }),
    )
    // Two rows from one press, so the row control could never be a one-press
    // way out of this section: that is the asymmetry the section control fixes.
    expect(screen.getAllByRole("button", { name: "Remove" })).toHaveLength(2)

    await user.click(
      screen.getByRole("button", { name: "Remove the routed pool" }),
    )

    expect(
      screen.queryByRole("button", { name: "Remove" }),
    ).not.toBeInTheDocument()
    // The affordance that summons it is back, which is what says the section is
    // gone rather than merely emptied.
    expect(
      screen.getByRole("button", { name: /Let a router pick the cheapest/ }),
    ).toBeInTheDocument()
  })

  it("offers a section control beside every section that can be summoned", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    for (const [summon, remove] of [
      [/Tier down when the budget fills up/, "Remove the budget tier-down"],
      [/Add a fallback chain/, "Remove the fallback chain"],
      [/Add guardrails/, "Remove the guardrails"],
    ] as const) {
      await user.click(screen.getByRole("button", { name: summon }))
      expect(screen.getByRole("button", { name: remove })).toBeInTheDocument()
    }
  })

  it("disables the guardrails affordance when no guardrails service is configured", async () => {
    mockApi([], null)
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    const add = await screen.findByRole("button", { name: /Add guardrails/ })

    // Disabled, and never silently: the reason and the route to fixing it sit next
    // to the control as text, so it works on touch and for a screen reader.
    expect(add).toBeDisabled()
    expect(
      screen.getByText(/No guardrails service is configured/),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: /Tools & Guardrails/ }),
    ).toHaveAttribute("href", "/tools")
  })

  it("refuses a policy name that would shadow a real model selector", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "openai:gpt-4o",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-nano",
    )
    await user.keyboard("{Escape}")

    expect(screen.getByText(/cannot contain/)).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    ).toBeDisabled()
  })

  it("warns when a guardrail makes the guardrails service a hard dependency", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.click(screen.getByRole("button", { name: /Add guardrails/ }))

    // block + block is the honest default, and its cost has to be visible where
    // the choice is made: an outage then refuses every request through the policy.
    expect(
      screen.getByText(/rejects every request through this policy/),
    ).toBeInTheDocument()
  })

  it("refuses a tier-down threshold that could never fire", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "thrifty",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-mini",
    )
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: /Tier down/ }))

    const threshold = screen.getByRole("textbox", {
      name: /budget used at least/i,
    })
    await user.clear(threshold)
    await user.type(threshold, "100")

    // The budget gate refuses the request before selection at 100%, so such a rule
    // is dead config. Saying so here beats a 400 from the server.
    expect(screen.getByText("Must be under 100.")).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    ).toBeDisabled()
  })

  it("renames a policy through the name field, sending rename_from", async () => {
    // The name is the key, so an edit that changes it has to say which row it moves.
    // Posting the new name alone would create a second policy and leave the old one
    // serving callers.
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const nameField = screen.getByRole("textbox", { name: /policy name/i })
    await user.clear(nameField)
    await user.type(nameField, "speedy")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const body = post!.body as {
      name: string
      rename_from?: string
      spec: PolicySpec
    }
    expect(body.name).toBe("speedy")
    expect(body.rename_from).toBe("fast")
    // The rest of the policy rides along on the same write, so a rename cannot land
    // half-applied.
    expect(body.spec.on_failure).toEqual(["anthropic:claude-haiku-4-5"])
    expect(await screen.findByText("speedy")).toBeInTheDocument()
    expect(screen.queryByText("fast")).not.toBeInTheDocument()
  })

  it("omits rename_from when an edit leaves the name alone", async () => {
    // Sending it unchanged would be harmless server-side, but a plain spec edit
    // reading as a rename in the audit log is not.
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    expect((post!.body as { rename_from?: string }).rename_from).toBeUndefined()
  })

  it("says what a pending rename will do before it is saved", async () => {
    mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const nameField = screen.getByRole("textbox", { name: /policy name/i })
    await user.clear(nameField)
    await user.type(nameField, "speedy")

    // Renaming changes what callers must send and splits historical usage, so the
    // consequence belongs next to the field rather than in a release note.
    expect(
      screen.getByText(/Callers have to send the new name/),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/usage already recorded keeps the old one/),
    ).toBeInTheDocument()
  })

  it("refuses a renamed policy whose new name carries a delimiter", async () => {
    // Same rule as a create: ":" or "/" would shadow a real model selector.
    const { calls } = mockApi([policy("fast", CHAIN)])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const nameField = screen.getByRole("textbox", { name: /policy name/i })
    await user.clear(nameField)
    await user.type(nameField, "openai:gpt-5")

    expect(screen.getByText(/cannot contain/)).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()
    expect(calls.some((call) => call.method === "POST")).toBe(false)
  })

  it("keeps an alias name fixed, since the alias API cannot rename", async () => {
    mockApi([], "http://guardrails:8000", [
      {
        name: "gpt",
        target: "openai:gpt-5-mini",
        source: "stored",
        user_id: null,
      },
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("gpt")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    expect(
      screen.queryByRole("textbox", { name: /policy name/i }),
    ).not.toBeInTheDocument()
    expect(screen.getByText(/An alias name is its key/)).toBeInTheDocument()
  })
})
