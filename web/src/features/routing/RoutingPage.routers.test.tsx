import { screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { PolicySpec } from "@/client"
import { RoutingPage } from "@/features/routing/RoutingPage"
import { API_ROOT } from "@/shared/api/client"
import {
  createTrigger,
  LEARNED,
  mockApi,
  policy,
  renderPage,
  WEIGHTED,
} from "@/tests/routing"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RoutingPage learned and weighted routers", () => {
  it("summarises a learned policy by its pool rather than as an opaque dynamic row", async () => {
    // "Chosen per request" is true of a tier-down too. What an operator needs to
    // see here is that a router picks between named models, and which one serves
    // when it declines.
    mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    expect(
      within(row).getByText(/Learned . 2 candidates, openai:gpt-5 by default/),
    ).toBeInTheDocument()
  })

  it("puts the fallback in the pool rather than asking for it twice", async () => {
    // The fallback is always one of the models the router may choose, so the form
    // shows one list with the safe one marked. A stored spec that omitted its default
    // target from `candidates` still shows it, because the gateway appends it.
    mockApi([
      policy("smart", {
        select: [
          { router: "knn", candidates: ["openai:gpt-5-nano"] },
          { default: "openai:gpt-5" },
        ],
      }),
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    // Both models in one list, and the default target marked.
    expect(screen.getByRole("combobox", { name: /model 1/i })).toHaveValue(
      "openai:gpt-5-nano",
    )
    expect(screen.getByRole("combobox", { name: /model 2/i })).toHaveValue(
      "openai:gpt-5",
    )
    const marks = screen.getAllByRole("radio", { name: /serves when unsure/i })
    expect(marks[1]).toBeChecked()
    // ...and no second field asking for the same model again.
    expect(
      screen.queryByRole("combobox", { name: /^serves$/i }),
    ).not.toBeInTheDocument()
  })

  it("marking a different model as the fallback changes the saved default", async () => {
    const { calls } = mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await user.click(
      screen.getAllByRole("radio", { name: /serves when unsure/i })[0],
    )
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[1]).toEqual({ default: "openai:gpt-5-nano" })
    // The pool is unchanged: marking a fallback is not reordering.
    expect(spec.select[0]).toEqual({
      router: "knn",
      candidates: ["openai:gpt-5-nano", "openai:gpt-5"],
    })
  })

  it("edits a learned policy without losing its candidate pool", async () => {
    // A router entry the form can represent must be editable: showing it read-only
    // would mean the only way to change a candidate is the API.
    const { calls } = mockApi([policy("smart", LEARNED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("smart")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    expect(screen.getByRole("combobox", { name: /model 1/i })).toHaveValue(
      "openai:gpt-5-nano",
    )
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[0]).toEqual({
      router: "knn",
      candidates: ["openai:gpt-5-nano", "openai:gpt-5"],
    })
    expect(spec.select[1]).toEqual({ default: "openai:gpt-5" })
  })

  it("will not save a pool of one, which is not a routing decision", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "smart",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5",
    )
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: /let a router pick/i }))
    // The pool is seeded with the policy's target plus one empty row, so removing
    // the empty row leaves a single candidate.
    await user.click(screen.getAllByRole("button", { name: "Remove" })[1])

    expect(screen.getByText(/at least two models/i)).toBeInTheDocument()
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    )
    expect(calls.some((call) => call.method === "POST")).toBe(false)
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

  it("creates a weighted policy from the split control", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "balanced",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5",
    )
    await user.keyboard("{Escape}")
    await user.click(
      screen.getByRole("button", { name: /split traffic across providers/i }),
    )

    // Seeded as an even split of the policy's own target plus one empty row, so the
    // operator names the second provider and skews the shares.
    await user.type(
      screen.getByRole("combobox", { name: /model 2/i }),
      "anthropic:claude-sonnet-4-5",
    )
    await user.keyboard("{Escape}")
    const shares = screen.getAllByRole("textbox", { name: /share/i })
    await user.clear(shares[0])
    await user.type(shares[0], "70")
    await user.clear(shares[1])
    await user.type(shares[1], "30")
    // Relative weights are hard to read, so the form says what they come to.
    expect(screen.getByText("70% of requests")).toBeInTheDocument()
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[0]).toEqual({
      router: "weighted",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
      weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30 },
    })
    expect(spec.select[1]).toEqual({ default: "openai:gpt-5" })
  })

  it("edits a weighted policy without losing its split", async () => {
    const { calls } = mockApi([
      policy("balanced", WEIGHTED, { is_dynamic: true }),
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const shares = screen.getAllByRole("textbox", { name: /share/i })
    expect(shares[0]).toHaveValue("70")
    expect(shares[1]).toHaveValue("30")
    // Drain the second provider without deleting it, which is what a zero share is
    // for. The form has to say the model is still there, or a zero reads as removal.
    await user.clear(shares[1])
    await user.type(shares[1], "0")
    expect(
      screen.getByText(/No weighted traffic; still tried if another fails/),
    ).toBeInTheDocument()
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[0]).toEqual({
      router: "weighted",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
      weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 0 },
    })
  })

  it("keeps a fractional share typeable and refuses a non-numeric one", async () => {
    // The field holds what was typed, so a decimal point survives the keystroke that
    // follows it. "Infinity" and a negative parse but are refused, matching the API's
    // finite, non-negative rule rather than being coerced to something else on save.
    const { calls } = mockApi([
      policy("balanced", WEIGHTED, { is_dynamic: true }),
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    const shares = screen.getAllByRole("textbox", { name: /share/i })
    await user.clear(shares[0])
    await user.type(shares[0], "Infinity")
    expect(
      screen.getByText(/Every share is a number of zero or more/),
    ).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()

    await user.clear(shares[0])
    await user.type(shares[0], "-5")
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()

    // The decimal has to survive the round trip, not only the keystroke: a field
    // that renders "7.5" but posts 7 would be the same bug one layer down.
    await user.clear(shares[0])
    await user.type(shares[0], "7.5")
    expect(shares[0]).toHaveValue("7.5")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[0].weights).toEqual({
      "openai:gpt-5": 7.5,
      "anthropic:claude-sonnet-4-5": 30,
    })
  })

  it("edits a weighted policy whose backend name is spelled loosely", async () => {
    // The gateway resolves a backend on `name.strip().lower()`, so " Weighted " is a
    // working policy. Reading it as an unknown backend would show it read-only and
    // label it wrong on a page that otherwise offers to edit it.
    const loose: PolicySpec = {
      select: [
        {
          router: " Weighted ",
          candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
          weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30 },
        },
        { default: "openai:gpt-5" },
      ],
    }
    const { calls } = mockApi([policy("balanced", loose, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    expect(within(row).getByText("WEIGHTED")).toBeInTheDocument()
    await user.click(within(row).getByRole("button", { name: "Edit" }))

    // Loading it as weighted is half the claim; saving it back unchanged is the
    // other half. The spelling is normalized on the way out, which is what the
    // gateway would have resolved it to anyway.
    const shares = screen.getAllByRole("textbox", { name: /share/i })
    expect(shares[0]).toHaveValue("70")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    )

    const post = calls.find(
      (call) =>
        call.method === "POST" &&
        call.url.includes(`${API_ROOT}/routing/policies`),
    )
    const spec = (post!.body as { spec: PolicySpec }).spec
    expect(spec.select[0]).toEqual({
      router: "weighted",
      candidates: ["openai:gpt-5", "anthropic:claude-sonnet-4-5"],
      weights: { "openai:gpt-5": 70, "anthropic:claude-sonnet-4-5": 30 },
    })
  })

  it("will not save a split where every share is zero", async () => {
    // It would select nothing and the policy would always serve its marked model,
    // which is a load balancer that balances nothing. The API refuses it too.
    const { calls } = mockApi([
      policy("balanced", WEIGHTED, { is_dynamic: true }),
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    for (const share of screen.getAllByRole("textbox", { name: /share/i })) {
      await user.clear(share)
      await user.type(share, "0")
    }

    expect(
      screen.getByText(/at least one model a share above zero/i),
    ).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()
    expect(calls.some((call) => call.method === "POST")).toBe(false)
  })

  it("adds a model to a split with no traffic until a share is set", async () => {
    // Adding a provider must not silently move traffic onto it.
    mockApi([policy("balanced", WEIGHTED, { is_dynamic: true })])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    await user.click(screen.getByRole("button", { name: "+ Another model" }))

    const shares = screen.getAllByRole("textbox", { name: /share/i })
    expect(shares[2]).toHaveValue("0")
  })

  it("will not save a split that names the same model twice", async () => {
    // Two rows collapse to one key in the weight map, so the split saved would not be
    // the split shown (and the API refuses a repeated candidate regardless).
    const { calls } = mockApi([
      policy("balanced", WEIGHTED, { is_dynamic: true }),
    ])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const row = (await screen.findByText("balanced")).closest("tr")!
    await user.click(within(row).getByRole("button", { name: "Edit" }))
    const second = screen.getByRole("combobox", { name: /model 2/i })
    await user.clear(second)
    await user.type(second, "openai:gpt-5")
    await user.keyboard("{Escape}")

    expect(screen.getByText(/name each model once/i)).toBeInTheDocument()
    expect(
      within(screen.getByRole("dialog")).getByRole("button", { name: "Save" }),
    ).toBeDisabled()
    expect(calls.some((call) => call.method === "POST")).toBe(false)
  })

  it("will not author a policy that dispatches more models than the server allows", async () => {
    // The cap counts the routed pool plus the fallback chain. Authoring past it and
    // finding out via a 400 on Save is the form lying about its own rules.
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "wide",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5",
    )
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: /let a router pick/i }))
    // Seeded with 2 candidates; add three more to reach the cap of 5.
    for (let i = 0; i < 3; i += 1) {
      await user.click(screen.getByRole("button", { name: "+ Another model" }))
    }

    expect(
      screen.getByRole("button", { name: "+ Another model" }),
    ).toBeDisabled()
    expect(screen.getByText(/dispatches at most 5 models/i)).toBeInTheDocument()
    expect(calls.some((call) => call.method === "POST")).toBe(false)
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
    expect(within(row).getByText(/cannot show yet/)).toBeInTheDocument()
    // Reading its readiness is still fine.
    expect(
      within(row).getByRole("button", { name: "Examples" }),
    ).toBeInTheDocument()
  })
})
