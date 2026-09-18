import { screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { PolicySpec } from "@/client"
import { RoutingPage } from "@/features/routing/RoutingPage"
import { API_ROOT } from "@/shared/api/client"
import {
  createTrigger,
  mockApi,
  policy,
  renderPage,
  WEIGHTED,
} from "@/tests/routing"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RoutingPage", () => {
  it("lists policies with what they serve and where they come from", async () => {
    mockApi()
    renderPage(<RoutingPage />)

    const fastRow = (await screen.findByText("fast")).closest("tr")!
    // The chain is summarised rather than hidden: an operator scanning the table
    // needs to see that a fallback exists without opening the policy.
    expect(within(fastRow).getByText(/openai:gpt-5-mini/)).toBeInTheDocument()
    expect(within(fastRow).getByText(/\+1 on failure/)).toBeInTheDocument()
    expect(within(fastRow).getByText("STORED")).toBeInTheDocument()
  })

  it("marks a policy that decides per request, since it has no single target", async () => {
    mockApi()
    renderPage(<RoutingPage />)

    const autoRow = (await screen.findByText("auto")).closest("tr")!
    expect(within(autoRow).getByText("DYNAMIC")).toBeInTheDocument()
    expect(within(autoRow).getByText(/Chosen per request/)).toBeInTheDocument()
  })

  it("does not offer to edit or delete a policy that lives in config.yml", async () => {
    mockApi()
    renderPage(<RoutingPage />)

    const autoRow = (await screen.findByText("auto")).closest("tr")!
    expect(within(autoRow).getByText("set in config.yml")).toBeInTheDocument()
    expect(
      within(autoRow).queryByRole("button", { name: "Delete" }),
    ).not.toBeInTheDocument()
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

  it("offers the same dialog from the empty state", async () => {
    // The empty state's explanation is the page's onboarding and stays; what it
    // gained is the action, so a first policy does not have to be started from
    // the heading a reader has already scrolled past.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    // Two of them on screen deliberately, the heading's and this one, so the
    // press is scoped to the empty state rather than picked by position, which
    // it no longer strictly needs now that its label is its own, and which is
    // kept because scoping is the right query either way.
    const empty = (
      await screen.findByRole("heading", {
        name: "No routing policies yet",
      })
    ).closest("div")!.parentElement!
    await user.click(
      within(empty).getByRole("button", { name: "Create your first policy" }),
    )
    const dialog = await screen.findByRole("dialog")
    expect(dialog).toHaveAccessibleName("New policy")
  })

  it("names the object in the title and the policy in the description when editing", async () => {
    // `title` is a string, so the old heading's `<code>` name moved into the
    // description rather than being dropped: it is the policy's identity and it
    // is what tells an operator which row they pressed Edit on.
    mockApi()
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    const fastRow = (await screen.findByText("fast")).closest("tr")!
    await user.click(within(fastRow).getByRole("button", { name: "Edit" }))
    const dialog = await screen.findByRole("dialog")
    expect(dialog).toHaveAccessibleName("Edit policy")
    expect(dialog).toHaveTextContent("fast")
    expect(
      within(dialog).getByRole("button", { name: "Save" }),
    ).toBeInTheDocument()
  })

  it("guards a half-built policy against a stray Escape", async () => {
    // This form grows a fallback chain, a condition tier and a guardrail list as
    // they are asked for, so it is exactly the one where ten minutes of work sits
    // behind one keystroke.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "cheap",
    )
    await user.keyboard("{Escape}")

    const dialog = screen.getByRole("dialog")
    expect(dialog).toHaveTextContent("Unsaved changes")
    expect(
      within(dialog).getByRole("button", { name: "Keep editing" }),
    ).toBeInTheDocument()
    await user.click(
      within(dialog).getByRole("button", { name: "Keep editing" }),
    )
    expect(
      within(dialog).getByRole("button", { name: "Create policy" }),
    ).toBeInTheDocument()
  })

  it("creates a one-target policy from three fields", async () => {
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "cheap",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-nano",
    )
    // Close the combobox popover, which otherwise aria-hides the submit button.
    await user.keyboard("{Escape}")
    await user.click(
      within(screen.getByRole("dialog")).getByRole("button", {
        name: "Create policy",
      }),
    )

    const post = calls.find((call) => call.method === "POST")
    expect(post).toBeDefined()
    const body = post!.body as { name: string; spec: PolicySpec }
    expect(body.name).toBe("cheap")
    // The fallthrough is explicit and last, which is what the schema requires.
    expect(body.spec.select).toEqual([{ default: "openai:gpt-5-nano" }])
    expect(body.spec.on_failure).toBeUndefined()
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
    expect(within(row).getByText(/cannot show yet/)).toBeInTheDocument()
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
    expect(within(row).getByText("openai:gpt-4o-mini")).toBeInTheDocument()
    expect(within(row).getByText("alias")).toBeInTheDocument()
    expect(
      within(row).getByRole("button", { name: "Delete" }),
    ).toBeInTheDocument()
  })

  it("does not fetch the form's own reads until the dialog opens", async () => {
    // The form stays mounted while closed so the frame can play its exit, which
    // puts its queries on the page unless they are gated: the roster and the
    // tool settings are the form's, not the table's.
    const { calls } = mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await screen.findByText("No routing policies yet")
    const formReads = () =>
      calls.filter(
        (call) =>
          call.url.includes(`${API_ROOT}/users`) ||
          call.url.includes(`${API_ROOT}/tool-settings`),
      )
    expect(formReads()).toHaveLength(0)

    await user.click(await createTrigger())
    await waitFor(() => expect(formReads().length).toBeGreaterThan(0))
  })

  it("offers a fresh draft on each open of the create dialog", async () => {
    // Reset on the way in, not on the way out: the frame keeps its content
    // while it animates out, so clearing on close blanks the body in front of
    // the operator. The page keys the form on an open counter instead.
    mockApi([])
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "half-typed",
    )

    // Out through the guard, which is the only way out of a dirty form.
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Discard" }))

    await user.click(await createTrigger())
    expect(screen.getByRole("textbox", { name: /policy name/i })).toHaveValue(
      "",
    )
  })

  it("reports a refused save inside the dialog, leaving the form filled", async () => {
    // The failure this guards against is the silent one: the mutation refuses,
    // the dialog stays, and nothing on screen says why. Its delete equivalent
    // is below; a page-level banner is no use here, because the operator is
    // looking at the modal and a message behind the backdrop is unread.
    mockApi([], "http://guardrails:8000", [], {
      saveBody: { status: 400, detail: "cheap already names an alias" },
    })
    const user = userEvent.setup()
    renderPage(<RoutingPage />)

    await user.click(await createTrigger())
    await user.type(
      screen.getByRole("textbox", { name: /policy name/i }),
      "cheap",
    )
    await user.type(
      screen.getByRole("combobox", { name: /^serves$/i }),
      "openai:gpt-5-nano",
    )
    await user.keyboard("{Escape}")
    const dialog = screen.getByRole("dialog")
    await user.click(
      within(dialog).getByRole("button", { name: "Create policy" }),
    )

    expect(
      await within(dialog).findByText(/already names an alias/),
    ).toBeVisible()
    // Still open with the work intact, so the operator can correct the name
    // rather than retyping the policy.
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: /policy name/i })).toHaveValue(
      "cheap",
    )
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
})
