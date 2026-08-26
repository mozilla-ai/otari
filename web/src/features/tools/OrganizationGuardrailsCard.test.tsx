import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { OrganizationGuardrail } from "@/client"
import { OrganizationGuardrailsCard } from "@/features/tools/OrganizationGuardrailsCard"
import { organizationContext, organizationGuardrail } from "@/tests/fixtures"
import { selectTrigger } from "@/tests/select"

const ALPHA = "11111111-1111-1111-1111-111111111111"
const BETA = "22222222-2222-2222-2222-222222222222"

function mockApi({
  guardrails = [] as OrganizationGuardrail[],
  role = "owner",
}: {
  guardrails?: OrganizationGuardrail[]
  role?: string
} = {}) {
  const calls: { url: string; method: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = init?.method ?? "GET"
    if (url.includes("/organizations/me/guardrails")) {
      calls.push({
        url,
        method,
        body:
          typeof init?.body === "string" ? JSON.parse(init.body) : init?.body,
      })
      if (method === "GET") {
        return Response.json({ data: guardrails, count: guardrails.length })
      }
      return Response.json(guardrails[0] ?? organizationGuardrail())
    }
    if (url.includes("/v1/workspaces")) {
      return Response.json({
        data: [
          { id: ALPHA, name: "Alpha" },
          { id: BETA, name: "Beta" },
        ],
        count: 2,
      })
    }
    return Response.json(organizationContext({ role }))
  })
  return calls
}

function renderCard() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <OrganizationGuardrailsCard onSaved={() => {}} />
    </QueryClientProvider>,
  )
}

describe("OrganizationGuardrailsCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("says nothing is mandated when the organization has no entries", async () => {
    mockApi()
    renderCard()

    expect(
      await screen.findByText(/No organization guardrails/),
    ).toBeInTheDocument()
  })

  it("hides the whole surface from a member who cannot manage the organization", async () => {
    const calls = mockApi({ role: "member" })
    renderCard()

    expect(
      await screen.findByText(/set by an owner or admin of the organization/),
    ).toBeInTheDocument()
    // The read is gated too, so nothing is asked for: the entries name the
    // endpoints this gateway connects to.
    expect(calls).toEqual([])
  })

  it("names the workspaces an entry runs in, and says when it runs everywhere", async () => {
    mockApi({
      guardrails: [
        organizationGuardrail({ profile: "pii", workspace_ids: [ALPHA] }),
        organizationGuardrail({
          id: "66666666-6666-6666-6666-666666666666",
          profile: "prompt-injection",
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    expect(await screen.findByText("Alpha")).toBeInTheDocument()
    expect(
      await screen.findByText("Every workspace, including new ones"),
    ).toBeInTheDocument()
  })

  it("marks a paused entry and one that carries its own endpoint and credential", async () => {
    mockApi({
      guardrails: [
        organizationGuardrail({
          enabled: false,
          url: "https://guardrails.example",
          has_credential: true,
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    expect(await screen.findByText("own endpoint")).toBeInTheDocument()
    expect(screen.getByText("credential set")).toBeInTheDocument()
    // "Paused" is also an option in the status picker, so the badge is asserted
    // through the picker's value rather than by matching the word twice.
    expect(selectTrigger("Status")).toHaveTextContent("Paused")
  })

  it("never renders a stored credential back, only offers to replace it", async () => {
    mockApi({
      guardrails: [
        organizationGuardrail({
          has_credential: true,
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    const field = await screen.findByLabelText(
      "New credential for prompt-injection",
    )
    expect(field).toHaveValue("")
    expect(field).toHaveAttribute("placeholder", "replace credential")
  })

  it("omits the credential from a save that did not touch it", async () => {
    const calls = mockApi({
      guardrails: [
        organizationGuardrail({
          has_credential: true,
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    await userEvent.click(
      await screen.findByRole("button", { name: "Save prompt-injection" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH")).toBe(true),
    )
    const patch = calls.find((call) => call.method === "PATCH")
    expect(patch?.body).not.toHaveProperty("credential")
    // And no workspace list either, since the entry applies to all of them and
    // the server refuses the pair.
    expect(patch?.body).not.toHaveProperty("workspace_ids")
  })

  it("sends the chosen workspaces when the entry does not apply to all of them", async () => {
    const calls = mockApi({
      guardrails: [organizationGuardrail({ workspace_ids: [ALPHA] })],
    })
    renderCard()

    await userEvent.click(
      await screen.findByLabelText("prompt-injection: Beta"),
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Save prompt-injection" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH")).toBe(true),
    )
    expect(calls.find((call) => call.method === "PATCH")?.body).toMatchObject({
      applies_to_all_workspaces: false,
      workspace_ids: [ALPHA, BETA],
    })
  })

  it("rewrites the endpoint in place, so a typo is not a delete and recreate", async () => {
    const calls = mockApi({
      guardrails: [
        organizationGuardrail({
          url: "https://wrong.example",
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    const endpoint = await screen.findByLabelText(
      "Endpoint for prompt-injection",
    )
    await userEvent.clear(endpoint)
    await userEvent.type(endpoint, "https://right.example")
    await userEvent.click(
      screen.getByRole("button", { name: "Save prompt-injection" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH")).toBe(true),
    )
    expect(calls.find((call) => call.method === "PATCH")?.body).toMatchObject({
      url: "https://right.example",
    })
  })

  it("leaves a stored endpoint alone on a save that did not touch it", async () => {
    const calls = mockApi({
      guardrails: [
        organizationGuardrail({
          url: "https://guardrails.example",
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    await userEvent.click(
      await screen.findByRole("button", { name: "Save prompt-injection" }),
    )

    await waitFor(() =>
      expect(calls.some((call) => call.method === "PATCH")).toBe(true),
    )
    expect(
      calls.find((call) => call.method === "PATCH")?.body,
    ).not.toHaveProperty("url")
  })

  it("keeps one row's unsaved edits when another row is saved", async () => {
    // Passes with either dependency list: TanStack Query's structural sharing
    // hands the untouched row back its previous object, so the refetch a save
    // triggers does not re-run its effect. Kept as the property worth holding
    // rather than as a regression test for the dependency array.
    mockApi({
      guardrails: [
        organizationGuardrail({
          profile: "pii",
          applies_to_all_workspaces: true,
        }),
        organizationGuardrail({
          id: "66666666-6666-6666-6666-666666666666",
          profile: "prompt-injection",
          applies_to_all_workspaces: true,
        }),
      ],
    })
    renderCard()

    const edited = await screen.findByLabelText("Endpoint for prompt-injection")
    await userEvent.type(edited, "https://half-typed.example")
    await userEvent.click(screen.getByRole("button", { name: "Save pii" }))

    await waitFor(() =>
      expect(
        screen.getByLabelText("Endpoint for prompt-injection"),
      ).toHaveValue("https://half-typed.example"),
    )
  })

  it("mandates a new guardrail from the add form", async () => {
    const calls = mockApi()
    renderCard()

    await userEvent.type(
      await screen.findByLabelText("Guardrail profile"),
      "pii",
    )
    await userEvent.click(screen.getByRole("button", { name: "Add" }))

    await waitFor(() =>
      expect(calls.some((call) => call.method === "POST")).toBe(true),
    )
    expect(calls.find((call) => call.method === "POST")?.body).toMatchObject({
      profile: "pii",
      mode: "monitor",
      url: null,
      credential: null,
      applies_to_all_workspaces: false,
      workspace_ids: [],
    })
  })
})
