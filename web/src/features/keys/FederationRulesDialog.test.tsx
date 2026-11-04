import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { FederationRule } from "@/client"
import {
  FederationRulesDialog,
  workloadEnvironment,
} from "@/features/keys/FederationRulesDialog"
import { API_ROOT } from "@/shared/api/client"
import { apiKey, organizationContext } from "@/tests/fixtures"

const RULES_URL = `${API_ROOT}/organizations/me/keys/key-1/federation-rules`

function rule(overrides: Partial<FederationRule> = {}): FederationRule {
  return {
    id: "rule-1",
    api_key_id: "key-1",
    issuer: "https://accounts.google.com",
    audience: "https://otari.example.com",
    subject: null,
    email: "hackbot@my-project.iam.gserviceaccount.com",
    member_managed: true,
    created_at: "2026-10-22T00:00:00+00:00",
    ...overrides,
  }
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

// A member's view: the scope is the member surface, and the organization is
// what the setup names.
function mockApi(initial: FederationRule[]) {
  let rules = [...initial]
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(
          organizationContext({
            deployment_operator: false,
            organization: {
              ...organizationContext().organization,
              id: "org-1",
            },
          }),
        )
      }
      if (url.endsWith(RULES_URL) && method === "POST") {
        const created = rule({
          id: "rule-new",
          ...JSON.parse(String(init?.body)),
        })
        rules = [...rules, created]
        return jsonResponse(created, 201)
      }
      if (url.includes(`${RULES_URL}/`) && method === "DELETE") {
        const id = decodeURIComponent(url.split("/").pop() ?? "")
        rules = rules.filter((row) => row.id !== id)
        return new Response(null, { status: 204 })
      }
      if (url.endsWith(RULES_URL)) return jsonResponse(rules)
      return jsonResponse([])
    })
}

function renderDialog() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <FederationRulesDialog
        apiKey={apiKey({ id: "key-1", key_name: "hackbot" })}
        onClose={() => {}}
      />
    </QueryClientProvider>,
  )
}

describe("FederationRulesDialog", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists the key's rules and sets up the newest", async () => {
    mockApi([
      rule({ id: "rule-old", email: "old@example.com", member_managed: false }),
      rule({ id: "rule-1" }),
    ])
    renderDialog()

    const rules = await screen.findByRole("list", { name: "Rules" })
    expect(within(rules).getAllByRole("listitem")).toHaveLength(2)
    expect(within(rules).getByText("Set by the operator")).toBeInTheDocument()
    const setup = screen.getByRole("region", { name: "Environment code" })
    await waitFor(() =>
      expect(setup).toHaveTextContent('ANTHROPIC_ORGANIZATION_ID="org-1"'),
    )
    expect(setup).toHaveTextContent('ANTHROPIC_FEDERATION_RULE_ID="rule-1"')
  })

  it("shows the setup of the rule asked for", async () => {
    mockApi([rule({ id: "rule-old", email: "old@example.com" }), rule()])
    const user = userEvent.setup()
    renderDialog()

    await user.click(
      await screen.findByRole("button", {
        name: "Show setup for old@example.com",
      }),
    )

    expect(
      screen.getByRole("region", { name: "Environment code" }),
    ).toHaveTextContent('ANTHROPIC_FEDERATION_RULE_ID="rule-old"')
  })

  it("adds a rule once it names an identity, and sets it up", async () => {
    const fetchMock = mockApi([])
    const user = userEvent.setup()
    renderDialog()

    await screen.findByText("No rules yet.")
    const add = screen.getByRole("button", { name: "Add rule" })
    expect(add).toBeDisabled()
    await user.type(
      screen.getByRole("textbox", { name: "Service account email" }),
      "bot@example.com",
    )
    await user.click(add)

    const setup = await screen.findByRole("region", {
      name: "Environment code",
    })
    expect(setup).toHaveTextContent('ANTHROPIC_FEDERATION_RULE_ID="rule-new"')
    const post = fetchMock.mock.calls.find(
      ([url, init]) =>
        String(url).endsWith(RULES_URL) && init?.method === "POST",
    )
    expect(JSON.parse(String(post?.[1]?.body))).toEqual({
      issuer: "https://accounts.google.com",
      audience: window.location.origin,
      email: "bot@example.com",
      subject: null,
    })
  })

  it("deletes a rule after a confirm", async () => {
    const fetchMock = mockApi([rule()])
    const user = userEvent.setup()
    renderDialog()

    await user.click(
      await screen.findByRole("button", {
        name: "Delete rule for hackbot@my-project.iam.gserviceaccount.com",
      }),
    )
    const confirm = await screen.findByRole("alertdialog")
    await user.click(
      within(confirm).getByRole("button", { name: "Delete rule" }),
    )

    expect(await screen.findByText("No rules yet.")).toBeInTheDocument()
    expect(
      fetchMock.mock.calls.some(
        ([url, init]) =>
          String(url).endsWith(`${RULES_URL}/rule-1`) &&
          init?.method === "DELETE",
      ),
    ).toBe(true)
  })
})

describe("workloadEnvironment", () => {
  it("points the SDKs at the gateway's /api and leaves out an unknown organization", () => {
    const environment = workloadEnvironment({
      baseUrl: "https://otari.example.com",
      ruleId: "rule-1",
      organizationId: "",
    })

    expect(environment).toContain(
      'export ANTHROPIC_BASE_URL="https://otari.example.com/api"',
    )
    expect(environment).not.toContain("ANTHROPIC_ORGANIZATION_ID")
  })
})
