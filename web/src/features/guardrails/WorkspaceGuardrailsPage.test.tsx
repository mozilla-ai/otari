import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  OrganizationGuardrail,
  OrganizationGuardrailDefinition,
} from "@/client"
import { WorkspaceGuardrailsPage } from "@/features/guardrails/WorkspaceGuardrailsPage"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import {
  organizationContext,
  organizationGuardrail,
  organizationGuardrailDefinition,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const ALPHA = "11111111-1111-1111-1111-111111111111"
const BETA = "22222222-2222-2222-2222-222222222222"

function mockApi({
  definitions = [] as OrganizationGuardrailDefinition[],
  mandates = [] as OrganizationGuardrail[],
  role = "owner",
} = {}) {
  const calls: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url.includes("/organizations/me/guardrail-definitions")) {
      calls.push(url)
      return Response.json({ data: definitions, count: definitions.length })
    }
    if (url.includes("/organizations/me/guardrails")) {
      calls.push(url)
      return Response.json({ data: mandates, count: mandates.length })
    }
    if (url.includes(`${API_ROOT}/tool-settings/guardrails/catalog`)) {
      calls.push(url)
      return Response.json({ guardrails: [] })
    }
    return Response.json(
      organizationContext({
        role,
        workspace_memberships: [
          { workspace_id: ALPHA, name: "Alpha", role: "admin" },
        ],
      }),
    )
  })
  return calls
}

async function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  await renderWithRouter(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <WorkspaceGuardrailsPage />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
    { url: "/tools/guardrails" },
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("WorkspaceGuardrailsPage", () => {
  it("lists only the guardrails running here, in one table with no heading", async () => {
    mockApi({
      definitions: [organizationGuardrailDefinition({ name: "prod-lakera" })],
      mandates: [
        organizationGuardrail({
          id: "a",
          profile: "here",
          workspace_ids: [ALPHA],
        }),
        organizationGuardrail({
          id: "b",
          profile: "everywhere",
          applies_to_all_workspaces: true,
        }),
        organizationGuardrail({
          id: "c",
          profile: "elsewhere",
          workspace_ids: [BETA],
        }),
        // Mandated here but paused, so not running.
        organizationGuardrail({
          id: "d",
          profile: "paused",
          workspace_ids: [ALPHA],
          enabled: false,
        }),
      ],
    })
    await renderPage()

    const running = within(
      await screen.findByRole("grid", {
        name: "Guardrails running in this workspace",
      }),
    )
    expect(await running.findByText("here")).toBeInTheDocument()
    expect(running.getByText("everywhere")).toBeInTheDocument()
    expect(running.queryByText("elsewhere")).toBeNull()
    expect(running.queryByText("paused")).toBeNull()
    // No table of what is configured, and no section headings.
    expect(screen.getAllByRole("grid")).toHaveLength(1)
    expect(screen.queryByText("prod-lakera")).toBeNull()
    expect(screen.queryByRole("heading", { level: 2 })).toBeNull()
  })

  it("offers nothing to change here, and points at where it is changed", async () => {
    mockApi({
      definitions: [organizationGuardrailDefinition()],
      mandates: [organizationGuardrail({ workspace_ids: [ALPHA] })],
    })
    await renderPage()

    await screen.findByRole("grid", {
      name: "Guardrails running in this workspace",
    })
    expect(screen.queryByRole("button")).toBeNull()
    expect(screen.getByRole("link", { name: "Guardrails" })).toHaveAttribute(
      "href",
      "/organization/guardrails",
    )
    // And none of the deployment's guardrails settings this page used to hold.
    expect(screen.queryByText(/Backend URL/)).toBeNull()
  })

  it("withholds the table and its reads from a member", async () => {
    const calls = mockApi({ role: "member" })
    await renderPage()

    expect(
      await screen.findByText(/set by an owner or admin of the organization/),
    ).toBeInTheDocument()
    expect(screen.queryByRole("grid")).toBeNull()
    expect(calls).toEqual([])
  })
})
