import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type {
  BuiltInGuardrailCatalog,
  OrganizationGuardrail,
  OrganizationGuardrailDefinition,
} from "@/client"
import { OrganizationGuardrailsPage } from "@/features/guardrails/OrganizationGuardrailsPage"
import { API_ROOT } from "@/shared/api/client"
import {
  organizationContext,
  organizationGuardrail,
  organizationGuardrailDefinition,
} from "@/tests/fixtures"

const ALPHA = "11111111-1111-1111-1111-111111111111"

const CATALOG = {
  guardrails: [
    {
      guardrail_name: "lakera_guard",
      display_name: "Lakera Guard",
      vendor: "Lakera",
      categories: ["prompt_injection", "pii"],
      create_parameters: [],
    },
  ],
} as unknown as BuiltInGuardrailCatalog

interface Call {
  url: string
  method: string
  body: unknown
}

function mockApi({
  definitions = [] as OrganizationGuardrailDefinition[],
  mandates = [] as OrganizationGuardrail[],
  role = "owner",
} = {}) {
  const calls: Call[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = init?.method ?? "GET"
    const body =
      typeof init?.body === "string" ? JSON.parse(init.body) : undefined
    if (url.includes("/organizations/me/guardrail-definitions")) {
      calls.push({ url, method, body })
      if (method === "GET") {
        return Response.json({ data: definitions, count: definitions.length })
      }
      return Response.json(definitions[0] ?? organizationGuardrailDefinition())
    }
    if (url.includes("/organizations/me/guardrails")) {
      calls.push({ url, method, body })
      if (method === "GET") {
        return Response.json({ data: mandates, count: mandates.length })
      }
      return Response.json(mandates[0] ?? organizationGuardrail())
    }
    if (url.includes(`${API_ROOT}/tool-settings/guardrails/catalog`)) {
      calls.push({ url, method, body })
      return Response.json(CATALOG)
    }
    if (url.includes(`${API_ROOT}/tool-settings/guardrails/profiles`)) {
      calls.push({ url, method, body })
      return Response.json({ available: false, reason: "unset", profiles: [] })
    }
    if (url.includes(`${API_ROOT}/workspaces`)) {
      return Response.json({ data: [{ id: ALPHA, name: "Alpha" }], count: 1 })
    }
    return Response.json(organizationContext({ role }))
  })
  return calls
}

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <OrganizationGuardrailsPage />
    </QueryClientProvider>,
  )
}

async function row(tableName: string, text: string) {
  const cell = await within(
    await screen.findByRole("grid", { name: tableName }),
  ).findByText(text)
  const found = cell.closest('[role="row"]')
  if (!(found instanceof HTMLElement)) throw new Error(`no row for ${text}`)
  return within(found)
}

const lakera = organizationGuardrailDefinition({
  id: "d1",
  name: "prod-lakera",
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe("OrganizationGuardrailsPage", () => {
  it("withholds the tables, and every guardrail and catalog read, from a member", async () => {
    const calls = mockApi({ role: "member" })
    renderPage()

    expect(
      await screen.findByText(/set by an owner or admin of the organization/),
    ).toBeInTheDocument()
    expect(screen.queryByRole("grid")).toBeNull()
    expect(calls).toEqual([])
  })

  it("lists a definition by what it is, never by its secret", async () => {
    mockApi({ definitions: [lakera] })
    renderPage()

    const definition = await row(
      "Guardrails you have configured",
      "prod-lakera",
    )
    expect(definition.getByText("Lakera Guard · Lakera")).toBeInTheDocument()
    expect(
      definition.getByText(
        "Personally identifiable information, Prompt injection",
      ),
    ).toBeInTheDocument()
    expect(definition.getByText("Api key set")).toBeInTheDocument()
    expect(definition.getByText("Active")).toBeInTheDocument()
  })

  it("says who runs each mandate, and what it does when the check cannot run", async () => {
    mockApi({
      definitions: [lakera],
      mandates: [
        organizationGuardrail({
          id: "m1",
          profile: "on-definition",
          definition_id: "d1",
          mode: "block",
          on_unavailable: "block",
        }),
        organizationGuardrail({
          id: "m2",
          profile: "on-service",
          url: "https://guardrails.example/validate",
        }),
        organizationGuardrail({ id: "m3", profile: "on-deployment" }),
      ],
    })
    renderPage()

    const onDefinition = await row("Where they run", "on-definition")
    expect(onDefinition.getByText("prod-lakera")).toBeInTheDocument()
    expect(onDefinition.getByText("Refuse the request")).toBeInTheDocument()
    expect(
      (await row("Where they run", "on-service")).getByText(
        "guardrails.example",
      ),
    ).toBeInTheDocument()
    const onDeployment = await row("Where they run", "on-deployment")
    expect(
      onDeployment.getByText("the deployment guardrails service"),
    ).toBeInTheDocument()
    // A monitoring mandate serves whatever it stored for this.
    expect(onDeployment.getByText("Serve it unchecked")).toBeInTheDocument()
  })

  it("puts the consequence of a broken definition on its mandates and in a banner", async () => {
    mockApi({
      definitions: [
        organizationGuardrailDefinition({
          id: "d1",
          name: "prod-lakera",
          build_state: "failed",
        }),
      ],
      mandates: [
        organizationGuardrail({ id: "a", profile: "a", definition_id: "d1" }),
        organizationGuardrail({ id: "b", profile: "b", definition_id: "d1" }),
      ],
    })
    renderPage()

    expect(
      await screen.findByText(
        "2 mandated guardrails are not running, so the requests they cover are being served unchecked.",
      ),
    ).toBeInTheDocument()
    expect(
      (await row("Guardrails you have configured", "prod-lakera")).getByText(
        "Not running",
      ),
    ).toBeInTheDocument()
    expect(
      (await row("Where they run", "a")).getByText("Requests served unchecked"),
    ).toBeInTheDocument()
  })

  it("says before the click that a mandated definition cannot be removed", async () => {
    mockApi({
      definitions: [lakera],
      mandates: [
        organizationGuardrail({ id: "a", profile: "pii", definition_id: "d1" }),
      ],
    })
    renderPage()

    await userEvent.click(
      (await row("Guardrails you have configured", "prod-lakera")).getByRole(
        "button",
        { name: "Remove prod-lakera" },
      ),
    )

    expect(
      await screen.findByText(
        /prod-lakera is still mandated by pii, so it cannot be removed/,
      ),
    ).toBeInTheDocument()
  })

  it("pauses a mandate from its row, first among its actions", async () => {
    const calls = mockApi({
      mandates: [organizationGuardrail({ id: "a", profile: "pii" })],
    })
    renderPage()

    const mandate = await row("Where they run", "pii")
    const pause = mandate.getByRole("button", { name: "Pause pii" })
    // Pause first, then edit and remove, and no status column beside them.
    const actions = pause.closest("td")
    expect(
      [...(actions?.querySelectorAll("button") ?? [])].map((control) =>
        control.getAttribute("aria-label"),
      ),
    ).toEqual(["Pause pii", "Edit pii", "Remove pii"])
    expect(
      within(screen.getByRole("grid", { name: "Where they run" })).queryByRole(
        "columnheader",
        { name: "Status" },
      ),
    ).toBeNull()

    await userEvent.click(pause)

    await waitFor(() =>
      expect(calls.find((call) => call.method === "PATCH")?.body).toEqual({
        enabled: false,
      }),
    )
  })

  it("says what removing a blocking mandate serves", async () => {
    mockApi({
      mandates: [
        organizationGuardrail({ id: "a", profile: "pii", mode: "block" }),
      ],
    })
    renderPage()

    await userEvent.click(
      (await row("Where they run", "pii")).getByRole("button", {
        name: "Remove pii",
      }),
    )

    expect(
      await screen.findByText(/Requests it would have blocked are served/),
    ).toBeInTheDocument()
  })

  it("opens a mandate for editing in the mandate dialog", async () => {
    mockApi({
      mandates: [organizationGuardrail({ id: "a", profile: "pii" })],
    })
    renderPage()

    await userEvent.click(
      (await row("Where they run", "pii")).getByRole("button", {
        name: "Edit pii",
      }),
    )

    expect(
      await within(await screen.findByRole("dialog")).findByRole("button", {
        name: "Save mandate",
      }),
    ).toBeInTheDocument()
  })
})
