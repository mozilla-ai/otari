import { readFileSync } from "node:fs"
import { join } from "node:path"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceWebSearchConfig } from "@/client"
import {
  MAX_RESULTS,
  WorkspaceWebSearchCard,
} from "@/features/tools/WorkspaceWebSearchCard"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { organizationContext, workspaceWebSearchConfig } from "@/tests/fixtures"
import { pickOption, selectTrigger } from "@/tests/select"

const ALPHA = "11111111-1111-1111-1111-111111111111"

function mockApi({
  memberships = [{ workspace_id: ALPHA, name: "Alpha", role: "admin" }],
  config = workspaceWebSearchConfig({ workspace_id: ALPHA }),
}: {
  memberships?: { workspace_id: string; name: string; role: string }[]
  config?: WorkspaceWebSearchConfig
} = {}) {
  const calls: { url: string; method: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = init?.method ?? "GET"
    if (url.includes("/web-search")) {
      calls.push({
        url,
        method,
        body:
          typeof init?.body === "string" ? JSON.parse(init.body) : init?.body,
      })
      return Response.json(config)
    }
    return Response.json(
      organizationContext({ workspace_memberships: memberships }),
    )
  })
  return calls
}

// The form hydrates from the row once it arrives, so a test that types before
// then would have its input overwritten by the load. The Save button is
// disabled while the query is in flight, which is the signal to wait on.
async function renderLoaded() {
  renderCard()
  await waitFor(() =>
    expect(screen.getByRole("button", { name: "Save" })).toBeEnabled(),
  )
}

function renderCard() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <WorkspaceWebSearchCard onSaved={() => {}} />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
  )
}

describe("WorkspaceWebSearchCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("reads an unconfigured workspace as using the deployment default", async () => {
    mockApi()
    renderCard()

    expect(await screen.findByText("NOTHING SET")).toBeInTheDocument()
    expect(selectTrigger("Web search")).toHaveTextContent("Deployment default")
  })

  it("shows a stored row's stance, ceiling and domain lists", async () => {
    mockApi({
      config: workspaceWebSearchConfig({
        workspace_id: ALPHA,
        configured: true,
        enabled: false,
        max_results: 3,
        allowed_domains: ["arxiv.org", "wikipedia.org"],
        blocked_domains: ["example.invalid"],
      }),
    })
    await renderLoaded()

    expect(selectTrigger("Web search")).toHaveTextContent(
      "Blocked (tool and /v1/search)",
    )
    expect(screen.getByLabelText("Max results")).toHaveValue("3")
    expect(screen.getByLabelText("Allowed domains")).toHaveValue(
      "arxiv.org, wikipedia.org",
    )
    expect(screen.getByLabelText("Blocked domains")).toHaveValue(
      "example.invalid",
    )
    expect(screen.queryByText("Nothing set")).not.toBeInTheDocument()
  })

  it("saves the stance, the ceiling and the domains the operator typed", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    await renderLoaded()

    await pickOption(user, "Web search", "Allowed")
    await user.type(screen.getByLabelText("Max results"), "4")
    await user.type(
      screen.getByLabelText("Blocked domains"),
      "Bad.Example, , other.example",
    )
    await user.click(screen.getByRole("button", { name: "Save" }))

    const put = calls.find((call) => call.method === "PUT")
    expect(put?.body).toEqual({
      enabled: true,
      max_results: 4,
      purpose_hint: null,
      allowed_domains: null,
      // Normalized and de-blanked here so the server is not asked to store a
      // domain named "".
      blocked_domains: ["bad.example", "other.example"],
      provider_options: null,
    })
  })

  it("preserves provider options it has no form for", async () => {
    // The bag is set over the API, and this is a PUT: sending nothing would
    // silently clear it on the next save from the dashboard.
    const calls = mockApi({
      config: workspaceWebSearchConfig({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        provider_options: { search_depth: "advanced" },
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("button", { name: "Save" }))

    const put = calls.find((call) => call.method === "PUT")
    expect(put?.body).toMatchObject({
      provider_options: { search_depth: "advanced" },
    })
  })

  it("clears the row rather than storing one when set back to the deployment default", async () => {
    const calls = mockApi({
      config: workspaceWebSearchConfig({
        workspace_id: ALPHA,
        configured: true,
        enabled: false,
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await pickOption(user, "Web search", "Deployment default")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(calls.some((call) => call.method === "DELETE")).toBe(true)
    expect(calls.some((call) => call.method === "PUT")).toBe(false)
  })

  it("refuses a ceiling the backend could never honor without asking the server", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    await renderLoaded()

    await pickOption(user, "Web search", "Allowed")
    await user.type(screen.getByLabelText("Max results"), "500")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Max results must be a whole number from 1 to 20.",
    )
    expect(calls.some((call) => call.method === "PUT")).toBe(false)
  })

  it("refuses a domain that is not a bare hostname without asking the server", async () => {
    // The server matches an entry against a result URL's hostname, so a scheme
    // or a path matches nothing: on a block-list that is a guardrail that reads
    // as set and blocks nothing.
    const calls = mockApi()
    const user = userEvent.setup()
    await renderLoaded()

    await pickOption(user, "Web search", "Allowed")
    await user.type(
      screen.getByLabelText("Blocked domains"),
      "https://evil.example",
    )
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "is not a bare hostname",
    )
    expect(calls.some((call) => call.method === "PUT")).toBe(false)
  })

  it("keeps Save disabled when the initial read failed, so a click cannot drop a stored row", async () => {
    // A failed GET leaves isLoading false and config undefined, so the form sits
    // at its initial "Deployment default" stance over a workspace that may have
    // a row. Saving from there would DELETE it.
    const calls: { url: string; method: string }[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.includes("/web-search")) {
        calls.push({ url, method: init?.method ?? "GET" })
        return new Response("boom", { status: 500 })
      }
      return Response.json(
        organizationContext({
          workspace_memberships: [
            { workspace_id: ALPHA, name: "Alpha", role: "admin" },
          ],
        }),
      )
    })
    renderCard()

    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Save" })).toBeDisabled(),
    )
    expect(calls.some((call) => call.method === "DELETE")).toBe(false)
  })

  it("says the in-loop tool is unavailable when the deployment has no backend, and that blocking still bites", async () => {
    mockApi({
      config: workspaceWebSearchConfig({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        web_search_configured: false,
      }),
    })
    renderCard()

    // Both halves: the capability ceiling is about the in-loop backend only, so
    // the banner must not claim the workspace's switch does nothing. It still
    // gates POST /v1/search, which runs off the search tools and not this URL.
    expect(
      await screen.findByText(/no in-loop search backend configured/i),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/still takes effect on POST \/v1\/search/i),
    ).toBeInTheDocument()
  })

  it("does not read the row at all for a member who cannot manage the workspace", async () => {
    // Reads take the management role server-side, so asking would earn a 403.
    // The card says who can set it instead of rendering a form over an error.
    const configRequests: string[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes("/web-search")) {
        configRequests.push(url)
        return new Response("forbidden", { status: 403 })
      }
      return Response.json(
        organizationContext({
          role: "member",
          workspace_memberships: [
            { workspace_id: ALPHA, name: "Alpha", role: "member" },
          ],
        }),
      )
    })
    renderCard()

    expect(
      await screen.findByText(/set by an owner or admin/i),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Save" }),
    ).not.toBeInTheDocument()
    expect(configRequests).toEqual([])
  })

  it("keeps its ceiling equal to the one the server enforces", () => {
    // Duplicated here because `openapi-typescript` drops `maximum` when it
    // generates `schema.ts`, so the spec is the only place both sides can be
    // compared. Without this, raising the backend cap would leave the form
    // quietly refusing values the server would take.
    const spec = JSON.parse(
      readFileSync(
        join(import.meta.dirname, "../../../../docs/public/openapi.json"),
        "utf8",
      ),
    ) as {
      components: {
        schemas: {
          WorkspaceWebSearchConfigUpdate: {
            properties: Record<string, { anyOf?: { maximum?: number }[] }>
          }
        }
      }
    }
    const properties =
      spec.components.schemas.WorkspaceWebSearchConfigUpdate.properties
    const ceiling = (field: string) =>
      properties[field]?.anyOf?.find((arm) => arm.maximum !== undefined)
        ?.maximum

    expect(ceiling("max_results")).toBe(MAX_RESULTS)
  })
})
