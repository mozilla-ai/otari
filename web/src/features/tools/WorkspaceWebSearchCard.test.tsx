import { readFileSync } from "node:fs"
import { join } from "node:path"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceWebSearchConfig } from "@/client"
import {
  MAX_RESULTS,
  WorkspaceWebSearchCard,
} from "@/features/tools/WorkspaceWebSearchCard"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { organizationContext, workspaceWebSearchConfig } from "@/tests/fixtures"

const ALPHA = "11111111-1111-1111-1111-111111111111"
const SWITCH = "Allow web access"

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

function renderCard({
  leading,
  isHosted = false,
  isAvailable = true,
}: {
  leading?: ReactNode
  isHosted?: boolean
  isAvailable?: boolean
} = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <WorkspaceWebSearchCard
          leading={leading}
          isHosted={isHosted}
          isAvailable={isAvailable}
        />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
  )
}

// Every control is disabled until the row has arrived, so a save cannot race
// the load that would overwrite the field under it. That is the signal to wait
// on before typing.
async function renderLoaded(opts?: Parameters<typeof renderCard>[0]) {
  renderCard(opts)
  await waitFor(() =>
    expect(screen.getByRole("switch", { name: SWITCH })).toBeEnabled(),
  )
}

/** Loaded, with the narrowing rows under Advanced shown. */
async function renderOpen(user: ReturnType<typeof userEvent.setup>) {
  await renderLoaded()
  await user.click(screen.getByRole("button", { name: "Advanced" }))
}

/** The one PUT body, once the write has gone out. */
async function putBody(calls: { method: string; body: unknown }[]) {
  await waitFor(() =>
    expect(calls.some((call) => call.method === "PUT")).toBe(true),
  )
  return calls.filter((call) => call.method === "PUT").at(-1)?.body
}

const allowed = (extra: Partial<WorkspaceWebSearchConfig> = {}) =>
  workspaceWebSearchConfig({
    workspace_id: ALPHA,
    configured: true,
    enabled: true,
    ...extra,
  })

describe("WorkspaceWebSearchCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("reads a workspace with no row as allowed, with the narrowing folded away", async () => {
    mockApi()
    await renderLoaded()

    expect(screen.getByRole("switch", { name: SWITCH })).toBeChecked()
    expect(screen.getByRole("button", { name: "Advanced" })).toHaveAttribute(
      "aria-expanded",
      "false",
    )
    // The purpose hint is left to the API.
    expect(screen.queryByText("Prompt hint")).toBeNull()
  })

  it("shows a blocked row switched off, with nothing to narrow", async () => {
    mockApi({ config: allowed({ enabled: false }) })
    const user = userEvent.setup()
    await renderOpen(user)

    expect(screen.getByRole("switch", { name: SWITCH })).not.toBeChecked()
    expect(
      screen.getByLabelText("Max results for this workspace"),
    ).toBeDisabled()
  })

  it("shows a stored row's ceiling and domain lists", async () => {
    mockApi({
      config: allowed({
        max_results: 3,
        allowed_domains: ["arxiv.org", "wikipedia.org"],
        blocked_domains: ["example.invalid"],
      }),
    })
    const user = userEvent.setup()
    await renderOpen(user)

    expect(screen.getByLabelText("Max results for this workspace")).toHaveValue(
      "3",
    )
    expect(screen.getByLabelText("Allowed domains")).toHaveValue(
      "arxiv.org, wikipedia.org",
    )
    expect(screen.getByLabelText("Blocked domains")).toHaveValue(
      "example.invalid",
    )
  })

  it("blocks the moment the switch flips, with no Save button anywhere", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("switch", { name: SWITCH }))

    expect(await putBody(calls)).toMatchObject({ enabled: false })
    expect(screen.queryByRole("button", { name: "Save" })).toBeNull()
  })

  it("drops a blocked row that narrows nothing when allowed again", async () => {
    const calls = mockApi({ config: allowed({ enabled: false }) })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("switch", { name: SWITCH }))

    await waitFor(() =>
      expect(calls.some((call) => call.method === "DELETE")).toBe(true),
    )
    expect(calls.some((call) => call.method === "PUT")).toBe(false)
  })

  it("keeps a blocked row's narrowing, including fields set over the API, when allowed again", async () => {
    const calls = mockApi({
      config: allowed({
        enabled: false,
        purpose_hint: "Official docs",
        provider_options: { search_depth: "advanced" },
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("switch", { name: SWITCH }))

    expect(await putBody(calls)).toMatchObject({
      enabled: true,
      purpose_hint: "Official docs",
      provider_options: { search_depth: "advanced" },
    })
    expect(calls.some((call) => call.method === "DELETE")).toBe(false)
  })

  it("saves a ceiling and a domain list when the field is left", async () => {
    const calls = mockApi({ config: allowed() })
    const user = userEvent.setup()
    await renderOpen(user)

    await user.type(
      screen.getByLabelText("Max results for this workspace"),
      "4",
    )
    await user.tab()
    await waitFor(() =>
      expect(calls.find((call) => call.method === "PUT")?.body).toMatchObject({
        enabled: true,
        max_results: 4,
      }),
    )

    await user.type(
      screen.getByLabelText("Blocked domains"),
      "Bad.Example, , other.example",
    )
    await user.tab()

    await waitFor(() =>
      expect(
        calls.filter((call) => call.method === "PUT").at(-1)?.body,
      ).toMatchObject({
        // Normalized and de-blanked here so the server is not asked to store a
        // domain named "".
        blocked_domains: ["bad.example", "other.example"],
      }),
    )
  })

  it("narrows a workspace with no row by storing an enabled one", async () => {
    // No row reads as on here, so the first narrowing creates the row on.
    const calls = mockApi()
    const user = userEvent.setup()
    await renderOpen(user)

    await user.type(
      screen.getByLabelText("Allowed domains"),
      "arxiv.org{Enter}",
    )

    expect(await putBody(calls)).toMatchObject({
      enabled: true,
      allowed_domains: ["arxiv.org"],
    })
  })

  it("refuses a ceiling the backend could never honor without asking the server", async () => {
    const calls = mockApi({ config: allowed() })
    const user = userEvent.setup()
    await renderOpen(user)

    await user.type(
      screen.getByLabelText("Max results for this workspace"),
      "500",
    )
    await user.tab()

    expect(
      await screen.findByText(
        `A whole number of results from 1 to ${MAX_RESULTS}.`,
      ),
    ).toBeInTheDocument()
    expect(calls.some((call) => call.method === "PUT")).toBe(false)
  })

  it("refuses a domain that is not a bare hostname without asking the server", async () => {
    // The server matches an entry against a result URL's hostname, so a scheme
    // or a path matches nothing: on a block-list that is a guardrail that reads
    // as set and blocks nothing.
    const calls = mockApi({ config: allowed() })
    const user = userEvent.setup()
    await renderOpen(user)

    await user.type(
      screen.getByLabelText("Blocked domains"),
      "https://evil.example",
    )
    await user.tab()

    expect(
      await screen.findByText(/is not a bare hostname/),
    ).toBeInTheDocument()
    expect(calls.some((call) => call.method === "PUT")).toBe(false)
  })

  it("keeps every control disabled when the initial read failed, so a change cannot drop a stored row", async () => {
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
    const user = userEvent.setup()
    renderCard()

    await waitFor(() =>
      expect(screen.getByRole("switch", { name: SWITCH })).toBeDisabled(),
    )
    await user.click(screen.getByRole("button", { name: "Advanced" }))
    expect(
      screen.getByLabelText("Max results for this workspace"),
    ).toBeDisabled()
    expect(calls.some((call) => call.method === "DELETE")).toBe(false)
  })

  it("switches off and explains when neither tool can run here", async () => {
    mockApi()
    renderCard({ isAvailable: false })

    expect(
      await screen.findByText(/Neither tool can run on this deployment/),
    ).toBeInTheDocument()
    const toggle = screen.getByRole("switch", { name: SWITCH })
    expect(toggle).not.toBeChecked()
    expect(toggle).toBeDisabled()
  })

  describe("on a hosted control plane", () => {
    // There a workspace with no row may not reach the web.
    async function renderHosted() {
      renderCard({ isHosted: true })
      await waitFor(() =>
        expect(screen.getByRole("switch", { name: SWITCH })).toBeEnabled(),
      )
    }

    it("reads no row as off", async () => {
      mockApi()
      await renderHosted()

      expect(screen.getByRole("switch", { name: SWITCH })).not.toBeChecked()
    })

    it("shows a member the tools alone, without the playground's reading of a missing row", async () => {
      const reads: string[] = []
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = String(input)
        if (url.includes("/playground/tools") || url.includes("/web-search")) {
          reads.push(url)
          return new Response("unexpected", { status: 500 })
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
      renderCard({ leading: <div>tool rows</div>, isHosted: true })

      expect(await screen.findByText("tool rows")).toBeInTheDocument()
      expect(screen.queryByText("Allowed in Alpha")).toBeNull()
      expect(reads).toEqual([])
    })

    it("turns on by storing an enabled row, never by deleting one", async () => {
      const calls = mockApi()
      const user = userEvent.setup()
      await renderHosted()

      await user.click(screen.getByRole("switch", { name: SWITCH }))

      expect(await putBody(calls)).toMatchObject({ enabled: true })
      expect(calls.some((call) => call.method === "DELETE")).toBe(false)
    })

    it("turns off by storing a disabled row", async () => {
      const calls = mockApi({ config: allowed() })
      const user = userEvent.setup()
      await renderHosted()

      expect(screen.getByRole("switch", { name: SWITCH })).toBeChecked()
      await user.click(screen.getByRole("switch", { name: SWITCH }))

      expect(await putBody(calls)).toMatchObject({ enabled: false })
      expect(calls.some((call) => call.method === "DELETE")).toBe(false)
    })
  })

  it.each([
    [true, "Yes"],
    [false, "No"],
  ])(
    "shows a member whether their workspace allows it, without reading the row (enabled %s)",
    async (enabled, answer) => {
      // Row reads take the management role server-side, so asking would earn a
      // 403. The playground's per-workspace answer is what a member may read.
      const configRequests: string[] = []
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = String(input)
        if (url.includes("/web-search")) {
          configRequests.push(url)
          return new Response("forbidden", { status: 403 })
        }
        if (url.includes("/playground/tools")) {
          return Response.json({
            web_search: {
              configured: true,
              enabled,
              reason: enabled ? null : "Turned off for this workspace.",
            },
            code_execution: { configured: false, enabled: false, reason: null },
            mcp_servers: [],
          })
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
      renderCard({ leading: <div>tool rows</div> })

      expect(await screen.findByText("Allowed in Alpha")).toBeInTheDocument()
      expect(screen.getByText(answer)).toBeInTheDocument()
      expect(screen.getByText("tool rows")).toBeInTheDocument()
      expect(screen.queryByRole("switch")).toBeNull()
      expect(configRequests).toEqual([])
    },
  )

  it("does not let a second row's save revert the first", async () => {
    // Autosave is what opens this: every control has its own save state, so two
    // rows can be in flight at once, and a PUT body built from the last value
    // the *query* returned still holds the pre-first-write state.
    const bodies: Record<string, unknown>[] = []
    let releaseFirst: (() => void) | undefined
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input)
      if (!url.includes("/web-search")) {
        return Response.json(
          organizationContext({
            workspace_memberships: [
              { workspace_id: ALPHA, name: "Alpha", role: "admin" },
            ],
          }),
        )
      }
      if ((init?.method ?? "GET") !== "PUT") {
        return Response.json(allowed())
      }
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>
      bodies.push(body)
      if (bodies.length === 1) {
        await new Promise<void>((resolve) => {
          releaseFirst = resolve
        })
      }
      return Response.json(allowed(body))
    })
    const user = userEvent.setup()
    await renderOpen(user)

    await user.type(screen.getByLabelText("Allowed domains"), "arxiv.org")
    await user.tab()
    await waitFor(() => expect(bodies).toHaveLength(1))

    // The first write has not answered yet, so the query still holds the row
    // without the allowed list on it.
    await user.type(screen.getByLabelText("Blocked domains"), "evil.example")
    await user.tab()
    releaseFirst?.()

    await waitFor(() => expect(bodies).toHaveLength(2))
    expect(bodies[1]).toMatchObject({
      allowed_domains: ["arxiv.org"],
      blocked_domains: ["evil.example"],
    })
  })

  it("sends only the writable half of the row", async () => {
    // The stored shape also carries workspace_id, configured, the server's own
    // web_search_configured and two timestamps. None of them belongs in a PUT.
    const calls = mockApi({ config: allowed() })
    const user = userEvent.setup()
    await renderOpen(user)

    await user.type(
      screen.getByLabelText("Max results for this workspace"),
      "4",
    )
    await user.tab()

    expect(Object.keys((await putBody(calls)) as object).sort()).toEqual([
      "allowed_domains",
      "blocked_domains",
      "enabled",
      "max_results",
      "provider_options",
      "purpose_hint",
    ])
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
