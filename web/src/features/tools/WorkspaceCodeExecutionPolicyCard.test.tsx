import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceCodeExecutionPolicy } from "@/client"
import { WorkspaceCodeExecutionPolicyCard } from "@/features/tools/WorkspaceCodeExecutionPolicyCard"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import {
  organizationContext,
  workspaceCodeExecutionPolicy,
} from "@/tests/fixtures"

const ALPHA = "11111111-1111-1111-1111-111111111111"
const SWITCH = "Allow code execution"

function mockApi({
  memberships = [{ workspace_id: ALPHA, name: "Alpha", role: "admin" }],
  policy = workspaceCodeExecutionPolicy({ workspace_id: ALPHA }),
}: {
  memberships?: { workspace_id: string; name: string; role: string }[]
  policy?: WorkspaceCodeExecutionPolicy
} = {}) {
  const calls: { url: string; method: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = init?.method ?? "GET"
    if (url.includes("/code-execution-policy")) {
      calls.push({
        url,
        method,
        body:
          typeof init?.body === "string" ? JSON.parse(init.body) : init?.body,
      })
      return Response.json(policy)
    }
    return Response.json(
      organizationContext({ workspace_memberships: memberships }),
    )
  })
  return calls
}

// Every control is disabled until the policy has arrived, so a save cannot race
// the load that would overwrite the field under it. That is the signal to wait
// on before typing.
async function renderLoaded() {
  renderCard()
  await waitFor(() =>
    expect(screen.getByRole("switch", { name: SWITCH })).toBeEnabled(),
  )
}

/** The one PUT body, once the write has gone out. */
async function putBody(calls: { method: string; body: unknown }[]) {
  await waitFor(() =>
    expect(calls.some((call) => call.method === "PUT")).toBe(true),
  )
  return calls.filter((call) => call.method === "PUT").at(-1)?.body
}

function renderCard(leading?: ReactNode, isHosted = false) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <WorkspaceCodeExecutionPolicyCard
          leading={leading}
          isHosted={isHosted}
        />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
  )
}

describe("WorkspaceCodeExecutionPolicyCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("reads a workspace with no policy as allowed, with nothing else to set", async () => {
    mockApi()
    await renderLoaded()

    expect(screen.getByRole("switch", { name: SWITCH })).toBeChecked()
    expect(screen.queryByRole("textbox")).toBeNull()
    expect(screen.queryByRole("button", { name: /Advanced/ })).toBeNull()
  })

  it("shows a blocked policy switched off", async () => {
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: false,
      }),
    })
    await renderLoaded()

    expect(screen.getByRole("switch", { name: SWITCH })).not.toBeChecked()
  })

  it("blocks the moment the switch flips, with no Save button anywhere", async () => {
    const calls = mockApi()
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("switch", { name: SWITCH }))

    expect(await putBody(calls)).toEqual({
      enabled: false,
      default_purpose_hint: null,
      max_iterations: null,
      exec_timeout_s: null,
      image: null,
      tools: null,
      executor: null,
    })
    expect(screen.queryByRole("button", { name: "Save" })).toBeNull()
  })

  it("keeps fields set through the API when blocking", async () => {
    const calls = mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        max_iterations: 3,
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("switch", { name: SWITCH }))

    expect(await putBody(calls)).toMatchObject({
      enabled: false,
      max_iterations: 3,
    })
  })

  it("drops a blocked policy that narrows nothing when allowed again", async () => {
    const calls = mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: false,
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("switch", { name: SWITCH }))

    await waitFor(() =>
      expect(calls.some((call) => call.method === "DELETE")).toBe(true),
    )
    expect(calls.some((call) => call.method === "PUT")).toBe(false)
  })

  it("keeps a blocked policy's API-set limits when allowed again", async () => {
    const calls = mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: false,
        max_iterations: 3,
        executor: "provider",
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    await user.click(screen.getByRole("switch", { name: SWITCH }))

    expect(await putBody(calls)).toMatchObject({
      enabled: true,
      max_iterations: 3,
      executor: "provider",
    })
    expect(calls.some((call) => call.method === "DELETE")).toBe(false)
  })

  it("warns about a policy naming a tool no longer served, and clears only that", async () => {
    // Admission refuses such a policy, so the switch alone would read "on"
    // over a workspace whose every request fails.
    const calls = mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        available_tools: ["code_execution"],
        tools: ["bash_code_execution"],
        max_iterations: 3,
      }),
    })
    const user = userEvent.setup()
    await renderLoaded()

    expect(
      screen.getByText(/no longer offers, so its requests are refused/i),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Clear it" }))

    expect(await putBody(calls)).toMatchObject({
      enabled: true,
      tools: null,
      max_iterations: 3,
    })
    expect(calls.some((call) => call.method === "DELETE")).toBe(false)
  })

  it("warns about a pin to an image the operator withdrew", async () => {
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: true,
        allowed_images: ["mzdotai/otari-sandbox-container:latest"],
        image: "ghcr.io/acme/withdrawn:1",
      }),
    })
    await renderLoaded()

    expect(
      screen.getByText(/no longer offers, so its requests are refused/i),
    ).toBeInTheDocument()
  })

  it("does not warn about a stale policy that is blocked anyway", async () => {
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        configured: true,
        enabled: false,
        available_tools: ["code_execution"],
        tools: ["bash_code_execution"],
      }),
    })
    await renderLoaded()

    expect(screen.queryByText(/no longer offers/i)).toBeNull()
  })

  it("switches off and explains when the deployment has no sandbox", async () => {
    mockApi({
      policy: workspaceCodeExecutionPolicy({
        workspace_id: ALPHA,
        sandbox_configured: false,
      }),
    })
    renderCard()

    expect(
      await screen.findByText(/no sandbox configured/i),
    ).toBeInTheDocument()
    const toggle = screen.getByRole("switch", { name: SWITCH })
    expect(toggle).not.toBeChecked()
    expect(toggle).toBeDisabled()
  })

  describe("on a hosted control plane", () => {
    // There a workspace with no policy may not run code, and the sandbox is
    // the platform's, so this process's `sandbox_configured` decides nothing.
    async function renderHosted() {
      renderCard(undefined, true)
      await waitFor(() =>
        expect(screen.getByRole("switch", { name: SWITCH })).toBeEnabled(),
      )
    }

    it("reads no policy as off, and never locks the switch over a sandbox it does not own", async () => {
      mockApi({
        policy: workspaceCodeExecutionPolicy({
          workspace_id: ALPHA,
          sandbox_configured: false,
        }),
      })
      await renderHosted()

      expect(screen.getByRole("switch", { name: SWITCH })).not.toBeChecked()
      expect(screen.queryByText(/no sandbox configured/i)).toBeNull()
    })

    it("turns on by storing an enabled policy, never by deleting one", async () => {
      const calls = mockApi()
      const user = userEvent.setup()
      await renderHosted()

      await user.click(screen.getByRole("switch", { name: SWITCH }))

      expect(await putBody(calls)).toMatchObject({ enabled: true })
      expect(calls.some((call) => call.method === "DELETE")).toBe(false)
    })

    it("turns off by storing a disabled policy", async () => {
      const calls = mockApi({
        policy: workspaceCodeExecutionPolicy({
          workspace_id: ALPHA,
          configured: true,
          enabled: true,
        }),
      })
      const user = userEvent.setup()
      await renderHosted()

      expect(screen.getByRole("switch", { name: SWITCH })).toBeChecked()
      await user.click(screen.getByRole("switch", { name: SWITCH }))

      expect(await putBody(calls)).toMatchObject({ enabled: false })
      expect(calls.some((call) => call.method === "DELETE")).toBe(false)
    })

    it("does not call a pinned image withdrawn against the control plane's own list", async () => {
      mockApi({
        policy: workspaceCodeExecutionPolicy({
          workspace_id: ALPHA,
          configured: true,
          enabled: true,
          allowed_images: [],
          image: "ghcr.io/acme/sandbox:2",
        }),
      })
      await renderHosted()

      expect(screen.queryByText(/no longer offers/i)).toBeNull()
    })

    it("shows a member the platform's answer for their workspace", async () => {
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
        String(input).includes("/playground/tools")
          ? Response.json({
              code_execution: {
                configured: true,
                enabled: false,
                reason: "Not turned on for this workspace.",
              },
            })
          : Response.json(
              organizationContext({
                role: "member",
                workspace_memberships: [
                  { workspace_id: ALPHA, name: "Alpha", role: "member" },
                ],
              }),
            ),
      )
      renderCard(<div>tool row</div>, true)

      expect(await screen.findByText("Allowed in Alpha")).toBeInTheDocument()
      expect(screen.getByText("No")).toBeInTheDocument()
      expect(screen.queryByRole("switch")).toBeNull()
    })
  })

  it("renders nothing for a member when there is neither a tool row nor a sandbox", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
      String(input).includes("/playground/tools")
        ? Response.json({
            code_execution: { configured: false, enabled: false, reason: null },
          })
        : Response.json(
            organizationContext({
              role: "member",
              workspace_memberships: [
                { workspace_id: ALPHA, name: "Alpha", role: "member" },
              ],
            }),
          ),
    )
    const { container } = renderCard()

    await waitFor(() => expect(container).toBeEmptyDOMElement())
  })

  it("names the workspace and its organization", async () => {
    mockApi()
    await renderLoaded()

    expect(screen.getByText("Allow in Alpha")).toBeInTheDocument()
    expect(screen.getByText(/^Workspace in /)).toBeInTheDocument()
  })

  it.each([
    [true, "Yes"],
    [false, "No"],
  ])(
    "shows a member whether their workspace allows it, without reading the policy (enabled %s)",
    async (enabled, answer) => {
      // Policy reads take the management role server-side, so asking would earn
      // a 403. The playground's per-workspace answer is what a member may read.
      const policyRequests: string[] = []
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = String(input)
        if (url.includes("/code-execution-policy")) {
          policyRequests.push(url)
          return new Response("forbidden", { status: 403 })
        }
        if (url.includes("/playground/tools")) {
          return Response.json({
            web_search: { configured: false, enabled: false, reason: null },
            code_execution: {
              configured: true,
              enabled,
              reason: enabled ? null : "Turned off for this workspace.",
            },
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
      renderCard(<div>tool row</div>)

      expect(await screen.findByText("Allowed in Alpha")).toBeInTheDocument()
      expect(screen.getByText(answer)).toBeInTheDocument()
      expect(screen.getByText("tool row")).toBeInTheDocument()
      expect(screen.queryByRole("switch")).toBeNull()
      expect(policyRequests).toEqual([])
    },
  )
})
