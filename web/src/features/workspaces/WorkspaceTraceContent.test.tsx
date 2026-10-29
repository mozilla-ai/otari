import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { TraceContentPolicy } from "@/client"

import { WorkspaceTraceContent } from "@/features/workspaces/WorkspaceTraceContent"
import { API_ROOT } from "@/shared/api/client"
import { workspace } from "@/tests/fixtures"

interface Sent {
  url: string
  method: string
  body: unknown
}

// What the policy route answers; `renderToggle` sets it.
let served: TraceContentPolicy

// Records every request but the policy read, which the component makes on its own.
function mockApi(refuse = ""): Sent[] {
  const sent: Sent[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const method = (init?.method ?? "GET").toUpperCase()
    if (String(input) === `${API_ROOT}/trace-content/policy`) {
      return new Response(JSON.stringify(served), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    }
    sent.push({
      url: String(input),
      method,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    if (method === refuse) {
      return new Response(
        JSON.stringify({ detail: "The database is unavailable" }),
        {
          status: 503,
          headers: { "Content-Type": "application/json" },
        },
      )
    }
    return new Response(JSON.stringify(workspace()), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
  return sent
}

function policy(
  overrides: Partial<TraceContentPolicy> = {},
): TraceContentPolicy {
  return {
    content_available: true,
    platform_break_glass: false,
    retention_days: 30,
    session_max_age_days: 90,
    ...overrides,
  }
}

// Rendered once the policy has loaded, which is what enables the switch where
// content is available.
async function renderToggle(
  ui: ReactNode,
  available = true,
  overrides: Partial<TraceContentPolicy> = {},
) {
  served = policy({ content_available: available, ...overrides })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const rendered = render(
    <QueryClientProvider client={client}>{ui}</QueryClientProvider>,
  )
  await waitFor(() =>
    expect(vi.mocked(globalThis.fetch)).toHaveBeenCalledWith(
      `${API_ROOT}/trace-content/policy`,
      expect.anything(),
    ),
  )
  if (available) {
    await waitFor(() =>
      expect(
        screen.getByRole("switch", {
          name: "Keep what agent sessions were about",
        }),
      ).toBeEnabled(),
    )
  }
  return rendered
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("WorkspaceTraceContent", () => {
  it("says who can read the content before anyone turns it on", async () => {
    mockApi()
    await renderToggle(<WorkspaceTraceContent workspace={workspace()} />)

    expect(
      screen.getByText(/Otari shows it only to the person whose session it is/),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/including its admins, see that a session ran/),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("switch", {
        name: "Keep what agent sessions were about",
      }),
    ).not.toBeChecked()
  })

  it("asks before turning it on, then turns it on", async () => {
    const sent = mockApi()
    await renderToggle(<WorkspaceTraceContent workspace={workspace()} />)

    await userEvent.click(
      screen.getByRole("switch", {
        name: "Keep what agent sessions were about",
      }),
    )
    expect(sent).toEqual([])
    await userEvent.click(screen.getByRole("button", { name: "Keep content" }))

    expect(sent).toEqual([
      {
        url: `${API_ROOT}/workspaces/${workspace().id}`,
        method: "PATCH",
        body: { trace_content_enabled: true },
      },
    ])
  })

  it("deletes what is stored only when asked to as it turns off", async () => {
    const sent = mockApi()
    await renderToggle(
      <WorkspaceTraceContent
        workspace={workspace({ trace_content_enabled: true })}
      />,
    )

    await userEvent.click(
      screen.getByRole("switch", {
        name: "Keep what agent sessions were about",
      }),
    )
    await userEvent.click(
      screen.getByRole("checkbox", {
        name: "Also delete the content already stored",
      }),
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Stop keeping content" }),
    )

    expect(sent.map(({ method, url }) => [method, url])).toEqual([
      ["PATCH", `${API_ROOT}/workspaces/${workspace().id}`],
      ["DELETE", `${API_ROOT}/workspaces/${workspace().id}/trace-content`],
    ])
  })

  it("cannot be turned on where the deployment keeps no content", async () => {
    mockApi()
    await renderToggle(<WorkspaceTraceContent workspace={workspace()} />, false)

    expect(
      screen.getByRole("switch", {
        name: "Keep what agent sessions were about",
      }),
    ).toBeDisabled()
    expect(
      screen.getByText(/This deployment does not keep session content/),
    ).toBeInTheDocument()
  })

  it("says when kept content is deleted", async () => {
    mockApi()
    await renderToggle(<WorkspaceTraceContent workspace={workspace()} />)

    expect(
      screen.getByText(
        /30 days after its last activity, and 90 days after it started/,
      ),
    ).toBeInTheDocument()
  })

  it("deletes what is stored while content is off", async () => {
    const sent = mockApi()
    await renderToggle(<WorkspaceTraceContent workspace={workspace()} />, false)

    await userEvent.click(
      screen.getByRole("button", { name: "Delete stored session content" }),
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Delete content" }),
    )

    expect(sent.map(({ method, url }) => [method, url])).toEqual([
      ["DELETE", `${API_ROOT}/workspaces/${workspace().id}/trace-content`],
    ])
  })

  it("keeps a failed delete on the dialog", async () => {
    mockApi("DELETE")
    await renderToggle(<WorkspaceTraceContent workspace={workspace()} />)

    await userEvent.click(
      screen.getByRole("button", { name: "Delete stored session content" }),
    )
    await userEvent.click(
      screen.getByRole("button", { name: "Delete content" }),
    )

    expect(
      await screen.findAllByText(/The database is unavailable/),
    ).not.toHaveLength(0)
    expect(
      screen.getByRole("button", { name: "Delete content" }),
    ).toBeInTheDocument()
  })

  it("says owners can open content by break-glass, and the operator only where it is on", async () => {
    mockApi()
    const { unmount } = await renderToggle(
      <WorkspaceTraceContent workspace={workspace()} />,
    )
    expect(
      screen.getByText(
        /organization's owners can still open it without a share/,
      ),
    ).toBeInTheDocument()
    expect(screen.queryByText(/where required by law/)).not.toBeInTheDocument()
    unmount()

    await renderToggle(
      <WorkspaceTraceContent workspace={workspace()} />,
      true,
      {
        platform_break_glass: true,
      },
    )
    expect(
      screen.getByText(/where required by law, recorded with a reference/),
    ).toBeInTheDocument()
  })

  it("reads the policy from its signed-in route, and offers nothing until it has", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async () =>
        new Response(JSON.stringify(policy({ retention_days: 14 })), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    )
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    render(
      <QueryClientProvider client={client}>
        <WorkspaceTraceContent workspace={workspace()} />
      </QueryClientProvider>,
    )
    const toggle = screen.getByRole("switch", {
      name: "Keep what agent sessions were about",
    })
    expect(toggle).toBeDisabled()

    expect(
      await screen.findByText(/14 days after its last activity/),
    ).toBeInTheDocument()
    expect(toggle).toBeEnabled()
    expect(String(vi.mocked(globalThis.fetch).mock.calls[0][0])).toBe(
      `${API_ROOT}/trace-content/policy`,
    )
  })
})
