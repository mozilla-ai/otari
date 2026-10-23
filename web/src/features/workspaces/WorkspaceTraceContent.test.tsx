import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactNode } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { WorkspaceTraceContent } from "@/features/workspaces/WorkspaceTraceContent"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, workspace } from "@/tests/fixtures"

interface Sent {
  url: string
  method: string
  body: unknown
}

function mockApi(): Sent[] {
  const sent: Sent[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    sent.push({
      url: String(input),
      method: (init?.method ?? "GET").toUpperCase(),
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    return new Response(JSON.stringify(workspace()), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
  return sent
}

function renderToggle(ui: ReactNode, available = true) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider
        value={bootstrap({ trace_content_available: available })}
      >
        {ui}
      </DeploymentProvider>
    </QueryClientProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("WorkspaceTraceContent", () => {
  it("says who can read the content before anyone turns it on", () => {
    mockApi()
    renderToggle(<WorkspaceTraceContent workspace={workspace()} />)

    expect(
      screen.getByText(/only the person whose session it is can read it/),
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
    renderToggle(<WorkspaceTraceContent workspace={workspace()} />)

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
    renderToggle(
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

  it("cannot be turned on where the deployment keeps no content", () => {
    mockApi()
    renderToggle(<WorkspaceTraceContent workspace={workspace()} />, false)

    expect(
      screen.getByRole("switch", {
        name: "Keep what agent sessions were about",
      }),
    ).toBeDisabled()
    expect(
      screen.getByText(/This deployment does not keep session content/),
    ).toBeInTheDocument()
  })
})
