import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { TeamHome } from "@/features/home/TeamHome"
import { OverviewIndex } from "@/features/overview/OverviewPage"
import { RouterSavingsPage } from "@/features/routerSavings/RouterSavingsPage"
import { SessionsPage } from "@/features/traces/SessionsPage"
import {
  SelectedWorkspaceProvider,
  useSelectedWorkspace,
} from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { jsonResponse, mockApi } from "@/tests/activity"
import {
  bootstrap,
  organizationContext,
  SURFACES,
  subagentSavings,
  workspace,
} from "@/tests/fixtures"
import { LocationProbe, renderWithRouter } from "@/tests/router"
import { pickOption, selectTrigger } from "@/tests/select"

const PRODUCTION = workspace({ id: "ws-1", name: "Production" })
const STAGING = workspace({ id: "ws-2", name: "Staging" })

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
})

/**
 * The caller is in Production and Staging, and the shell has Production
 * selected, being the first. Returns every URL read, as it grows.
 */
function mockGateway({ workspacesFail = false } = {}): string[] {
  const urls: string[] = []
  const { mock } = mockApi()
  const answer = mock.getMockImplementation() as typeof fetch
  mock.mockImplementation(async (input, init) => {
    const url = String(input)
    urls.push(url)
    if (url.includes("/subagent-savings")) {
      return jsonResponse(subagentSavings())
    }
    // Something to call, so the first-request guide stays out of the way.
    if (url.endsWith("/providers")) {
      return jsonResponse({ providers: [{ provider: "openai" }] })
    }
    if (url.endsWith("/hosted-providers")) {
      return jsonResponse({ data: [], count: 0 })
    }
    if (url.includes("/spend-ceilings")) {
      return jsonResponse({ data: [], count: 0 })
    }
    if (/\/workspaces(\?|$)/.test(url)) {
      if (workspacesFail) return jsonResponse({ detail: "unavailable" }, 503)
      return jsonResponse({ data: [PRODUCTION, STAGING], count: 2 })
    }
    if (/\/organizations\/me$/.test(url)) {
      return jsonResponse(
        organizationContext({
          workspace_memberships: [PRODUCTION, STAGING].map((one) => ({
            workspace_id: one.id,
            name: one.name,
            role: "owner",
          })),
        }),
      )
    }
    return answer(input, init)
  })
  return urls
}

/** The `workspace_id` each read of `path` asked for, in order; "" for none. */
const scopesOf = (urls: string[], path: RegExp) =>
  urls
    .filter((url) => path.test(url))
    .map(
      (url) =>
        new URL(url, "http://localhost").searchParams.get("workspace_id") ?? "",
    )

const SAVINGS = /\/subagent-savings/
const SESSION_LIST = /\/traces\?/

// Stands in for the switcher, which lives in the app shell.
function ShellSwitcher() {
  const { select } = useSelectedWorkspace()
  return (
    <>
      <button type="button" onClick={() => select(PRODUCTION.id)}>
        Select Production in the shell
      </button>
      <button type="button" onClick={() => select(STAGING.id)}>
        Select Staging in the shell
      </button>
    </>
  )
}

function renderSessions(subject: ReactElement, url: string) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(subject, {
    url,
    shell: (
      <QueryClientProvider client={client}>
        <DeploymentProvider
          value={bootstrap({ surfaces: [...SURFACES, "traces"] })}
        >
          <SelectedWorkspaceProvider>
            <ShellSwitcher />
            <OverviewIndex />
            <LocationProbe />
          </SelectedWorkspaceProvider>
        </DeploymentProvider>
      </QueryClientProvider>
    ),
    routes: [
      ...(url.startsWith("/sessions")
        ? []
        : [{ path: "/sessions", element: <SessionsPage embedded /> }]),
      ...(url.startsWith("/router-savings")
        ? []
        : [{ path: "/router-savings", element: <RouterSavingsPage /> }]),
    ],
  })
}

const searchOf = () =>
  new URLSearchParams(
    (screen.getByTestId("location").textContent ?? "").split("?")[1] ?? "",
  )

describe("A Sessions view whose link names a workspace", () => {
  it("reads that workspace rather than the shell's, and names it", async () => {
    const urls = mockGateway()
    await renderSessions(
      <RouterSavingsPage />,
      "/router-savings?workspace=ws-2",
    )

    await waitFor(() =>
      expect(selectTrigger("Workspace")).toHaveTextContent("Staging"),
    )
    await waitFor(() => expect(scopesOf(urls, SAVINGS)).toContain("ws-2"))
    expect(scopesOf(urls, SAVINGS)).not.toContain("ws-1")
  })

  it("reads every workspace for an explicit All workspaces", async () => {
    const urls = mockGateway()
    await renderSessions(<RouterSavingsPage />, "/router-savings?workspace=all")

    await waitFor(() =>
      expect(selectTrigger("Workspace")).toHaveTextContent("All workspaces"),
    )
    await waitFor(() => expect(scopesOf(urls, SAVINGS)).toEqual([""]))
  })

  it("follows the shell, with no control of its own, where the link names none", async () => {
    const urls = mockGateway()
    await renderSessions(<RouterSavingsPage />, "/router-savings")

    await waitFor(() => expect(scopesOf(urls, SAVINGS)).toEqual(["ws-1"]))
    expect(screen.queryByRole("button", { name: /Workspace$/ })).toBeNull()
  })

  it("reads the workspace picked on the page", async () => {
    const user = userEvent.setup()
    const urls = mockGateway()
    await renderSessions(<RouterSavingsPage />, "/router-savings?workspace=all")

    await pickOption(user, "Workspace", "Production")

    await waitFor(() => expect(scopesOf(urls, SAVINGS)).toContain("ws-1"))
    expect(searchOf().get("workspace")).toBe("ws-1")
  })

  it("keeps the workspace on the other views", async () => {
    const user = userEvent.setup()
    const urls = mockGateway()
    await renderSessions(
      <RouterSavingsPage />,
      "/router-savings?workspace=ws-2",
    )

    await user.click(await screen.findByRole("link", { name: "Sessions" }))

    await waitFor(() => expect(scopesOf(urls, SESSION_LIST)).toContain("ws-2"))
    expect(scopesOf(urls, SESSION_LIST)).not.toContain("ws-1")
    expect(searchOf().get("workspace")).toBe("ws-2")
    expect(selectTrigger("Workspace")).toHaveTextContent("Staging")
  })

  it("follows the shell again once a workspace is picked there", async () => {
    const user = userEvent.setup()
    const urls = mockGateway()
    await renderSessions(<RouterSavingsPage />, "/router-savings?workspace=all")
    await waitFor(() => expect(scopesOf(urls, SAVINGS)).toEqual([""]))

    await user.click(
      screen.getByRole("button", { name: "Select Staging in the shell" }),
    )

    await waitFor(() => expect(searchOf().get("workspace")).toBeNull())
    await waitFor(() => expect(scopesOf(urls, SAVINGS)).toContain("ws-2"))
    expect(screen.queryByRole("button", { name: /Workspace$/ })).toBeNull()
  })

  it("follows the shell again when the workspace already selected there is picked", async () => {
    const user = userEvent.setup()
    const urls = mockGateway()
    await renderSessions(
      <RouterSavingsPage />,
      "/router-savings?workspace=ws-2",
    )
    await waitFor(() => expect(scopesOf(urls, SAVINGS)).toContain("ws-2"))

    await user.click(
      screen.getByRole("button", { name: "Select Production in the shell" }),
    )

    await waitFor(() => expect(searchOf().get("workspace")).toBeNull())
    await waitFor(() => expect(scopesOf(urls, SAVINGS)).toContain("ws-1"))
  })

  it("still names the workspace when the list of workspaces fails", async () => {
    mockGateway({ workspacesFail: true })
    await renderSessions(
      <RouterSavingsPage />,
      "/router-savings?workspace=ws-2",
    )

    await waitFor(() =>
      expect(selectTrigger("Workspace")).toHaveTextContent("ws-2"),
    )
  })

  it("closes the open session when another workspace is picked", async () => {
    const user = userEvent.setup()
    mockGateway()
    await renderSessions(
      <SessionsPage embedded />,
      "/sessions?workspace=ws-2&peek=trace-1&peek_workspace=ws-2&span=span-1",
    )

    await pickOption(user, "Workspace", "Production")

    await waitFor(() => expect(searchOf().get("workspace")).toBe("ws-1"))
    expect(searchOf().get("peek")).toBeNull()
    expect(searchOf().get("peek_workspace")).toBeNull()
    expect(searchOf().get("span")).toBeNull()
  })
})

describe("A link from Home into Sessions", () => {
  it("opens the sessions of the workspace Home shows, not the shell's", async () => {
    const user = userEvent.setup()
    const urls = mockGateway()
    await renderSessions(<TeamHome />, "/?workspace=ws-2")

    await user.click(
      await screen.findByRole("link", { name: "See the sessions" }),
    )

    await waitFor(() => expect(searchOf().get("workspace")).toBe("ws-2"))
    await waitFor(() => expect(scopesOf(urls, SESSION_LIST)).toContain("ws-2"))
    expect(scopesOf(urls, SESSION_LIST)).not.toContain("ws-1")
    expect(selectTrigger("Workspace")).toHaveTextContent("Staging")
  })

  it("opens router savings on every workspace when Home shows them all", async () => {
    const user = userEvent.setup()
    const urls = mockGateway()
    await renderSessions(<TeamHome />, "/")

    await user.click(
      await screen.findByRole("link", { name: "See router savings" }),
    )

    await screen.findByRole("grid", { name: "Savings by subagent type" })
    expect(searchOf().get("workspace")).toBe("all")
    expect(selectTrigger("Workspace")).toHaveTextContent("All workspaces")
    expect(scopesOf(urls, SAVINGS)).not.toContain("ws-1")
  })
})
