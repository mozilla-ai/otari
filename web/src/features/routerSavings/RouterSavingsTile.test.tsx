import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useState } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { SubagentSavings } from "@/client"
import { RouterSavingsTile } from "@/features/routerSavings/RouterSavingsTile"
import { usageRange, windowOf } from "@/features/spend/period"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  SURFACES,
  savingsAmounts,
  subagentSavings,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const WINDOW = windowOf("7d", Date.parse("2026-11-24T12:00:00Z"))

function mockSavings(savings: SubagentSavings) {
  const urls: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    urls.push(String(input))
    return new Response(JSON.stringify(savings), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
  return urls
}

function renderTile(
  props: { workspaceId?: string; userId?: string } = {},
  surfaces: string[] = SURFACES,
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <DeploymentProvider value={bootstrap({ surfaces })}>
      <QueryClientProvider client={client}>
        <RouterSavingsTile
          window={WINDOW}
          workspaceId={props.workspaceId ?? ""}
          userId={props.userId}
        />
        <span>rendered</span>
      </QueryClientProvider>
    </DeploymentProvider>,
    { url: "/" },
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("RouterSavingsTile", () => {
  it("says what the router saved and links to the page", async () => {
    const urls = mockSavings(subagentSavings())
    await renderTile({ workspaceId: "ws-1", userId: "me" })

    expect(
      await screen.findByText(
        /The subagent router saved \$5\.50 .*\(estimated\)\./,
      ),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "See router savings" }),
    ).toHaveAttribute("href", "/router-savings?period=7d&who=me")
    expect(urls.at(-1)).toContain("workspace_id=ws-1")
    expect(urls.at(-1)).toContain("user_id=me")
  })

  it("links to everyone's savings where it counts everyone's", async () => {
    mockSavings(subagentSavings())
    await renderTile()

    expect(
      await screen.findByRole("link", { name: "See router savings" }),
    ).toHaveAttribute("href", "/router-savings?period=7d")
  })

  it("says a loss as a loss", async () => {
    mockSavings(subagentSavings({ totals: { net_saved: -0.3 } }))
    await renderTile()

    expect(
      await screen.findByText(/cost \$0\.30 more than it saved/),
    ).toBeInTheDocument()
  })

  it("does not call a measured saving an estimate", async () => {
    mockSavings(subagentSavings({ totals: { estimated: 0 } }))
    await renderTile()

    const line = await screen.findByText(/The subagent router saved/)
    expect(line).not.toHaveTextContent("estimated")
  })

  it("draws nothing until a subagent has a priced outcome", async () => {
    const urls = mockSavings(
      subagentSavings({
        totals: {
          ...savingsAmounts({
            decisions: 2,
            router_cost: 0.02,
            net_saved: -0.02,
          }),
          no_outcome: 2,
          unpriced: 0,
        },
      }),
    )
    await renderTile()

    await waitFor(() => expect(urls.length).toBe(1))
    await screen.findByText("rendered")
    expect(screen.queryByText(/subagent router/)).not.toBeInTheDocument()
  })

  it("does not show one window's saving under another while the new one loads", async () => {
    let answerMonth: (response: Response) => void = () => {}
    const month = new Promise<Response>((resolve) => {
      answerMonth = resolve
    })
    const json = (savings: SubagentSavings) =>
      new Response(JSON.stringify(savings), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    const wider = windowOf("30d", Date.parse("2026-11-24T12:00:00Z"))
    const widerStart = new URLSearchParams({
      start_date: usageRange(wider).start_date,
    }).toString()
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
      String(input).includes(widerStart) ? month : json(subagentSavings()),
    )
    function Switching() {
      const [window, setWindow] = useState(WINDOW)
      return (
        <>
          <RouterSavingsTile window={window} workspaceId="" />
          <button type="button" onClick={() => setWindow(wider)}>
            Widen
          </button>
        </>
      )
    }
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    await renderWithRouter(
      <DeploymentProvider value={bootstrap()}>
        <QueryClientProvider client={client}>
          <Switching />
        </QueryClientProvider>
      </DeploymentProvider>,
      { url: "/" },
    )
    await screen.findByText(/saved \$5\.50/)

    await userEvent.click(screen.getByRole("button", { name: "Widen" }))

    await waitFor(() =>
      expect(screen.queryByText(/saved \$5\.50/)).not.toBeInTheDocument(),
    )
    answerMonth(json(subagentSavings({ totals: { net_saved: 9 } })))
    expect(await screen.findByText(/saved \$9\.00/)).toBeInTheDocument()
  })

  it("asks for nothing where the deployment does not serve the page", async () => {
    const urls = mockSavings(subagentSavings())
    await renderTile(
      {},
      SURFACES.filter((surface) => surface !== "subagent_savings"),
    )

    await screen.findByText("rendered")
    expect(urls).toEqual([])
  })
})
