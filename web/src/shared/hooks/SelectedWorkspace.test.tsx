import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  SelectedWorkspaceProvider,
  useSelectedWorkspace,
} from "@/shared/hooks/SelectedWorkspace"
import { organizationContext } from "@/tests/fixtures"

const ALPHA = "11111111-1111-1111-1111-111111111111"
const BETA = "22222222-2222-2222-2222-222222222222"

function mockContext(
  memberships: { workspace_id: string; name: string; role: string }[],
) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
    Response.json(organizationContext({ workspace_memberships: memberships })),
  )
}

function Probe() {
  const { selected, memberships, select } = useSelectedWorkspace()
  return (
    <div>
      <span data-testid="selected">{selected?.name ?? "none"}</span>
      <span data-testid="count">{memberships.length}</span>
      {memberships.map((m) => (
        <button
          key={m.workspace_id}
          type="button"
          onClick={() => select(m.workspace_id)}
        >
          pick {m.name}
        </button>
      ))}
    </div>
  )
}

function renderProbe() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <Probe />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
  )
}

describe("SelectedWorkspace", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("selects the first membership when nothing is remembered", async () => {
    mockContext([
      { workspace_id: ALPHA, name: "Alpha", role: "member" },
      { workspace_id: BETA, name: "Beta", role: "admin" },
    ])
    renderProbe()

    expect(await screen.findByText("Alpha")).toBeInTheDocument()
  })

  it("remembers the chosen workspace across a remount", async () => {
    mockContext([
      { workspace_id: ALPHA, name: "Alpha", role: "member" },
      { workspace_id: BETA, name: "Beta", role: "admin" },
    ])
    const first = renderProbe()
    await userEvent.click(
      await screen.findByRole("button", { name: "pick Beta" }),
    )
    expect(screen.getByTestId("selected")).toHaveTextContent("Beta")

    first.unmount()
    renderProbe()
    expect(await screen.findByText("Beta")).toBeInTheDocument()
  })

  it("falls back to the first membership when the remembered one is gone", async () => {
    // Removed from that workspace, or a rebuilt deployment minted new ids.
    // Keeping the stored id would leave the switcher pointing at nothing.
    window.localStorage.setItem("otari.dashboard.selectedWorkspace", "vanished")
    mockContext([{ workspace_id: ALPHA, name: "Alpha", role: "member" }])
    renderProbe()

    expect(await screen.findByText("Alpha")).toBeInTheDocument()
    expect(
      window.localStorage.getItem("otari.dashboard.selectedWorkspace"),
    ).toBe(ALPHA)
  })

  it("reports no selection when the caller belongs to no workspace", async () => {
    mockContext([])
    renderProbe()

    expect(await screen.findByTestId("count")).toHaveTextContent("0")
    expect(screen.getByTestId("selected")).toHaveTextContent("none")
  })
})
