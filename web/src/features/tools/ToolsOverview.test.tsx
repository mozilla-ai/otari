import { Outlet } from "@tanstack/react-router"
import { screen, within } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { organizationContext } from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"
import { ToolsOverview } from "./ToolsOverview"

type Read = "tools" | "web" | "mcp"

const TOOLS = ["otari_web_search", "otari_web_fetch", "otari_code_execution"]

/** An overview of Platform, an owner's by default, with whichever reads refused. */
async function renderOverview(refused: Read[] = [], role = "owner") {
  const refuse = () => Response.json({ detail: "Forbidden" }, { status: 403 })
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url === `${API_ROOT}/organizations/me`) {
      return Response.json(
        organizationContext({
          role,
          deployment_operator: false,
          workspace_memberships: [
            { workspace_id: "ws-platform", name: "Platform", role: "admin" },
          ],
        }),
      )
    }
    if (url === `${API_ROOT}/tools`) {
      if (refused.includes("tools")) return refuse()
      return Response.json({
        data: TOOLS.map((id) => ({ id, available: true })),
      })
    }
    if (url.endsWith("/web-search")) {
      return refused.includes("web")
        ? refuse()
        : Response.json({ enabled: false })
    }
    // What Platform's requests may declare, which any member may read.
    if (url.includes("/playground/tools")) {
      const on = { configured: true, enabled: true, reason: null }
      return Response.json({
        web_search: {
          configured: true,
          enabled: false,
          reason: "Turned off for this workspace.",
        },
        web_fetch: on,
        code_execution: on,
        files: on,
        mcp_servers: [],
      })
    }
    if (url.endsWith("/code-execution-policy")) {
      return Response.json({ enabled: false })
    }
    if (url.includes("/mcp-servers")) {
      if (refused.includes("mcp")) return refuse()
      return Response.json({ data: [], count: 0 })
    }
    return Response.json({ data: [], count: 0 })
  })
  await renderWithRouter(<ToolsOverview />, {
    shell: (
      <Provider>
        <SelectedWorkspaceProvider>
          <Outlet />
        </SelectedWorkspaceProvider>
      </Provider>
    ),
  })
}

async function row(name: string) {
  return screen.findByRole("link", { name: new RegExp(`^${name}`) })
}

describe("the Tools overview", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("says what is off and what is connected once it has read them", async () => {
    await renderOverview()
    expect(
      await within(await row("Web search")).findByText("Off"),
    ).toBeVisible()
    expect(
      await within(await row("MCP servers")).findByText(
        "No server is connected to Platform. Connect one to give agents its tools.",
      ),
    ).toBeVisible()
  })

  it("tells a member which tools their workspace has off", async () => {
    await renderOverview([], "member")
    const web = await row("Web search")
    expect(await within(web).findByText("Off")).toBeVisible()
    expect(
      within(web).getByText(/^Turned off for this workspace\./),
    ).toBeVisible()
    expect(
      await within(await row("Web fetch")).findByText("Available"),
    ).toBeVisible()
  })

  it("does not call a tool off when its switch could not be read", async () => {
    await renderOverview(["web"])
    const web = await row("Web search")
    expect(await within(web).findByText("Could not read")).toBeVisible()
    expect(within(web).queryByText("Off")).toBeNull()
    expect(screen.getByRole("alert")).toBeVisible()
  })

  it("does not call a tool unavailable when the deployment's tools could not be read", async () => {
    await renderOverview(["tools"])
    const code = await row("Code execution")
    expect(await within(code).findByText("Could not read")).toBeVisible()
    expect(within(code).queryByText("Unavailable")).toBeNull()
  })

  it("does not say no server is connected when the servers could not be read", async () => {
    await renderOverview(["mcp"])
    const mcp = await row("MCP servers")
    expect(await within(mcp).findByText("Could not read")).toBeVisible()
    expect(within(mcp).queryByText(/No server is connected/)).toBeNull()
  })

  it("still says what MCP servers do when they could not be read", async () => {
    await renderOverview(["mcp"])
    const mcp = await row("MCP servers")
    expect(await within(mcp).findByText("Could not read")).toBeVisible()
    expect(
      within(mcp).getByText(
        "Lets an agent call the tools a connected server offers.",
      ),
    ).toBeVisible()
  })
})
