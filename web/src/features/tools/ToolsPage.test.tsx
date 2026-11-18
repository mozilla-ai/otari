import { screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { organizationContext } from "@/tests/fixtures"
import { AppProviders as Provider } from "@/tests/providers"
import { renderWithRouter } from "@/tests/router"
import { ToolsPage } from "./ToolsPage"

function mockApi(operator: boolean) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    if (url === `${API_ROOT}/organizations/me`) {
      return Response.json(
        organizationContext({
          deployment_operator: operator,
          role: operator ? "owner" : "member",
        }),
      )
    }
    if (url.includes("/tool-settings")) return Response.json({ fields: [] })
    if (url === `${API_ROOT}/tools`) return Response.json({ data: [] })
    return Response.json({ data: [], count: 0 })
  })
}

async function renderPage(url: string) {
  await renderWithRouter(
    <Provider>
      <SelectedWorkspaceProvider>
        <ToolsPage />
      </SelectedWorkspaceProvider>
    </Provider>,
    { url },
  )
}

const OVERVIEW_ROW = "Lets an agent call the tools a connected server offers."
const GUARDRAILS_LINK = { name: "Guardrails" }

describe("ToolsPage", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows the overview under the Tools tabs by default", async () => {
    mockApi(true)
    await renderPage("/tools")

    expect(
      await screen.findByRole("heading", { name: "Tools" }),
    ).toBeInTheDocument()
    expect(await screen.findByText(OVERVIEW_ROW)).toBeInTheDocument()
    expect(screen.queryByRole("link", GUARDRAILS_LINK)).not.toBeInTheDocument()
  })

  it("gives an operator with ?service=guardrails the guardrail services view", async () => {
    mockApi(true)
    await renderPage("/tools?service=guardrails")

    expect(await screen.findByRole("link", GUARDRAILS_LINK)).toHaveAttribute(
      "href",
      "/organization/guardrails",
    )
  })

  it("gives a non-operator with ?service=guardrails the overview", async () => {
    mockApi(false)
    await renderPage("/tools?service=guardrails")

    expect(await screen.findByText(OVERVIEW_ROW)).toBeInTheDocument()
    expect(screen.queryByRole("link", GUARDRAILS_LINK)).not.toBeInTheDocument()
  })
})
