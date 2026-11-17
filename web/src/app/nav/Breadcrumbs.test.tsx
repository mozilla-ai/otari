import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { Breadcrumbs } from "@/app/nav/Breadcrumbs"
import {
  BreadcrumbLeafProvider,
  useBreadcrumbLeaf,
} from "@/shared/hooks/BreadcrumbLeaf"
import { withRouter } from "@/tests/router"

vi.mock("@/shared/api/organizations", () => ({
  useOrganizationContext: () => ({ data: undefined }),
}))
vi.mock("@/shared/hooks/SelectedWorkspace", () => ({
  useSelectedWorkspace: () => ({
    selected: { workspace_id: "w", name: "Default workspace" },
  }),
}))

function ServerPage({ name }: { name: string }) {
  useBreadcrumbLeaf(name)
  return null
}

function renderTrail(pathname: string, leaf: string) {
  return render(
    <BreadcrumbLeafProvider>
      <Breadcrumbs pathname={pathname} />
      {leaf ? <ServerPage name={leaf} /> : null}
    </BreadcrumbLeafProvider>,
    { wrapper: withRouter({ url: pathname }) },
  )
}

describe("Breadcrumbs", () => {
  it("ends a record's page on its name, linking back to the list", async () => {
    renderTrail("/tools/mcp-servers/abc", "notion")

    const trail = await screen.findByRole("navigation", { name: "Breadcrumb" })
    expect(trail).toHaveTextContent("Default workspace/MCP servers/notion")
    expect(screen.getByRole("link", { name: "MCP servers" })).toHaveAttribute(
      "href",
      "/tools/mcp-servers",
    )
  })

  it("names a list page without a link to itself", async () => {
    renderTrail("/tools/mcp-servers", "")

    const trail = await screen.findByRole("navigation", { name: "Breadcrumb" })
    expect(trail).toHaveTextContent("Default workspace/MCP servers")
    expect(screen.queryByRole("link")).not.toBeInTheDocument()
  })
})
