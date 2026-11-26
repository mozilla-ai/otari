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
    // An MCP server opens from Tools, which is the list it links back to.
    renderTrail("/tools/mcp-servers/abc", "notion")

    const trail = await screen.findByRole("navigation", { name: "Breadcrumb" })
    expect(trail).toHaveTextContent("Default workspace/Tools/notion")
    expect(screen.getByRole("link", { name: "Tools" })).toHaveAttribute(
      "href",
      "/tools",
    )
  })

  it("names a list page without a link to itself", async () => {
    renderTrail("/tools", "")

    const trail = await screen.findByRole("navigation", { name: "Breadcrumb" })
    expect(trail).toHaveTextContent("Default workspace/Tools")
    expect(screen.queryByRole("link")).not.toBeInTheDocument()
  })

  it("puts a page with no row under the row that stands for it", async () => {
    renderTrail("/routing", "")

    const trail = await screen.findByRole("navigation", { name: "Breadcrumb" })
    expect(trail).toHaveTextContent("Default workspace/Rules/Routing")
    expect(screen.queryByRole("link")).not.toBeInTheDocument()
  })

  it("files the workspace roster under People", async () => {
    renderTrail("/members", "")

    const trail = await screen.findByRole("navigation", { name: "Breadcrumb" })
    expect(trail).toHaveTextContent("Default workspace/People/Members")
  })

  it("ends a policy's page on its name, after Rules and a link to Routing", async () => {
    renderTrail("/routing/coding", "coding")

    const trail = await screen.findByRole("navigation", { name: "Breadcrumb" })
    expect(trail).toHaveTextContent("Default workspace/Rules/Routing/coding")
    expect(screen.getByRole("link", { name: "Routing" })).toHaveAttribute(
      "href",
      "/routing",
    )
  })
})
