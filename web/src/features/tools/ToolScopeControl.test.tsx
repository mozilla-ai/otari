import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ToolScopeControl } from "@/features/tools/ToolScopeControl"
import { API_ROOT } from "@/shared/api/client"
import { organizationContext } from "@/tests/fixtures"

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

// GET /search-tools is operator-only, so `operator` decides whether the list
// of tools is read at all.
function mockApi({ operator = false } = {}) {
  const urls: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input)
    urls.push(url)
    if (url.endsWith(`${API_ROOT}/organizations/me`)) {
      return json(organizationContext({ deployment_operator: operator }))
    }
    if (operator && url.includes(`${API_ROOT}/search-tools`)) {
      return json({
        stored: [{ name: "news", kind: "search", provider: "exa" }],
        config: [
          { name: "deepseek", kind: "search", provider: "deepseek" },
          { name: "news", kind: "search", provider: "exa", shadowed: true },
        ],
      })
    }
    return json({ detail: "Forbidden" }, 403)
  })
  return urls
}

function renderControl(initial: string[] | undefined) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  const onChange = vi.fn()
  render(
    <QueryClientProvider client={client}>
      <ToolScopeControl
        initial={initial}
        onChange={onChange}
        description="Which search tools this key may run."
        inheritLabel="Inherit owner access"
      />
    </QueryClientProvider>,
  )
  return onChange
}

describe("ToolScopeControl", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("offers an operator every tool by name, once each, and the star", async () => {
    mockApi({ operator: true })
    const user = userEvent.setup()
    const onChange = renderControl(["news"])

    // Every option has an "e" in it, so the filter keeps them all.
    await user.type(screen.getByLabelText("Add a search tool"), "e")
    expect(
      await screen.findByRole("option", { name: "deepseek" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("option", { name: "* · every search tool" }),
    ).toBeInTheDocument()
    // Already on the list, and listed once though stored and config both name it.
    expect(
      screen.queryByRole("option", { name: "news" }),
    ).not.toBeInTheDocument()

    await user.click(screen.getByRole("option", { name: "deepseek" }))
    expect(onChange).toHaveBeenLastCalledWith(["news", "deepseek"], true)
  })

  it("offers anyone else every tool or none, without reading the list", async () => {
    const urls = mockApi()
    const user = userEvent.setup()
    const onChange = renderControl(["news"])

    expect(
      await screen.findByText(/Only the deployment operator can list/),
    ).toBeInTheDocument()
    await user.type(screen.getByLabelText("Add a search tool"), "e")
    await user.click(
      await screen.findByRole("option", { name: "* · every search tool" }),
    )
    expect(onChange).toHaveBeenLastCalledWith(["news", "*"], true)
    expect(urls.some((url) => url.includes(`${API_ROOT}/search-tools`))).toBe(
      false,
    )
  })

  it("tells inheriting from blocking, and an empty selection is incomplete", async () => {
    mockApi({ operator: true })
    const user = userEvent.setup()
    const onChange = renderControl(undefined)

    await user.click(screen.getByRole("button", { name: "No tool" }))
    expect(onChange).toHaveBeenLastCalledWith([], true)
    await user.click(screen.getByRole("button", { name: "Selected tools" }))
    expect(onChange).toHaveBeenLastCalledWith([], false)
    await user.click(
      screen.getByRole("button", { name: "Inherit owner access" }),
    )
    expect(onChange).toHaveBeenLastCalledWith(undefined, true)
  })

  it("says blocking covers /v1/search, where the list is checked", async () => {
    mockApi({ operator: true })
    renderControl([])

    expect(
      screen.getByText(/search tool on \/v1\/search until you change/),
    ).toBeInTheDocument()
  })

  it("says the tools are loading rather than offering the star alone", async () => {
    let release: (response: Response) => void = () => {}
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return json(organizationContext({ deployment_operator: true }))
      }
      return new Promise<Response>((resolve) => {
        release = resolve
      })
    })
    const user = userEvent.setup()
    renderControl(["deepseek"])

    await user.type(await screen.findByLabelText("Add a search tool"), "*")
    expect(
      await screen.findByText("Loading the search tools…"),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("option", { name: "* · every search tool" }),
    ).not.toBeInTheDocument()
    release(json({ stored: [], config: [] }))
    expect(
      await screen.findByRole("option", { name: "* · every search tool" }),
    ).toBeInTheDocument()
  })
})
