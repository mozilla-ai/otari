import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { SearchProviderInfo, SearchToolsResponse } from "@/client"
import { SearchToolsCard } from "@/features/tools/SearchToolsCard"
import { pickOption } from "@/tests/select"

const PROVIDERS: SearchProviderInfo[] = [
  {
    id: "exa",
    requires_api_key: true,
    requires_api_base: false,
    default_api_base: "https://api.exa.ai",
  },
  {
    id: "searxng",
    requires_api_key: false,
    requires_api_base: true,
    default_api_base: "http://searxng:8080",
  },
]

const TOOLS: SearchToolsResponse = {
  stored: [
    {
      name: "local",
      provider: "searxng",
      api_base: "http://searxng:8080",
      last4: null,
      timeout: null,
      options: {},
      created_at: null,
      updated_at: "2026-08-14T00:00:00+00:00",
      decryptable: true,
      shadows_config: false,
    },
  ],
  config: [
    {
      name: "from-file",
      provider: "exa",
      api_base: null,
      has_api_key: true,
      shadowed: false,
    },
  ],
}

function renderWithClient(ui: ReactElement) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>)
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

interface MockOpts {
  tools?: SearchToolsResponse
  providers?: SearchProviderInfo[]
  writeStatus?: number
  writeDetail?: string
}

function mockApi(opts: MockOpts = {}) {
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      if (url.includes("/v1/search-tools/providers")) {
        return jsonResponse(opts.providers ?? PROVIDERS)
      }
      if (url.includes("/v1/search-tools")) {
        if (method !== "GET") {
          if (opts.writeStatus && opts.writeStatus >= 400) {
            return jsonResponse(
              { detail: opts.writeDetail ?? "bad" },
              opts.writeStatus,
            )
          }
          return jsonResponse(
            { name: "new", provider: "searxng" },
            method === "POST" ? 201 : 200,
          )
        }
        return jsonResponse(opts.tools ?? TOOLS)
      }
      return jsonResponse([])
    })
}

describe("SearchToolsCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("lists stored tools as editable and config-file tools as read-only", async () => {
    mockApi()
    renderWithClient(<SearchToolsCard onSaved={() => {}} />)

    expect(await screen.findByText("local")).toBeInTheDocument()
    expect(screen.getByLabelText("Backend URL for local")).toHaveValue(
      "http://searxng:8080",
    )
    expect(screen.getByText("from-file")).toBeInTheDocument()
    expect(screen.getByText("Config file")).toBeInTheDocument()
    // A config-file tool has no editable box of its own.
    expect(
      screen.queryByLabelText("Backend URL for from-file"),
    ).not.toBeInTheDocument()
  })

  it("says the endpoint refuses everything when nothing is configured", async () => {
    mockApi({ tools: { stored: [], config: [] } })
    renderWithClient(<SearchToolsCard onSaved={() => {}} />)

    expect(await screen.findByText(/refuses every request/)).toBeInTheDocument()
  })

  it("adds a tool with the chosen provider", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<SearchToolsCard onSaved={() => {}} />)
    await screen.findByText("local")

    await user.type(screen.getByLabelText("Search tool name"), "second")
    await pickOption(user, "Search provider", "searxng")
    await user.type(
      screen.getByLabelText("Search backend URL"),
      "http://other:8080",
    )
    await user.click(screen.getByRole("button", { name: "Add" }))

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([, init]) => (init?.method ?? "") === "POST",
      )
      expect(call).toBeDefined()
      expect(JSON.parse(String(call?.[1]?.body))).toEqual({
        name: "second",
        provider: "searxng",
        api_base: "http://other:8080",
        api_key: null,
      })
    })
  })

  it("will not submit an exa tool without the key exa requires", async () => {
    mockApi()
    const user = userEvent.setup()
    renderWithClient(<SearchToolsCard onSaved={() => {}} />)
    await screen.findByText("local")

    await user.type(screen.getByLabelText("Search tool name"), "keyless-exa")
    expect(screen.getByRole("button", { name: "Add" })).toBeDisabled()

    await user.type(screen.getByLabelText("Search API key"), "exa-live")
    expect(screen.getByRole("button", { name: "Add" })).toBeEnabled()
  })

  it("omits api_key from a save that only changes the backend URL", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<SearchToolsCard onSaved={() => {}} />)
    await screen.findByText("local")

    const input = screen.getByLabelText("Backend URL for local")
    await user.clear(input)
    await user.type(input, "http://moved:8080")
    await user.click(screen.getByRole("button", { name: "Save local" }))

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([, init]) => (init?.method ?? "") === "PATCH",
      )
      expect(call).toBeDefined()
      const body = JSON.parse(String(call?.[1]?.body)) as Record<
        string,
        unknown
      >
      expect(body).not.toHaveProperty("api_key")
      expect(body.api_base).toBe("http://moved:8080")
      expect(body.expected_updated_at).toBe("2026-08-14T00:00:00+00:00")
    })
  })

  it("surfaces a rejected write next to the tool", async () => {
    mockApi({
      writeStatus: 422,
      writeDetail: "search_tools.local.api_key is required for provider 'exa'.",
    })
    const user = userEvent.setup()
    renderWithClient(<SearchToolsCard onSaved={() => {}} />)
    await screen.findByText("local")

    const input = screen.getByLabelText("Backend URL for local")
    await user.clear(input)
    await user.type(input, "http://moved:8080")
    await user.click(screen.getByRole("button", { name: "Save local" }))

    expect(await screen.findByText(/api_key is required/)).toBeInTheDocument()
  })

  it("removes a tool after the confirm step", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    renderWithClient(<SearchToolsCard onSaved={() => {}} />)
    await screen.findByText("local")

    // The two steps read differently now: the trigger names the object and the
    // armed confirm names the consequence, which is what the second click does.
    await user.click(screen.getByRole("button", { name: "Remove tool" }))
    await user.click(screen.getByRole("button", { name: "Remove permanently" }))

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([, init]) => (init?.method ?? "") === "DELETE",
      )
      expect(String(call?.[0])).toContain("/v1/search-tools/local")
    })
  })
})
