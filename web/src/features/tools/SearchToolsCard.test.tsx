import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { SearchProviderInfo, SearchToolsResponse } from "@/client"
import { SearchToolsCard } from "@/features/tools/SearchToolsCard"
import { API_ROOT } from "@/shared/api/client"
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

// The list is a drill-in now, so every assertion about it opens the row first.
async function renderOpened(user: ReturnType<typeof userEvent.setup>) {
  renderWithClient(<SearchToolsCard docsHref="https://docs.example/tools" />)
  await user.click(
    await screen.findByRole("button", { name: /Configure search tools/ }),
  )
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
      if (url.includes(`${API_ROOT}/search-tools/providers`)) {
        return jsonResponse(opts.providers ?? PROVIDERS)
      }
      if (url.includes(`${API_ROOT}/search-tools`)) {
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
    const user = userEvent.setup()
    await renderOpened(user)

    expect(await screen.findByText("local")).toBeInTheDocument()
    expect(screen.getByLabelText("Backend URL for local")).toHaveValue(
      "http://searxng:8080",
    )
    expect(screen.getByText("from-file")).toBeInTheDocument()
    expect(screen.getByText(/config file/)).toBeInTheDocument()
    // A config-file tool has no editable box of its own.
    expect(
      screen.queryByLabelText("Backend URL for from-file"),
    ).not.toBeInTheDocument()
  })

  it("says the endpoint refuses everything when nothing is configured", async () => {
    mockApi({ tools: { stored: [], config: [] } })
    renderWithClient(<SearchToolsCard docsHref="https://docs.example/tools" />)

    expect(await screen.findByText(/refuses every request/)).toBeInTheDocument()
    expect(await screen.findByText("0 tools")).toBeInTheDocument()
  })

  it("does not read a failed request as an empty deployment", async () => {
    // `isLoading` goes false with no data behind it, so the fallback would
    // otherwise claim no tools are configured and that POST /api/v1/search refuses
    // every request, on a read that never answered.
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input)
      if (url.includes(`${API_ROOT}/search-tools/providers`)) {
        return jsonResponse(PROVIDERS)
      }
      if (url.includes(`${API_ROOT}/search-tools`)) {
        return jsonResponse({ detail: "boom" }, 500)
      }
      return jsonResponse([])
    })
    renderWithClient(<SearchToolsCard docsHref="https://docs.example/tools" />)

    expect(
      await screen.findByText(
        "Could not read the tools this deployment serves.",
      ),
    ).toBeInTheDocument()
    expect(screen.queryByText(/refuses every request/)).toBeNull()
    expect(screen.queryByText("0 tools")).toBeNull()
  })

  it("opens the drill-in when a tool is created, so the new row is on screen", async () => {
    // The row lands inside a disclosure that is collapsed by default, and the
    // trigger is on the group heading outside it, so without this the dialog
    // closes and the only thing that changes is the trailing count.
    mockApi()
    const user = userEvent.setup()
    renderWithClient(<SearchToolsCard docsHref="https://docs.example/tools" />)

    await user.click(
      await screen.findByRole("button", { name: "Add search tool" }),
    )
    const dialog = within(await screen.findByRole("dialog"))
    await user.type(dialog.getByLabelText(/^Name/), "second")
    await pickOption(user, "Provider", "searxng")
    await user.type(dialog.getByLabelText(/^Backend URL/), "http://other:8080")
    await user.click(dialog.getByRole("button", { name: "Add search tool" }))

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())
    // The drill-in itself reports open, which is what puts the new row in
    // front of the operator; asserting a row's text would pass on a closed
    // disclosure whose content is still in the DOM.
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Configure search tools/ }),
      ).toHaveAttribute("aria-expanded", "true"),
    )
  })

  it("keeps the heading trigger on screen and opens on a blank draft", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderOpened(user)
    await screen.findByText("local")

    const trigger = screen.getByRole("button", { name: "Add search tool" })
    await user.click(trigger)
    expect(await screen.findByRole("dialog")).toBeInTheDocument()
    expect(trigger).toBeVisible()

    await user.type(
      within(screen.getByRole("dialog")).getByLabelText(/^Name/),
      "second",
    )
    // A draft this far along is dirty, so the way out is through the guard.
    await user.keyboard("{Escape}")
    await user.click(screen.getByRole("button", { name: "Discard" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull())

    await user.click(trigger)
    expect(
      within(await screen.findByRole("dialog")).getByLabelText(/^Name/),
    ).toHaveValue("")
  })

  it("adds a tool with the chosen provider", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    await renderOpened(user)
    await screen.findByText("local")

    await user.click(screen.getByRole("button", { name: "Add search tool" }))
    // Scoped to the dialog: the heading's trigger and the submit say the same
    // words, and every row behind it has a backend-URL box of its own.
    const dialog = within(await screen.findByRole("dialog"))
    await user.type(dialog.getByLabelText(/^Name/), "second")
    await pickOption(user, "Provider", "searxng")
    await user.type(dialog.getByLabelText(/^Backend URL/), "http://other:8080")
    await user.click(dialog.getByRole("button", { name: "Add search tool" }))

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
    await renderOpened(user)
    await screen.findByText("local")

    await user.click(screen.getByRole("button", { name: "Add search tool" }))
    const dialog = within(await screen.findByRole("dialog"))
    await user.type(dialog.getByLabelText(/^Name/), "keyless-exa")
    const submit = dialog.getByRole("button", { name: "Add search tool" })
    expect(submit).toBeDisabled()

    await user.type(dialog.getByLabelText(/^API key/), "exa-live")
    expect(submit).toBeEnabled()
  })

  it("omits api_key from a save that only changes the backend URL", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    await renderOpened(user)
    await screen.findByText("local")

    const input = screen.getByLabelText("Backend URL for local")
    await user.clear(input)
    await user.type(input, "http://moved:8080")
    await user.tab()

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
    await renderOpened(user)
    await screen.findByText("local")

    const input = screen.getByLabelText("Backend URL for local")
    await user.clear(input)
    await user.type(input, "http://moved:8080")
    await user.tab()

    expect(await screen.findByText(/api_key is required/)).toBeInTheDocument()
  })

  it("saves the row on blur, with no Save button of its own", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    await renderOpened(user)
    await screen.findByText("local")

    expect(screen.queryByRole("button", { name: "Save local" })).toBeNull()

    const input = screen.getByLabelText("Backend URL for local")
    await user.clear(input)
    await user.type(input, "http://moved:8080{Enter}")

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([, init]) => (init?.method ?? "") === "PATCH",
        ),
      ).toBe(true),
    )
  })

  it("leaves the stored key alone when the key field is left empty", async () => {
    // Blank is not a value in that field, it is "keep what is stored", so a
    // focus and a blur must not write anything at all.
    const fetchMock = mockApi()
    const user = userEvent.setup()
    await renderOpened(user)
    await screen.findByText("local")

    await user.click(screen.getByLabelText("New API key for local"))
    await user.tab()

    expect(
      fetchMock.mock.calls.some(([, init]) => (init?.method ?? "") === "PATCH"),
    ).toBe(false)
  })

  it("carries the fresh expected_updated_at into a second write on one row", async () => {
    // Under a Save button this was one write per click. Under autosave, leaving
    // the URL and then the key fires two, and the second would otherwise still
    // carry the `updated_at` captured before the first, which optimistic
    // concurrency refuses.
    const bodies: Record<string, unknown>[] = []
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      if (url.includes(`${API_ROOT}/search-tools/providers`)) {
        return jsonResponse(PROVIDERS)
      }
      if (url.includes(`${API_ROOT}/search-tools`) && method === "PATCH") {
        const body = JSON.parse(String(init?.body)) as Record<string, unknown>
        bodies.push(body)
        return jsonResponse({
          ...TOOLS.stored[0],
          ...body,
          updated_at: `2026-08-14T00:00:0${bodies.length}+00:00`,
        })
      }
      return jsonResponse(TOOLS)
    })
    const user = userEvent.setup()
    await renderOpened(user)
    await screen.findByText("local")

    const url = screen.getByLabelText("Backend URL for local")
    await user.clear(url)
    await user.type(url, "http://moved:8080")
    await user.tab()
    await waitFor(() => expect(bodies).toHaveLength(1))

    await user.type(screen.getByLabelText("New API key for local"), "sk-second")
    await user.tab()

    await waitFor(() => expect(bodies).toHaveLength(2))
    expect(bodies[0].expected_updated_at).toBe("2026-08-14T00:00:00+00:00")
    expect(bodies[1].expected_updated_at).toBe("2026-08-14T00:00:01+00:00")
    expect(bodies[1].api_key).toBe("sk-second")
  })

  it("removes a tool only through the confirm dialog", async () => {
    // otari-ai#2110. The trigger names the object and the dialog names the
    // consequence, which the armed button had no room to say.
    const fetchMock = mockApi()
    const user = userEvent.setup()
    await renderOpened(user)
    await screen.findByText("local")

    await user.click(screen.getByRole("button", { name: "Remove local" }))
    const dialog = await screen.findByRole("alertdialog")
    expect(
      within(dialog).getByText(/local and the key stored with it/),
    ).toBeVisible()
    expect(
      fetchMock.mock.calls.some(
        ([, init]) => (init?.method ?? "") === "DELETE",
      ),
    ).toBe(false)

    await user.click(
      within(dialog).getByRole("button", { name: "Remove permanently" }),
    )

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(
        ([, init]) => (init?.method ?? "") === "DELETE",
      )
      expect(String(call?.[0])).toContain(`${API_ROOT}/search-tools/local`)
    })
  })
})
