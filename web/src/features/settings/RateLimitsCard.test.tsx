import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { RateLimitRule } from "@/client"
import { AuthProvider } from "@/features/auth/AuthContext"
import { RateLimitsCard } from "@/features/settings/RateLimitsCard"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

const FROM_FILE: RateLimitRule = {
  name: "everyone",
  per: "deployment",
  rpm: null,
  tpm: null,
  max_concurrent: 200,
  lease_sec: 900,
  tpm_admission: "estimate",
  source: "config",
}

const STORED: RateLimitRule = {
  name: "keys",
  per: "key",
  rpm: 600,
  tpm: 200000,
  max_concurrent: null,
  lease_sec: 900,
  tpm_admission: "estimate",
  source: "dashboard",
  updated_at: "2026-10-03T12:00:00Z",
}

/** Serves the rules and records every write. */
function mockApi(rules: RateLimitRule[]) {
  const writes: { method: string; url: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const method = (init?.method ?? "GET").toUpperCase()
    const url = String(input)
    if (method === "GET") return jsonResponse({ rules })
    writes.push({
      method,
      url,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    })
    if (method === "DELETE") return new Response(null, { status: 204 })
    return jsonResponse(STORED, method === "POST" ? 201 : 200)
  })
  return writes
}

function renderCard() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <AuthProvider>
        <RateLimitsCard />
      </AuthProvider>
    </QueryClientProvider>,
  )
}

describe("RateLimitsCard", () => {
  beforeEach(() => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1")
  })

  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("lists every rule, with config.yml rules read-only", async () => {
    mockApi([FROM_FILE, STORED])
    renderCard()

    expect(
      await screen.findByText("600 requests/min · 200,000 tokens/min"),
    ).toBeInTheDocument()
    expect(screen.getByText("200 in flight")).toBeInTheDocument()
    expect(screen.getByText("config.yml")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Edit keys" }),
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Edit everyone" }),
    ).not.toBeInTheDocument()
  })

  it("lists a per-model rule with the models it limits", async () => {
    mockApi([
      {
        name: "flash-cap",
        per: "model",
        models: ["vertex:gemini-2.5-flash"],
        rpm: 100,
        tpm: null,
        max_concurrent: null,
        lease_sec: 900,
        tpm_admission: "estimate",
        source: "config",
      },
    ])
    renderCard()

    expect(
      await screen.findByText("vertex:gemini-2.5-flash · 100 requests/min"),
    ).toBeInTheDocument()
    expect(screen.getByText("per model")).toBeInTheDocument()
  })

  it("says what limits apply when there are no rules", async () => {
    mockApi([])
    renderCard()

    expect(
      await screen.findByText(
        "No rules, so requests are limited only by rate_limit_rpm.",
      ),
    ).toBeInTheDocument()
  })

  it("adds a rule once it has a name and a limit", async () => {
    const writes = mockApi([])
    const user = userEvent.setup()
    renderCard()

    await user.click(await screen.findByRole("button", { name: "Add rule" }))
    const dialog = await screen.findByRole("dialog")
    const submit = within(dialog).getByRole("button", { name: "Add rule" })
    await user.type(within(dialog).getByLabelText(/^Name/), "keys")
    expect(submit).toBeDisabled()

    await user.type(within(dialog).getByLabelText("Requests per minute"), "60")
    await user.click(submit)

    await waitFor(() =>
      expect(writes).toEqual([
        {
          method: "POST",
          url: expect.stringContaining("/rate-limits"),
          body: {
            name: "keys",
            per: "key",
            models: null,
            rpm: 60,
            tpm: null,
            max_concurrent: null,
            lease_sec: 900,
            tpm_admission: "estimate",
          },
        },
      ]),
    )
  })

  it("adds a per-model rule naming the models it limits", async () => {
    const writes = mockApi([])
    const user = userEvent.setup()
    renderCard()

    await user.click(await screen.findByRole("button", { name: "Add rule" }))
    const dialog = await screen.findByRole("dialog")
    await user.type(within(dialog).getByLabelText(/^Name/), "flash-cap")
    await user.click(
      within(dialog).getByRole("button", { name: /Counted for/ }),
    )
    await user.click(await screen.findByRole("option", { name: "Each model" }))
    await user.type(
      within(dialog).getByRole("combobox", { name: /Model 1/ }),
      "vertex:gemini-2.5-flash",
    )
    await user.keyboard("{Escape}")
    await user.type(within(dialog).getByLabelText("Requests per minute"), "100")
    await user.click(within(dialog).getByRole("button", { name: "Add rule" }))

    await waitFor(() =>
      expect(writes[0]?.body).toEqual({
        name: "flash-cap",
        per: "model",
        models: ["vertex:gemini-2.5-flash"],
        rpm: 100,
        tpm: null,
        max_concurrent: null,
        lease_sec: 900,
        tpm_admission: "estimate",
      }),
    )
  })

  it("asks for each model as instance:model, and counts both spellings as one", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderCard()

    await user.click(await screen.findByRole("button", { name: "Add rule" }))
    const dialog = await screen.findByRole("dialog")
    await user.type(within(dialog).getByLabelText(/^Name/), "flash-cap")
    await user.click(
      within(dialog).getByRole("button", { name: /Counted for/ }),
    )
    await user.click(await screen.findByRole("option", { name: "Each model" }))
    await user.type(within(dialog).getByLabelText("Requests per minute"), "100")
    await user.type(
      within(dialog).getByRole("combobox", { name: /Model 1/ }),
      "gpt-4o",
    )
    await user.keyboard("{Escape}")

    expect(
      within(dialog).getByText("Write each model as instance:model."),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "Add rule" }),
    ).toBeDisabled()

    await user.clear(within(dialog).getByRole("combobox", { name: /Model 1/ }))
    await user.type(
      within(dialog).getByRole("combobox", { name: /Model 1/ }),
      "openai/gpt-4o",
    )
    await user.keyboard("{Escape}")
    await user.click(
      within(dialog).getByRole("button", { name: "Add a model" }),
    )
    await user.type(
      within(dialog).getByRole("combobox", { name: /Model 2/ }),
      "openai:gpt-4o",
    )
    await user.keyboard("{Escape}")

    expect(
      within(dialog).getByText("Name each model once."),
    ).toBeInTheDocument()
  })

  it("refuses a limit that is not a whole number", async () => {
    mockApi([])
    const user = userEvent.setup()
    renderCard()

    await user.click(await screen.findByRole("button", { name: "Add rule" }))
    const dialog = await screen.findByRole("dialog")
    await user.type(within(dialog).getByLabelText(/^Name/), "keys")
    await user.type(within(dialog).getByLabelText("Requests per minute"), "1.5")

    expect(
      within(dialog).getByText("A whole number above zero."),
    ).toBeInTheDocument()
    expect(
      within(dialog).getByRole("button", { name: "Add rule" }),
    ).toBeDisabled()
  })

  it("clears a limit on edit by sending null", async () => {
    const writes = mockApi([STORED])
    const user = userEvent.setup()
    renderCard()

    await user.click(await screen.findByRole("button", { name: "Edit keys" }))
    const dialog = await screen.findByRole("dialog")
    await user.clear(within(dialog).getByLabelText("Tokens per minute"))
    await user.click(within(dialog).getByRole("button", { name: "Save rule" }))

    await waitFor(() =>
      expect(writes).toEqual([
        {
          method: "PATCH",
          url: expect.stringContaining("/rate-limits/keys"),
          body: {
            per: "key",
            models: null,
            rpm: 600,
            tpm: null,
            max_concurrent: null,
            lease_sec: 900,
            tpm_admission: "estimate",
          },
        },
      ]),
    )
  })

  it("removes a stored rule after confirming", async () => {
    const writes = mockApi([STORED])
    const user = userEvent.setup()
    renderCard()

    await user.click(await screen.findByRole("button", { name: "Remove keys" }))
    await user.click(await screen.findByRole("button", { name: "Remove rule" }))

    await waitFor(() =>
      expect(writes).toEqual([
        {
          method: "DELETE",
          url: expect.stringContaining("/rate-limits/keys"),
          body: undefined,
        },
      ]),
    )
  })
})
