import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ReactElement } from "react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { BuiltInGuardrailSpec, StoredGuardrail } from "@/client"
import { LocalGuardrailsCard } from "@/features/tools/LocalGuardrailsCard"
import { API_ROOT } from "@/shared/api/client"
import {
  builtInGuardrail,
  configGuardrail,
  organizationContext,
  storedGuardrail,
} from "@/tests/fixtures"

const CATALOG = [builtInGuardrail()]
const STORED = storedGuardrail()
const CONFIG = configGuardrail()

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
  catalog?: BuiltInGuardrailSpec[]
  stored?: StoredGuardrail[]
  listStatus?: number
  writeStatus?: number
  writeDetail?: string
}

function mockApi(opts: MockOpts = {}) {
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = (init?.method ?? "GET").toUpperCase()
      if (url.includes(`${API_ROOT}/tool-settings/guardrails/catalog`)) {
        return jsonResponse({ guardrails: opts.catalog ?? CATALOG })
      }
      if (url.includes(`${API_ROOT}/guardrail-credentials`)) {
        if (method !== "GET") {
          if (opts.writeStatus && opts.writeStatus >= 400) {
            return jsonResponse(
              { detail: opts.writeDetail ?? "bad" },
              opts.writeStatus,
            )
          }
          return method === "DELETE"
            ? new Response(null, { status: 204 })
            : jsonResponse(STORED, method === "POST" ? 201 : 200)
        }
        if (opts.listStatus && opts.listStatus >= 400) {
          return jsonResponse({ detail: "nope" }, opts.listStatus)
        }
        return jsonResponse({
          stored: opts.stored ?? [STORED],
          config: [CONFIG],
        })
      }
      if (url.includes(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(organizationContext())
      }
      return jsonResponse([])
    })
}

async function renderOpened(user: ReturnType<typeof userEvent.setup>) {
  renderWithClient(
    <LocalGuardrailsCard docsHref="https://docs.example/tools" />,
  )
  await user.click(
    await screen.findByRole("button", { name: /Configure local guardrails/ }),
  )
}

describe("LocalGuardrailsCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("names a stored guardrail and its task in the words the form asked in", async () => {
    mockApi()
    await renderOpened(userEvent.setup())

    expect(await screen.findByText("prompt-injection")).toBeInTheDocument()
    expect(screen.getByText("Lakera Guard")).toBeInTheDocument()
    expect(screen.getByText("Prompt injection")).toBeInTheDocument()
  })

  it("falls back to the wire name for a class this build no longer ships", async () => {
    mockApi({
      stored: [storedGuardrail({ guardrail_name: "retired_guard" })],
    })
    await renderOpened(userEvent.setup())

    expect(await screen.findByText("retired_guard")).toBeInTheDocument()
    expect(screen.getByText("Not in this build")).toBeInTheDocument()
  })

  it("shows a config-file guardrail read-only, with no controls of its own", async () => {
    mockApi()
    await renderOpened(userEvent.setup())

    expect(await screen.findByText("from-file")).toBeInTheDocument()
    expect(screen.getByText(/config file/)).toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Edit from-file" }),
    ).not.toBeInTheDocument()
  })

  it("warns when the stored secrets cannot be decrypted", async () => {
    mockApi({
      stored: [storedGuardrail({ decryptable: false, create_secrets: {} })],
    })
    await renderOpened(userEvent.setup())

    expect(
      await screen.findByText(/Secrets unreadable: check OTARI_SECRET_KEY/),
    ).toBeInTheDocument()
  })

  it("warns when a stored guardrail overrides a config-file one", async () => {
    mockApi({ stored: [storedGuardrail({ shadows_config: true })] })
    await renderOpened(userEvent.setup())

    expect(
      await screen.findByText(/Overrides the config-file guardrail/),
    ).toBeInTheDocument()
  })

  it("turns one off through the row's switch", async () => {
    const fetchSpy = mockApi()
    const user = userEvent.setup()
    await renderOpened(user)

    await user.click(
      await screen.findByRole("switch", { name: "Run prompt-injection" }),
    )

    await waitFor(() => {
      const patch = fetchSpy.mock.calls.find(
        ([, init]) => init?.method === "PATCH",
      )
      expect(patch).toBeDefined()
      expect(JSON.parse(String(patch?.[1]?.body))).toMatchObject({
        enabled: false,
        expected_updated_at: STORED.updated_at,
      })
    })
  })

  it("asks before removing one, and names what goes with it", async () => {
    const fetchSpy = mockApi()
    const user = userEvent.setup()
    await renderOpened(user)

    await user.click(
      await screen.findByRole("button", { name: "Remove prompt-injection" }),
    )
    expect(
      await screen.findByText(/credentials stored with it are removed/),
    ).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Remove permanently" }))

    await waitFor(() => {
      expect(
        fetchSpy.mock.calls.some(([, init]) => init?.method === "DELETE"),
      ).toBe(true)
    })
  })

  it("keeps a refused remove out of the next row's confirm", async () => {
    mockApi({ writeStatus: 409, writeDetail: "still referenced" })
    const user = userEvent.setup()
    await renderOpened(user)

    await user.click(
      await screen.findByRole("button", { name: "Remove prompt-injection" }),
    )
    await user.click(screen.getByRole("button", { name: "Remove permanently" }))
    expect(await screen.findByText(/still referenced/)).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Cancel" }))
    await user.click(
      await screen.findByRole("button", { name: "Remove prompt-injection" }),
    )
    expect(screen.queryByText(/still referenced/)).not.toBeInTheDocument()
  })

  it("says the read failed rather than claiming the deployment is empty", async () => {
    mockApi({ listStatus: 500 })
    renderWithClient(
      <LocalGuardrailsCard docsHref="https://docs.example/tools" />,
    )

    expect(
      await screen.findByText(
        /Could not read the guardrails this deployment defines/,
      ),
    ).toBeInTheDocument()
    expect(screen.queryByText("0 guardrails")).not.toBeInTheDocument()
  })

  it("refuses the add control while the build ships no runnable catalog", async () => {
    mockApi({ catalog: [] })
    renderWithClient(
      <LocalGuardrailsCard docsHref="https://docs.example/tools" />,
    )

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Add guardrail" }),
      ).toBeDisabled()
    })
  })
})
