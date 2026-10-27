import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type {
  CountryDatabaseStatus,
  StoredCountryDatabasePublic,
} from "@/client"
import { AuthProvider } from "@/features/auth/AuthContext"
import { CountryDatabaseCard } from "@/features/settings/CountryDatabaseCard"

const STORED: StoredCountryDatabasePublic = {
  database_type: "DBIP-Country-Lite",
  built_at: "2026-10-01T01:36:12Z",
  size_bytes: 8_348_509,
  sha256: "ab".repeat(32),
  uploaded_at: "2026-10-26T13:42:00Z",
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/** Serves the stored file and records each write; `refuseUpload` answers an upload with a 400. */
function mockApi(
  initial: CountryDatabaseStatus,
  { refuseUpload }: { refuseUpload?: string } = {},
) {
  let current = initial
  const writes: { method: string; body: unknown }[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
    const method = (init?.method ?? "GET").toUpperCase()
    if (method === "PUT") {
      writes.push({ method, body: init?.body })
      if (refuseUpload !== undefined) {
        return jsonResponse({ detail: refuseUpload }, 400)
      }
      current = {
        ...current,
        stored: { ...STORED, database_type: "GeoLite2-Country" },
      }
      return jsonResponse(current)
    }
    if (method === "DELETE") {
      writes.push({ method, body: undefined })
      current = { ...current, stored: null }
      return new Response(null, { status: 204 })
    }
    return jsonResponse(current)
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
        <CountryDatabaseCard />
      </AuthProvider>
    </QueryClientProvider>,
  )
}

function fileInput(): HTMLInputElement {
  const input = document.querySelector<HTMLInputElement>(
    'input[type="file"][accept*="mmdb"]',
  )
  if (!input) throw new Error("no country file input")
  return input
}

const MMDB = new File([new Uint8Array([0xab, 0xcd, 0xef])], "countries.mmdb")

describe("CountryDatabaseCard", () => {
  beforeEach(() => {
    window.localStorage.setItem("otari.dashboard.hasSession", "1")
  })

  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("offers an upload when no file is stored", async () => {
    mockApi({ stored: null, client_country_database_set: false })
    renderCard()

    expect(
      await screen.findByText("No country file stored"),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Upload file" })).toBeEnabled()
    expect(screen.queryByRole("button", { name: "Remove" })).toBeNull()
  })

  it("describes the stored file", async () => {
    mockApi({ stored: STORED, client_country_database_set: false })
    renderCard()

    expect(await screen.findByText("DBIP-Country-Lite")).toBeInTheDocument()
    expect(screen.getByText("8 MB")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Replace file" })).toBeEnabled()
    expect(screen.getByRole("button", { name: "Remove" })).toBeEnabled()
  })

  it("uploads the chosen file and shows what it stored", async () => {
    const user = userEvent.setup()
    const writes = mockApi({ stored: null, client_country_database_set: false })
    renderCard()
    await screen.findByText("No country file stored")

    await user.upload(fileInput(), MMDB)

    expect(await screen.findByText("GeoLite2-Country")).toBeInTheDocument()
    expect(writes).toHaveLength(1)
    const body = writes[0].body as FormData
    expect(body.get("file")).toBeInstanceOf(File)
  })

  it("says why an upload was refused", async () => {
    const user = userEvent.setup()
    mockApi(
      { stored: null, client_country_database_set: false },
      {
        refuseUpload:
          "The file is a 'GeoLite2-City' database, not a country database",
      },
    )
    renderCard()
    await screen.findByText("No country file stored")

    await user.upload(fileInput(), MMDB)

    expect(
      await screen.findByText(/not a country database/),
    ).toBeInTheDocument()
    expect(screen.getByText("No country file stored")).toBeInTheDocument()
  })

  it("asks before removing the stored file", async () => {
    const user = userEvent.setup()
    const writes = mockApi({
      stored: STORED,
      client_country_database_set: false,
    })
    renderCard()

    await user.click(await screen.findByRole("button", { name: "Remove" }))
    expect(writes).toHaveLength(0)
    const dialog = await screen.findByRole("alertdialog")
    await user.click(
      within(dialog).getByRole("button", { name: "Remove file" }),
    )

    await waitFor(() =>
      expect(writes).toEqual([{ method: "DELETE", body: undefined }]),
    )
    expect(
      await screen.findByText("No country file stored"),
    ).toBeInTheDocument()
  })

  it("says when a configured file is read instead", async () => {
    mockApi({ stored: STORED, client_country_database_set: true })
    renderCard()

    expect(
      await screen.findByText(/A file stored here is kept but not read/),
    ).toBeInTheDocument()
  })
})
