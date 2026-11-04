import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { SandboxCredentialPublic } from "@/client"
import { SandboxCredentialRow } from "@/features/tools/SandboxCredentialRow"
import { API_ROOT } from "@/shared/api/client"

const PATH = `${API_ROOT}/tool-settings/sandbox/credential`

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

function renderRow(credential: SandboxCredentialPublic) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SandboxCredentialRow credential={credential} />
    </QueryClientProvider>,
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("SandboxCredentialRow", () => {
  it("stores a typed key on Enter and empties the field", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () =>
        jsonResponse({
          source: "stored",
          last4: "bbbb",
          env_var: "E2B_API_KEY",
        }),
      )
    const user = userEvent.setup()
    renderRow({ source: "environment", last4: "aaaa", env_var: "E2B_API_KEY" })

    expect(
      screen.getByText(/From E2B_API_KEY, ending aaaa/),
    ).toBeInTheDocument()
    const field = screen.getByLabelText("API key")
    await user.type(field, "e2b_rotated_bbbb{Enter}")

    await waitFor(() => {
      const call = fetchMock.mock.calls.find(([url]) =>
        String(url).includes(PATH),
      )
      expect(call?.[1]?.method).toBe("PUT")
      expect(JSON.parse(String(call?.[1]?.body))).toEqual({
        api_key: "e2b_rotated_bbbb",
      })
    })
    await waitFor(() => expect(field).toHaveValue(""))
  })

  it("writes nothing when the field is left empty", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
    const user = userEvent.setup()
    renderRow({ source: null, last4: null, env_var: "E2B_API_KEY" })

    expect(screen.getByText(/No key: code execution fails/)).toBeInTheDocument()
    await user.click(screen.getByLabelText("API key"))
    await user.tab()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it("clears a stored key, and shows a refusal beside the field", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () =>
        jsonResponse({ detail: "OTARI_SECRET_KEY is not set" }, 500),
      )
    const user = userEvent.setup()
    renderRow({ source: "stored", last4: "cccc", env_var: "E2B_API_KEY" })

    await user.click(
      screen.getByRole("button", { name: "Clear the stored sandbox key" }),
    )

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([url, init]) =>
            String(url).includes(PATH) && init?.method === "DELETE",
        ),
      ).toBe(true),
    )
    expect(
      await screen.findByText(/OTARI_SECRET_KEY is not set/),
    ).toBeInTheDocument()
  })

  it("disables Clear while a key is typed, so it cannot race that key's save", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      jsonResponse({ source: "stored", last4: "dddd", env_var: "E2B_API_KEY" }),
    )
    const user = userEvent.setup()
    renderRow({ source: "stored", last4: "cccc", env_var: "E2B_API_KEY" })

    await user.type(screen.getByLabelText("API key"), "e2b_typed_dddd")

    expect(
      screen.getByRole("button", { name: "Clear the stored sandbox key" }),
    ).toBeDisabled()
  })
})
