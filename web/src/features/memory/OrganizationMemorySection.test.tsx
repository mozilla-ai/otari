import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { OrganizationMemorySection } from "@/features/memory/OrganizationMemorySection"
import { API_ROOT } from "@/shared/api/client"

function mockApi(enabled: boolean): unknown[] {
  const sent: unknown[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    expect(String(input)).toBe(`${API_ROOT}/organizations/me/memory`)
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    if (body !== undefined) sent.push(body)
    return new Response(
      JSON.stringify({
        organization_id: "org-1",
        deployment_enabled: true,
        enabled: body?.enabled ?? enabled,
      }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    )
  })
  return sent
}

async function renderSection() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <OrganizationMemorySection />
    </QueryClientProvider>,
  )
  const toggle = await screen.findByRole("switch", { name: "Agent memory" })
  await waitFor(() => expect(toggle).toBeEnabled())
  return toggle
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("OrganizationMemorySection", () => {
  it("says before turning memory off that every workspace stops and nothing is deleted", async () => {
    const sent = mockApi(true)
    const toggle = await renderSection()
    expect(toggle).toBeChecked()

    await userEvent.click(toggle)
    expect(
      await screen.findByText(/What each workspace holds stays/),
    ).toBeInTheDocument()
    expect(sent).toEqual([])
    await userEvent.click(
      screen.getByRole("button", { name: "Turn off memory" }),
    )

    await waitFor(() => expect(sent).toEqual([{ enabled: false }]))
    await waitFor(() =>
      expect(
        screen.getByRole("switch", { name: "Agent memory" }),
      ).not.toBeChecked(),
    )
  })

  it("turns memory on for the organization once confirmed", async () => {
    const sent = mockApi(false)
    await userEvent.click(await renderSection())

    await userEvent.click(
      await screen.findByRole("button", { name: "Turn on memory" }),
    )

    await waitFor(() => expect(sent).toEqual([{ enabled: true }]))
  })
})
