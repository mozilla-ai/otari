import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ActivityPage } from "@/features/activity/ActivityPage"
import { API_ROOT } from "@/shared/api/client"
import { entry, listCalls, mockApi, renderPage } from "@/tests/activity"
import { pickOption, selectTrigger } from "@/tests/select"

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ActivityPage filter serialization", () => {
  it("sends every active filter to the server, not just the chip", async () => {
    // The list, count, and timeline must all carry the active tool filter.
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />, "/activity?tool=web_search&range=24h")

    await screen.findByText("gpt-4o")
    const requested = calls.map((c) => c.url)
    for (const path of [
      `${API_ROOT}/usage?`,
      `${API_ROOT}/usage/count`,
      `${API_ROOT}/usage/summary`,
    ]) {
      expect(
        requested.some(
          (url) => url.includes(path) && url.includes("tool=web_search"),
        ),
        `${path} dropped the tool filter`,
      ).toBe(true)
    }
  })

  it("offers Web fetch as a named tool filter", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />, "/activity?tool=web_search&range=24h")
    await screen.findByText("gpt-4o")

    await pickOption(user, "Tool", "Web fetch")

    expect(selectTrigger("Tool")).toHaveTextContent("Web fetch")
    await waitFor(() =>
      expect(
        listCalls(calls).some((url) => url.includes("tool=web_fetch")),
      ).toBe(true),
    )
  })
})
