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

  it("narrows to one status code a link carries, and says so", async () => {
    // Home's "See the refusals" links here with the code it counted, so the
    // list is the refused requests rather than every failure.
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(<ActivityPage />, "/activity?status_code=403&range=24h")

    await screen.findByText("gpt-4o")
    const requested = calls.map((c) => c.url)
    for (const path of [`${API_ROOT}/usage?`, `${API_ROOT}/usage/count`]) {
      expect(
        requested.some(
          (url) => url.includes(path) && url.includes("status_code=403"),
        ),
        `${path} dropped the status code`,
      ).toBe(true)
    }
    expect(screen.getByText(/Status code/)).toBeVisible()
  })

  it("narrows to what hosted providers served, and marks those rows", async () => {
    const { calls } = mockApi({ rows: [entry({ deployment_paid: true })] })
    renderPage(<ActivityPage />, "/activity?paid_by=true&range=24h")

    await screen.findByText("gpt-4o")
    const requested = calls.map((c) => c.url)
    for (const path of [
      `${API_ROOT}/usage?`,
      `${API_ROOT}/usage/count`,
      `${API_ROOT}/usage/summary`,
    ]) {
      expect(
        requested.some(
          (url) => url.includes(path) && url.includes("deployment_paid=true"),
        ),
        `${path} dropped the paid-by filter`,
      ).toBe(true)
    }
    expect(selectTrigger("Paid by")).toHaveTextContent("Hosted")
    expect(
      screen.getByRole("img", { name: "Paid by: hosted provider" }),
    ).toBeVisible()
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
