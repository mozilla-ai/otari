import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { ActivityPage } from "@/features/activity/ActivityPage"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentScope } from "@/shared/hooks/DeploymentScope"
import { entry, listCalls, mockApi, renderPage } from "@/tests/activity"
import { pickOption, selectTrigger } from "@/tests/select"

// The deployment's page, whose reads go to the deployment-wide `/usage` routes.
function DeploymentActivityPage() {
  return (
    <DeploymentScope>
      <ActivityPage />
    </DeploymentScope>
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("ActivityPage filter serialization", () => {
  it("sends every active filter to the server, not just the chip", async () => {
    // The list, count, and timeline must all carry the active tool filter.
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(
      <DeploymentActivityPage />,
      "/activity?tool=web_search&range=24h",
    )

    await screen.findByText("gpt-4o")
    const requested = calls.map((c) => c.url)
    for (const path of [
      `${API_ROOT}/usage?`,
      `${API_ROOT}/usage/count`,
      `${API_ROOT}/usage/summary`,
    ]) {
      const hit = requested.find((url) => url.includes(path))
      expect(hit, `no request to ${path}`).toBeDefined()
      expect(hit, `${path} dropped the tool filter`).toContain(
        "tool=web_search",
      )
    }
  })

  it("offers Web fetch as a named tool filter", async () => {
    const user = userEvent.setup()
    const { calls } = mockApi({ rows: [entry()] })
    renderPage(
      <DeploymentActivityPage />,
      "/activity?tool=web_search&range=24h",
    )
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
