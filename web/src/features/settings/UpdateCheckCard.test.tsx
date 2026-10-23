import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { UpdateCheckStatus } from "@/client"
import { UpdateCheckCard } from "@/features/settings/UpdateCheckCard"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { bootstrap, updateCheckStatus } from "@/tests/fixtures"

const SWITCH = "Get notified when a new Otari release is available"

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

function renderCard(status: UpdateCheckStatus, surfaces?: string[]) {
  let current = status
  const answers: boolean[] = []
  const fetchMock = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (_input, init) => {
      if ((init?.method ?? "GET").toUpperCase() === "PUT") {
        const body = JSON.parse(String(init?.body)) as { enabled: boolean }
        answers.push(body.enabled)
        current = { ...current, enabled: body.enabled, answered_by: "operator" }
      }
      return jsonResponse(current)
    })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <DeploymentProvider
        value={bootstrap(surfaces === undefined ? {} : { surfaces })}
      >
        <UpdateCheckCard />
      </DeploymentProvider>
    </QueryClientProvider>,
  )
  return { answers, fetchMock }
}

describe("UpdateCheckCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows exactly what a check sends and where", async () => {
    const status = updateCheckStatus()
    renderCard(status)

    expect(await screen.findByText(status.install_id)).toBeInTheDocument()
    expect(screen.getByText(status.current_version)).toBeInTheDocument()
    expect(screen.getByText(status.endpoint)).toBeInTheDocument()
  })

  it("turns the check on", async () => {
    const { answers } = renderCard(updateCheckStatus())
    const user = userEvent.setup()

    const toggle = await screen.findByRole("switch", { name: SWITCH })
    expect(toggle).toHaveAttribute("aria-checked", "false")
    await user.click(toggle)

    await waitFor(() => expect(answers).toEqual([true]))
    await waitFor(() => expect(toggle).toHaveAttribute("aria-checked", "true"))
  })

  it("locks the switch where config answered, and says so", async () => {
    renderCard(updateCheckStatus({ answered_by: "config", enabled: false }))

    const toggle = await screen.findByRole("switch", { name: SWITCH })
    expect(toggle).toBeDisabled()
    expect(
      screen.getByText(/Decided by this deployment's configuration/),
    ).toBeInTheDocument()
  })

  it("says a development build is never checked", async () => {
    renderCard(
      updateCheckStatus({ checkable: false, current_version: "0.0.0-dev" }),
    )

    expect(
      await screen.findByText(/development build \(0\.0\.0-dev\)/),
    ).toBeInTheDocument()
  })

  it("is absent where the deployment does not publish the check", () => {
    const { fetchMock } = renderCard(updateCheckStatus(), ["settings"])

    expect(screen.queryByText("Update check")).not.toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
