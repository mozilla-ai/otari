import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { OrganizationContext, UpdateCheckStatus } from "@/client"
import { UpdateCheckPrompt } from "@/features/updateCheck/UpdateCheckPrompt"
import { API_ROOT } from "@/shared/api/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  updateCheckStatus,
} from "@/tests/fixtures"

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/** Serves the organization context and the update check, recording each answer. */
function mockApi(
  status: UpdateCheckStatus,
  context: OrganizationContext = organizationContext(),
) {
  let current = status
  const answers: boolean[] = []
  const fetchMock = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(context)
      }
      if (url.endsWith(`${API_ROOT}/update-check`)) {
        if ((init?.method ?? "GET").toUpperCase() === "PUT") {
          const body = JSON.parse(String(init?.body)) as { enabled: boolean }
          answers.push(body.enabled)
          current = {
            ...current,
            enabled: body.enabled,
            answered_by: "operator",
          }
        }
        return jsonResponse(current)
      }
      return jsonResponse({})
    })
  return { answers, fetchMock }
}

function renderPrompt(surfaces?: string[]) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider
        value={bootstrap(surfaces === undefined ? {} : { surfaces })}
      >
        <UpdateCheckPrompt />
      </DeploymentProvider>
    </QueryClientProvider>,
  )
}

const TITLE = "Get notified when a new Otari release is available"

describe("UpdateCheckPrompt", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("asks an operator who has not answered, naming what is sent and where", async () => {
    mockApi(updateCheckStatus())
    renderPrompt()

    const dialog = await screen.findByRole("dialog", { name: TITLE })
    expect(dialog).toHaveTextContent("random install ID to api.otari.ai")
    expect(dialog).toHaveTextContent(
      "sends its version and a random install ID",
    )
    expect(dialog).toHaveTextContent("update_check in config.yml")
    expect(
      screen.getByRole("link", { name: "See exactly what’s sent" }),
    ).toHaveAttribute("href", expect.stringContaining("update-check.md"))
  })

  it.each([
    ["Turn on notifications", true],
    ["No thanks", false],
  ])("stores %s as the answer", async (label, expected) => {
    const { answers } = mockApi(updateCheckStatus())
    const user = userEvent.setup()
    renderPrompt()

    await user.click(await screen.findByRole("button", { name: label }))

    await waitFor(() => expect(answers).toEqual([expected]))
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )
  })

  it("closing without answering answers nothing", async () => {
    const { answers } = mockApi(updateCheckStatus())
    const user = userEvent.setup()
    renderPrompt()

    await screen.findByRole("dialog", { name: TITLE })
    await user.keyboard("{Escape}")

    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )
    expect(answers).toEqual([])
  })

  it.each([
    [
      "config answered",
      updateCheckStatus({ answered_by: "config", enabled: false }),
    ],
    [
      "an operator answered",
      updateCheckStatus({ answered_by: "operator", enabled: true }),
    ],
    ["the build is never checked", updateCheckStatus({ checkable: false })],
  ])("does not ask when %s", async (_case, status) => {
    const { fetchMock } = mockApi(status)
    renderPrompt()

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).endsWith(`${API_ROOT}/update-check`),
        ),
      ).toBe(true),
    )
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("never asks a caller who is not a deployment operator", async () => {
    const { fetchMock } = mockApi(
      updateCheckStatus(),
      organizationContext({ deployment_operator: false }),
    )
    renderPrompt()

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).endsWith(`${API_ROOT}/organizations/me`),
        ),
      ).toBe(true),
    )
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).endsWith(`${API_ROOT}/update-check`),
      ),
    ).toBe(false)
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("never asks where the deployment does not publish the check", async () => {
    const { fetchMock } = mockApi(updateCheckStatus())
    renderPrompt(["settings"])

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).endsWith(`${API_ROOT}/update-check`),
      ),
    ).toBe(false)
  })
})
