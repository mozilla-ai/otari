import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type {
  OrganizationContext,
  UpdateCheckStatus,
  WorkspaceActivation,
} from "@/client"
import { UpdateCheckPrompt } from "@/features/updateCheck/UpdateCheckPrompt"
import { useWorkspaceActivation } from "@/shared/api/activation"
import { API_ROOT } from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  bootstrap,
  organizationContext,
  updateCheckStatus,
  workspaceActivation,
} from "@/tests/fixtures"

const WORKSPACE = "44444444-4444-4444-4444-444444444444"
const ACTIVATED = workspaceActivation({
  status: "activated",
  experience_eligible: false,
})

/** An operator who belongs to one workspace, the one the shell selects. */
const operatorContext = (overrides: Partial<OrganizationContext> = {}) =>
  organizationContext({
    workspace_memberships: [
      { workspace_id: WORKSPACE, name: "Default Workspace", role: "owner" },
    ],
    ...overrides,
  })

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

/**
 * Serves the organization context, the selected workspace's activation and the
 * update check, recording each answer. `activation` can be moved mid-test, the
 * way a first request landing would move it.
 */
function mockApi(
  status: UpdateCheckStatus,
  context: OrganizationContext = operatorContext(),
  providerCount = 1,
  initialActivation: WorkspaceActivation = ACTIVATED,
) {
  let current = status
  const activation = { current: initialActivation }
  const answers: boolean[] = []
  const fetchMock = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      if (url.endsWith(`${API_ROOT}/organizations/me`)) {
        return jsonResponse(context)
      }
      if (url.endsWith(`${API_ROOT}/providers`)) {
        return jsonResponse({
          providers: Array.from({ length: providerCount }, (_p, index) => ({
            instance: `openai-${index}`,
          })),
        })
      }
      if (url.includes(`${API_ROOT}/hosted-providers`)) {
        return jsonResponse({ data: [] })
      }
      if (url.endsWith(`/workspaces/${WORKSPACE}/activation`)) {
        return jsonResponse(activation.current)
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
  return { answers, fetchMock, activation }
}

/** Stands in for the setup guide, whose observer polls the same activation. */
function GuidePoll() {
  useWorkspaceActivation(WORKSPACE)
  return null
}

function renderPrompt(surfaces?: string[], withGuide = false) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <DeploymentProvider
        value={bootstrap(surfaces === undefined ? {} : { surfaces })}
      >
        <SelectedWorkspaceProvider>
          <UpdateCheckPrompt />
          {withGuide ? <GuidePoll /> : null}
        </SelectedWorkspaceProvider>
      </DeploymentProvider>
    </QueryClientProvider>,
  )
}

const TITLE = "Get alerts for security fixes and new releases?"

describe("UpdateCheckPrompt", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    localStorage.clear()
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
    expect(dialog).toHaveTextContent(
      "otari.ai also uses these checks to count active installs by version",
    )
    expect(
      screen.getByRole("link", { name: "See exactly what’s sent" }),
    ).toHaveAttribute("href", expect.stringContaining("update-check.md"))
  })

  it.each([
    ["Turn on update checks", true],
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

  it("closing without answering answers nothing, and snoozes the question for a week", async () => {
    const { answers } = mockApi(updateCheckStatus())
    const user = userEvent.setup()
    const first = renderPrompt()

    await screen.findByRole("dialog", { name: TITLE })
    await user.keyboard("{Escape}")

    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument(),
    )
    expect(answers).toEqual([])

    first.unmount()
    const later = renderPrompt()
    await waitFor(() =>
      expect(
        later.container.ownerDocument.querySelector("[role=dialog]"),
      ).toBeNull(),
    )

    // A week on, the snooze has run out and the question comes back.
    later.unmount()
    vi.spyOn(Date, "now").mockReturnValue(
      Date.now() + 7 * 24 * 60 * 60 * 1000 + 1,
    )
    renderPrompt()
    expect(
      await screen.findByRole("dialog", { name: TITLE }),
    ).toBeInTheDocument()
  })

  it("shows the pressed answer as pending, and blocks the other", async () => {
    mockApi(updateCheckStatus())
    const fetchMock = vi.mocked(globalThis.fetch)
    const served = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((input, init) =>
      (init?.method ?? "GET").toUpperCase() === "PUT"
        ? new Promise<Response>(() => {})
        : (served?.(input, init) ?? Promise.resolve(jsonResponse({}))),
    )
    const user = userEvent.setup()
    renderPrompt()

    await user.click(await screen.findByRole("button", { name: "No thanks" }))

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /No thanks/ })).toHaveAttribute(
        "data-pending",
        "true",
      ),
    )
    expect(
      screen.getByRole("button", { name: "Turn on update checks" }),
    ).toBeDisabled()
  })

  it("does not ask while the deployment has no provider yet", async () => {
    const { fetchMock } = mockApi(updateCheckStatus(), operatorContext(), 0)
    renderPrompt()

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).includes(`${API_ROOT}/hosted-providers`),
        ),
      ).toBe(true),
    )
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    // Nor reads the activation, whose cache the setup guide would show.
    expect(
      fetchMock.mock.calls.some(([url]) => String(url).includes("/activation")),
    ).toBe(false)
  })

  it("does not ask before the workspace's first successful request", async () => {
    const { fetchMock } = mockApi(
      updateCheckStatus(),
      operatorContext(),
      1,
      workspaceActivation(),
    )
    renderPrompt()

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).endsWith(`/workspaces/${WORKSPACE}/activation`),
        ),
      ).toBe(true),
    )
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("asks an operator who dismissed the setup guide before any request succeeded", async () => {
    mockApi(
      updateCheckStatus(),
      operatorContext(),
      1,
      workspaceActivation({ dismissed: true, experience_eligible: false }),
    )
    renderPrompt()

    expect(
      await screen.findByRole("dialog", { name: TITLE }),
    ).toBeInTheDocument()
  })

  it("does not ask over the setup sheet when the first request lands during this load", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const { fetchMock, activation } = mockApi(
        updateCheckStatus(),
        operatorContext(),
        1,
        workspaceActivation(),
      )
      const activationReads = () =>
        fetchMock.mock.calls.filter(([url]) =>
          String(url).endsWith(`/workspaces/${WORKSPACE}/activation`),
        ).length
      renderPrompt(undefined, true)

      await waitFor(() => expect(activationReads()).toBeGreaterThan(0))
      activation.current = ACTIVATED
      // The guide's poll finds the request a few seconds later.
      const before = activationReads()
      await vi.advanceTimersByTimeAsync(10_000)
      expect(activationReads()).toBeGreaterThan(before)
      await vi.advanceTimersByTimeAsync(100)
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("does not ask when the activation read fails", async () => {
    const { fetchMock } = mockApi(updateCheckStatus())
    const served = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((input, init) =>
      String(input).endsWith(`/workspaces/${WORKSPACE}/activation`)
        ? Promise.resolve(jsonResponse({ detail: "boom" }, 500))
        : (served?.(input, init) ?? Promise.resolve(jsonResponse({}))),
    )
    renderPrompt()

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).endsWith(`/workspaces/${WORKSPACE}/activation`),
        ),
      ).toBe(true),
    )
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("does not ask when the organization context read fails, which leaves no workspace selected", async () => {
    const { fetchMock } = mockApi(updateCheckStatus())
    const served = fetchMock.getMockImplementation()
    fetchMock.mockImplementation((input, init) =>
      String(input).endsWith(`${API_ROOT}/organizations/me`)
        ? Promise.resolve(jsonResponse({ detail: "boom" }, 500))
        : (served?.(input, init) ?? Promise.resolve(jsonResponse({}))),
    )
    renderPrompt()

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) =>
          String(url).includes(`${API_ROOT}/hosted-providers`),
        ),
      ).toBe(true),
    )
    await new Promise((resolve) => setTimeout(resolve, 50))
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("asks an operator who belongs to no workspace, who has no first request to wait for", async () => {
    mockApi(updateCheckStatus(), operatorContext({ workspace_memberships: [] }))
    renderPrompt()

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
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
      operatorContext({ deployment_operator: false }),
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
