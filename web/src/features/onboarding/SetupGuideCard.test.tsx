import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { WorkspaceActivation } from "@/client"
import { SetupGuideCard } from "@/features/onboarding/SetupGuideCard"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import {
  activationAttempt,
  bootstrap,
  organizationContext,
  workspaceActivation,
} from "@/tests/fixtures"
import { renderWithRouter } from "@/tests/router"

const WORKSPACE = "44444444-4444-4444-4444-444444444444"
const KEY = "gw-setup-guide-key"

const MEMBERSHIPS = [
  { workspace_id: WORKSPACE, name: "Default Workspace", role: "owner" },
]

interface ApiOptions {
  activations?: WorkspaceActivation[]
  models?: string[]
}

/**
 * The transport, answering each of the card's reads.
 *
 * `activations` is a queue rather than a value, so a test can let the poll see
 * the first request land: each GET takes the next entry and the last one repeats.
 */
function mockApi({
  activations = [workspaceActivation()],
  models = ["openai:gpt-4o-mini"],
}: ApiOptions = {}) {
  const queue = [...activations]
  return vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = init?.method ?? "GET"
      if (url.includes("/activation/key")) {
        return Response.json({
          key: KEY,
          key_id: "88888888-8888-8888-8888-888888888888",
          key_prefix: KEY.slice(0, 10),
          key_name: "Setup guide",
        })
      }
      if (url.includes("/activation/dismiss")) {
        return Response.json({ message: "Setup guide dismissed" })
      }
      if (url.includes("/activation")) {
        const next = queue.length > 1 ? queue.shift() : queue[0]
        return Response.json(next)
      }
      if (url.includes("/v1/models")) {
        return Response.json({
          object: "list",
          data: models.map((id) => ({
            id,
            object: "model",
            created: 0,
            owned_by: "openai",
          })),
        })
      }
      if (method !== "GET") {
        return Response.json({})
      }
      return Response.json(
        organizationContext({ workspace_memberships: MEMBERSHIPS }),
      )
    })
}

function renderCard(hasProviders = true) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={bootstrap()}>
        <SelectedWorkspaceProvider>
          <SetupGuideCard hasProviders={hasProviders} />
        </SelectedWorkspaceProvider>
      </DeploymentProvider>
    </QueryClientProvider>,
  )
}

describe("SetupGuideCard", () => {
  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("offers the guide, naming the workspace the request will land in", async () => {
    mockApi()
    await renderCard()

    expect(
      await screen.findByRole("heading", { name: "Send your first request" }),
    ).toBeInTheDocument()
    expect(screen.getByText("Default Workspace")).toBeInTheDocument()
    expect(
      screen.getByText("Listening for your first request"),
    ).toBeInTheDocument()
  })

  it("holds back while the gateway has no provider to serve the request", async () => {
    // The Overview's own getting-started panel is the guide at that point, and
    // a key handed out here would be for a call that cannot succeed.
    mockApi()
    await renderCard(false)

    await waitFor(() => {
      expect(
        screen.queryByRole("heading", { name: "Send your first request" }),
      ).not.toBeInTheDocument()
    })
  })

  it("shows nothing for a workspace that is not being offered the guide", async () => {
    mockApi({
      activations: [workspaceActivation({ experience_eligible: false })],
    })
    await renderCard()

    await waitFor(() => {
      expect(
        screen.queryByRole("heading", { name: "Send your first request" }),
      ).not.toBeInTheDocument()
    })
  })

  it("issues the key only when asked, then shows it with runnable snippets", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    await renderCard()

    await screen.findByRole("button", { name: "Create a setup key" })
    // Nothing has been minted yet: opening the page must not create a credential.
    expect(
      fetchMock.mock.calls.some(([input]) =>
        String(input).includes("/activation/key"),
      ),
    ).toBe(false)

    await user.click(screen.getByRole("button", { name: "Create a setup key" }))

    expect(await screen.findByDisplayValue(KEY)).toBeInTheDocument()
    const curl = screen.getByDisplayValue(
      new RegExp(`Otari-Key: ${KEY}`),
    ) as HTMLTextAreaElement
    expect(curl.value).toContain(
      `${window.location.origin}/v1/chat/completions`,
    )
    // The model comes from the catalog, so the snippet runs as pasted.
    expect(curl.value).toContain("openai:gpt-4o-mini")
  })

  it("names the placeholder, and where to fix it, when no model is being served", async () => {
    mockApi({ models: [] })
    const user = userEvent.setup()
    await renderCard()

    await user.click(
      await screen.findByRole("button", { name: "Create a setup key" }),
    )

    expect(
      await screen.findByText(/No model is being served yet/),
    ).toBeVisible()
    expect(screen.getByRole("link", { name: "Models" })).toBeInTheDocument()
  })

  it("reports a failed request with its cause and where to fix it", async () => {
    mockApi({
      activations: [
        workspaceActivation({
          status: "failed",
          latest_attempt: activationAttempt({
            status: "failed",
            error_category: "policy",
            cost_usd: null,
            latency_ms: null,
          }),
        }),
      ],
    })
    await renderCard()

    expect(
      await screen.findByText(
        /A budget, a model allow-list, or a rate limit rejected the request/,
      ),
    ).toBeInTheDocument()
    // Still listening: a failure is news, not the end of the guide.
    expect(screen.getByText(/Still listening/)).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Open budgets" }),
    ).toBeInTheDocument()
  })

  it("celebrates the first request when it lands while the guide is on screen", async () => {
    mockApi({
      activations: [
        workspaceActivation(),
        workspaceActivation({
          status: "activated",
          experience_eligible: false,
          activation_attempt: activationAttempt(),
        }),
      ],
    })
    const user = userEvent.setup()
    await renderCard()

    await screen.findByRole("heading", { name: "Send your first request" })
    // The poll interval is longer than a test should sleep for; "Check now" is
    // the same refetch an operator can trigger.
    await user.click(screen.getByRole("button", { name: "Check now" }))

    expect(
      await screen.findByRole("heading", {
        name: "Your first request went through",
      }),
    ).toBeInTheDocument()
    expect(screen.getByText(/412 ms/)).toBeInTheDocument()
  })

  it("does not congratulate a workspace that had already activated on arrival", async () => {
    mockApi({
      activations: [
        workspaceActivation({
          status: "activated",
          experience_eligible: false,
          activation_attempt: activationAttempt(),
        }),
      ],
    })
    await renderCard()

    await waitFor(() => {
      expect(
        screen.queryByRole("heading", {
          name: "Your first request went through",
        }),
      ).not.toBeInTheDocument()
    })
  })

  it("skipping dismisses the guide for the workspace and takes the card away", async () => {
    const fetchMock = mockApi({
      activations: [
        workspaceActivation(),
        workspaceActivation({ experience_eligible: false, dismissed: true }),
      ],
    })
    const user = userEvent.setup()
    await renderCard()

    await user.click(
      await screen.findByRole("button", { name: "Skip this guide" }),
    )

    await waitFor(() => {
      expect(
        screen.queryByRole("heading", { name: "Send your first request" }),
      ).not.toBeInTheDocument()
    })
    const dismissed = fetchMock.mock.calls.find(([input]) =>
      String(input).includes("/activation/dismiss"),
    )
    expect(dismissed).toBeDefined()
    expect(dismissed?.[1]?.method).toBe("POST")
  })
})
