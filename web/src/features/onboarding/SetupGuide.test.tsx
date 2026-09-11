import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { StrictMode } from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type { DeploymentBootstrap, WorkspaceActivation } from "@/client"
import { SetupGuide } from "@/features/onboarding/SetupGuide"
import { resetSetupConfetti } from "@/features/onboarding/setupConfetti"
import { API_ROOT } from "@/shared/api/client"
import {
  SelectedWorkspaceProvider,
  useSelectedWorkspace,
} from "@/shared/hooks/SelectedWorkspace"
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
const OTHER_KEY = "gw-other-workspace-key"

const OTHER_WORKSPACE = "55555555-5555-5555-5555-555555555555"

const MEMBERSHIPS = [
  { workspace_id: WORKSPACE, name: "Default Workspace", role: "owner" },
  { workspace_id: OTHER_WORKSPACE, name: "Research", role: "owner" },
]

interface ApiOptions {
  activation?: WorkspaceActivation
  models?: string[]
}

/**
 * The transport, answering each of the guide's reads.
 *
 * The activation state is a variable a test can move, not a queue the transport
 * advances on its own. Minting the key invalidates that query, so a queue would
 * step forward on a refetch nobody asked for and the sheet would change state
 * before the test had touched it.
 */
function mockApi({
  activation = workspaceActivation(),
  models = ["openai:gpt-4o-mini"],
}: ApiOptions = {}) {
  let current = activation
  const mock = vi
    .spyOn(globalThis, "fetch")
    .mockImplementation(async (input, init) => {
      const url = String(input)
      const method = init?.method ?? "GET"
      if (url.includes("/activation/key")) {
        // Distinct per workspace, so a key left over from another one is
        // recognizable rather than indistinguishable.
        const key = url.includes(OTHER_WORKSPACE) ? OTHER_KEY : KEY
        return Response.json({
          key,
          key_id: "88888888-8888-8888-8888-888888888888",
          key_prefix: key.slice(0, 10),
          key_name: "Setup guide",
        })
      }
      if (url.includes("/activation/dismiss")) {
        current = workspaceActivation({
          experience_eligible: false,
          dismissed: true,
        })
        return Response.json({ message: "Setup guide dismissed" })
      }
      if (url.includes("/activation")) {
        return Response.json(current)
      }
      if (url.includes(`${API_ROOT}/models`)) {
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
  /** Moves the state the guide is polling for, the way a real request would. */
  return Object.assign(mock, {
    setActivation(next: WorkspaceActivation) {
      current = next
    },
  })
}

/** Stands in for the shell's workspace switcher. */
function Switcher() {
  const { memberships, select } = useSelectedWorkspace()
  return (
    <>
      {MEMBERSHIPS.map((membership) => (
        <button
          key={membership.workspace_id}
          type="button"
          // Disabled until the organization context resolves, which is what the
          // real switcher does with a list it does not have yet.
          disabled={memberships.length === 0}
          onClick={() => select(membership.workspace_id)}
        >
          switch to {membership.name}
        </button>
      ))}
    </>
  )
}

function renderGuide(
  canServeRequests = true,
  deployment: DeploymentBootstrap = bootstrap(),
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return renderWithRouter(
    <QueryClientProvider client={client}>
      <DeploymentProvider value={deployment}>
        <SelectedWorkspaceProvider>
          <SetupGuide canServeRequests={canServeRequests} />
          <Switcher />
        </SelectedWorkspaceProvider>
      </DeploymentProvider>
    </QueryClientProvider>,
  )
}

/** The snippet on screen, which is a code block rather than a form field. */
function snippet(tab: string): HTMLElement {
  return screen.getByRole("region", { name: `${tab} code` })
}

describe("SetupGuide", () => {
  beforeEach(() => {
    resetSetupConfetti()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    window.localStorage.clear()
  })

  it("offers the sheet, naming the workspace the request will land in", async () => {
    mockApi()
    await renderGuide()

    expect(
      await screen.findByRole("heading", { name: "Send your first request" }),
    ).toBeInTheDocument()
    expect(screen.getByText("Default Workspace")).toBeInTheDocument()
    expect(
      await screen.findByText("Listening for your first request"),
    ).toBeInTheDocument()
  })

  it("holds back while nothing can serve the request", async () => {
    // The Overview's own getting-started panel is the guide at that point, and
    // a key handed out here would be for a call that cannot succeed.
    mockApi()
    await renderGuide(false)

    await waitFor(() => {
      expect(
        screen.queryByRole("heading", { name: "Send your first request" }),
      ).not.toBeInTheDocument()
    })
  })

  it("shows nothing for a workspace that is not being offered the guide", async () => {
    mockApi({ activation: workspaceActivation({ experience_eligible: false }) })
    await renderGuide()

    await waitFor(() => {
      expect(
        screen.queryByRole("heading", { name: "Send your first request" }),
      ).not.toBeInTheDocument()
    })
  })

  it("hands over the key under StrictMode's double-invoked effects", async () => {
    // The bug this pins shipped and was found by hand: the mint fires once
    // across both invocations, so an "ignore a late result" flag scoped to the
    // first invocation discarded the only request in flight and the second
    // declined to replace it. The key minted, arrived, and was dropped, leaving
    // the sheet on "Creating your API key…" forever.
    //
    // Only StrictMode double-invokes, so a production build never showed it and
    // neither did this suite, which renders without one. This is the one test
    // that does.
    const fetchMock = mockApi()
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    })
    await renderWithRouter(
      <StrictMode>
        <QueryClientProvider client={client}>
          <DeploymentProvider value={bootstrap()}>
            <SelectedWorkspaceProvider>
              <SetupGuide canServeRequests />
            </SelectedWorkspaceProvider>
          </DeploymentProvider>
        </QueryClientProvider>
      </StrictMode>,
    )

    expect(await screen.findByLabelText("Your API key")).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.queryByText("Creating your API key…")).toBeNull()
    })
    // And exactly one key was minted, not one per invocation.
    const mints = fetchMock.mock.calls.filter(([input]) =>
      String(input).includes("/activation/key"),
    )
    expect(mints).toHaveLength(1)
  })

  it("mints the key as the sheet opens, and only once", async () => {
    // The sheet is the press: it takes the whole screen and exists to hand a
    // key over, so asking for one more click before showing it is a step with
    // nothing behind it. Once, though: a re-render must not rotate the key an
    // operator is in the middle of copying.
    const fetchMock = mockApi()
    await renderGuide()

    await screen.findByRole("heading", { name: "Send your first request" })
    await waitFor(() => {
      expect(screen.getByLabelText("Your API key")).toBeInTheDocument()
    })
    await waitFor(() => {
      const mints = fetchMock.mock.calls.filter(([input]) =>
        String(input).includes("/activation/key"),
      )
      expect(mints).toHaveLength(1)
    })
  })

  it("says the key is being made rather than showing an empty concealed field", async () => {
    // Every credential field here conceals, so an empty one would show the same
    // run of bullets a real key does and invite a copy that yields nothing.
    let release: (() => void) | undefined
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    const fetchMock = mockApi()
    const real = fetchMock.getMockImplementation()
    fetchMock.mockImplementation(async (input, init) => {
      if (String(input).includes("/activation/key")) await held
      return real?.(input, init) as Promise<Response>
    })
    await renderGuide()

    expect(await screen.findByText("Creating your API key…")).toBeVisible()
    expect(screen.queryByLabelText("Your API key")).toBeNull()

    release?.()
    expect(await screen.findByLabelText("Your API key")).toBeInTheDocument()
  })

  it("does not tell an operator to copy a key that failed to mint", async () => {
    // The contradiction this pins: "the key could not be created" and "copy
    // this key now, it is shown once" on the same first-run screen, where the
    // reader has no context to reconcile them.
    const fetchMock = mockApi()
    const real = fetchMock.getMockImplementation()
    fetchMock.mockImplementation(async (input, init) => {
      if (String(input).includes("/activation/key")) {
        return new Response(JSON.stringify({ detail: "no" }), { status: 500 })
      }
      return real?.(input, init) as Promise<Response>
    })
    await renderGuide()

    await screen.findByRole("heading", { name: "Send your first request" })
    await waitFor(() => {
      expect(screen.queryByText(/Copy this key now/)).toBeNull()
    })
  })

  it("conceals the key and the example built around it until asked", async () => {
    // A key nobody has asked to see is not on screen (otari-ai#2111), and the
    // example carries the same secret, so revealing is one decision.
    mockApi()
    const user = userEvent.setup()
    await renderGuide()

    expect(await screen.findByLabelText("Your API key")).not.toHaveValue(KEY)
    await user.click(await screen.findByRole("button", { name: "cURL" }))
    expect(snippet("curl")).not.toHaveTextContent(KEY)

    await user.click(screen.getByRole("button", { name: "Show Your API key" }))
    expect(screen.getByDisplayValue(KEY)).toBeInTheDocument()
    expect(snippet("curl")).toHaveTextContent(`Otari-Key: ${KEY}`)
  })

  it("offers an agent prompt, cURL, Python and TypeScript", async () => {
    mockApi()
    const user = userEvent.setup()
    await renderGuide()

    // The agent prompt is the default, and names the environment variable
    // rather than carrying the key.
    expect(
      await screen.findByRole("region", { name: "agent code" }),
    ).toHaveTextContent("OTARI_API_KEY")

    for (const [label, region] of [
      ["cURL", "curl"],
      ["Python", "python"],
      ["TypeScript", "typescript"],
    ] as const) {
      await user.click(screen.getByRole("button", { name: label }))
      expect(snippet(region)).toBeInTheDocument()
    }
    await user.click(screen.getByRole("button", { name: "Show Your API key" }))
    // The model comes from the catalog, so the example runs as pasted.
    expect(snippet("typescript")).toHaveTextContent("openai:gpt-4o-mini")
  })

  it("sends the example at the data plane a hosted deployment published", async () => {
    // The control plane serves this guide and is deliberately not where
    // inference belongs (otari#822), so the origin is the one address the
    // example must not name.
    mockApi()
    const person = userEvent.setup()
    await renderGuide(
      true,
      bootstrap({
        deployment_type: "hosted",
        data_plane_url: "https://gateway.otari.ai",
      }),
    )

    await person.click(await screen.findByRole("button", { name: "cURL" }))
    const curl = snippet("curl")
    expect(curl).toHaveTextContent(
      `https://gateway.otari.ai${API_ROOT}/chat/completions`,
    )
    expect(curl).not.toHaveTextContent(window.location.origin)
    // Concealed, so the address it names is readable while the key is not.
    expect(curl).not.toHaveTextContent(KEY)
  })

  it("offers the key but no example when a hosted deployment published no data plane", async () => {
    // Withheld rather than aimed at this host: a placeholder would be a URL
    // nobody reading it could replace, and the origin would be the bug itself.
    mockApi()
    await renderGuide(
      true,
      bootstrap({ deployment_type: "hosted", data_plane_url: null }),
    )

    expect(await screen.findByLabelText("Your API key")).toBeInTheDocument()
    expect(screen.queryByRole("region", { name: /code$/ })).toBeNull()
    expect(
      screen.getByText(/has not published the gateway address/),
    ).toBeInTheDocument()
  })

  it("names the placeholder, and where to fix it, when no model is being served", async () => {
    mockApi({ models: [] })
    await renderGuide()

    expect(
      await screen.findByText(/No model is being served yet/),
    ).toBeVisible()
    expect(screen.getByRole("link", { name: "Models" })).toBeInTheDocument()
  })

  it("reports a failed request with its cause, and keeps listening", async () => {
    mockApi({
      activation: workspaceActivation({
        status: "failed",
        latest_attempt: activationAttempt({
          status: "failed",
          error_category: "policy",
          cost_usd: null,
          latency_ms: null,
        }),
      }),
    })
    await renderGuide()

    expect(
      await screen.findByText(
        /A budget, a model allow-list, or a rate limit rejected the request/,
      ),
    ).toBeInTheDocument()
    // A failure is news, not the end of the guide: the panel keeps its shape
    // and "Check now" stays where it was.
    expect(screen.getByText(/Still listening/)).toBeInTheDocument()
    expect(
      screen.getByRole("link", { name: "Open budgets" }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Check now" }),
    ).toBeInTheDocument()
  })

  it("sends a malformed request to the example that answers it, without leaving", async () => {
    mockApi({
      activation: workspaceActivation({
        status: "failed",
        latest_attempt: activationAttempt({
          status: "failed",
          error_category: "invalid_request",
        }),
      }),
    })
    const user = userEvent.setup()
    await renderGuide()

    await user.click(
      await screen.findByRole("button", {
        name: "Compare with the cURL example",
      }),
    )

    expect(snippet("curl")).toBeInTheDocument()
    // Still on the sheet: the answer to a malformed body is already here.
    expect(
      screen.getByRole("heading", { name: "Send your first request" }),
    ).toBeInTheDocument()
  })

  it("celebrates the first request when it lands while the sheet is open", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    await renderGuide()

    await screen.findByRole("heading", { name: "Send your first request" })
    // The request lands while the sheet is open. The poll interval is longer
    // than a test should sleep for, and "Check now" is the same refetch an
    // operator can trigger.
    fetchMock.setActivation(
      workspaceActivation({
        status: "activated",
        experience_eligible: false,
        activation_attempt: activationAttempt(),
      }),
    )
    await user.click(await screen.findByRole("button", { name: "Check now" }))

    expect(
      await screen.findByRole("heading", {
        name: "Your first call went through",
      }),
    ).toBeInTheDocument()
    expect(screen.getByText(/412 ms/)).toBeInTheDocument()
  })

  it("does not congratulate a workspace that had already activated on arrival", async () => {
    mockApi({
      activation: workspaceActivation({
        status: "activated",
        experience_eligible: false,
        activation_attempt: activationAttempt(),
      }),
    })
    await renderGuide()

    await waitFor(() => {
      expect(
        screen.queryByRole("heading", {
          name: "Your first call went through",
        }),
      ).not.toBeInTheDocument()
    })
  })

  it("mints a fresh key per workspace, never carrying one across", async () => {
    // The key belongs to one workspace, and showing it under another
    // workspace's heading would offer a credential that bills somewhere else.
    // The sheet traps focus, so the switcher is out of reach while it is open;
    // closing it first is the path an operator has, and the guide being keyed
    // on the workspace is what resets the state behind it.
    mockApi()
    const user = userEvent.setup()
    await renderGuide()

    await user.click(
      await screen.findByRole("button", { name: "Show Your API key" }),
    )
    expect(await screen.findByDisplayValue(KEY)).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Close" }))
    await user.click(
      await screen.findByRole("button", { name: "switch to Research" }),
    )

    // The other workspace's sheet, with the other workspace's key: concealed
    // again, because the reveal belonged to the key that is gone.
    await user.click(
      await screen.findByRole("button", { name: "Show Your API key" }),
    )
    expect(await screen.findByDisplayValue(OTHER_KEY)).toBeInTheDocument()
    expect(screen.queryByDisplayValue(KEY)).not.toBeInTheDocument()
  })

  it("skipping retires the guide for the workspace and takes the sheet away", async () => {
    const fetchMock = mockApi()
    const user = userEvent.setup()
    await renderGuide()

    await user.click(await screen.findByRole("button", { name: "Skip" }))

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

  it("closing the sheet is not the same as skipping it", async () => {
    // Escape and the close control put the sheet away for this page load only.
    // Retiring the guide is permanent and the server records it, so it takes a
    // press on the control that says so.
    const fetchMock = mockApi()
    const user = userEvent.setup()
    await renderGuide()

    await user.click(await screen.findByRole("button", { name: "Close" }))

    await waitFor(() => {
      expect(
        screen.queryByRole("heading", { name: "Send your first request" }),
      ).not.toBeInTheDocument()
    })
    expect(
      fetchMock.mock.calls.some(([input]) =>
        String(input).includes("/activation/dismiss"),
      ),
    ).toBe(false)
  })
})
