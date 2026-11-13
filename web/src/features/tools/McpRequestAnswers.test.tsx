import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { McpServerRequest, WorkspaceMcpServer } from "@/client"
import {
  answerFor,
  McpRequestAnswers,
} from "@/features/tools/McpRequestAnswers"
import { workspaceMcpServer } from "@/tests/fixtures"

const WORKSPACE = "11111111-1111-1111-1111-111111111111"

const LINEAR = workspaceMcpServer({
  id: "22222222-2222-2222-2222-222222222222",
  workspace_id: WORKSPACE,
  name: "Linear",
  credential_mode: "per_user",
})
const SENTRY = workspaceMcpServer({
  id: "33333333-3333-3333-3333-333333333333",
  workspace_id: WORKSPACE,
  name: "Sentry",
})

let made = 0
function request(overrides: Partial<McpServerRequest> = {}): McpServerRequest {
  made += 1
  return {
    id: `aaaaaaaa-aaaa-aaaa-aaaa-${String(made).padStart(12, "0")}`,
    workspace_id: WORKSPACE,
    catalog_name: "app.linear/linear",
    title: "Linear",
    url: "https://mcp.linear.example/mcp",
    note: null,
    status: "approved",
    decision_reason: null,
    decided_at: "2026-11-11T09:00:00+00:00",
    server_id: LINEAR.id,
    catalog_server: null,
    created_at: "2026-11-10T09:00:00+00:00",
    ...overrides,
  }
}

/** Answers the member's own requests, which an acknowledgment then shortens. */
function mockApi(mine: McpServerRequest[], { refuses = false } = {}) {
  const acknowledged: string[] = []
  let own = [...mine]
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    if (url.endsWith("/mcp-server-requests/mine")) {
      return Response.json({ data: own })
    }
    if (url.endsWith("/acknowledge") && init?.method === "POST") {
      if (refuses) return Response.json({ detail: "No" }, { status: 500 })
      const id = url.split("/").at(-2) as string
      acknowledged.push(id)
      own = own.filter((entry) => entry.id !== id)
      return new Response(null, { status: 204 })
    }
    return Response.json({})
  })
  return acknowledged
}

function renderAnswers({
  servers = [LINEAR, SENTRY],
  connected = [],
  onConnect = vi.fn(),
}: {
  servers?: WorkspaceMcpServer[]
  connected?: string[]
  onConnect?: (server: WorkspaceMcpServer) => void
} = {}) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <McpRequestAnswers
        workspaceId={WORKSPACE}
        servers={servers}
        connected={new Set(connected)}
        onConnect={onConnect}
        connectingId=""
      />
    </QueryClientProvider>,
  )
  return onConnect
}

describe("McpRequestAnswers", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("says what each answer was, in the member's words", () => {
    const none = new Set<string>()
    expect(answerFor(request(), [LINEAR], none)?.message).toBe(
      "Linear was added. Connect your account to use it.",
    )
    expect(
      answerFor(request({ server_id: SENTRY.id }), [SENTRY], none)?.message,
    ).toBe("Sentry was added. It is ready to use.")
    expect(
      answerFor(
        request({ server_id: SENTRY.id }),
        [{ ...SENTRY, needs_setup: true }],
        none,
      )?.message,
    ).toBe("Sentry was added. An admin still has to set up its account.")
    expect(
      answerFor(request({ status: "declined", server_id: null }), [], none)
        ?.message,
    ).toBe("Your request for Linear was declined.")
    expect(
      answerFor(
        request({
          status: "declined",
          server_id: null,
          decision_reason: "Finance owns this",
        }),
        [],
        none,
      )?.message,
    ).toBe("Your request for Linear was declined. Reason: Finance owns this")
    // Nothing true to say: waiting, a server the member cannot see, one that is off.
    expect(answerFor(request({ status: "pending" }), [LINEAR], none)).toBe(
      undefined,
    )
    expect(answerFor(request(), [], none)).toBe(undefined)
    expect(answerFor(request(), [{ ...LINEAR, enabled: false }], none)).toBe(
      undefined,
    )
  })

  it("offers Connect on a Personal server and stores a dismissal", async () => {
    const approved = request()
    const declined = request({
      status: "declined",
      server_id: null,
      title: "Stripe",
      decision_reason: "Finance owns this",
    })
    const acknowledged = mockApi([approved, declined])
    const user = userEvent.setup()
    const onConnect = renderAnswers()

    const answers = await screen.findByRole("region", {
      name: "Answers to your requests",
    })
    expect(
      within(answers).getByText(
        "Your request for Stripe was declined. Reason: Finance owns this",
      ),
    ).toBeVisible()
    await user.click(
      within(answers).getByRole("button", { name: "Connect Linear" }),
    )
    expect(onConnect).toHaveBeenCalledWith(LINEAR)
    // Starting a sign-in is not yet an answer seen.
    expect(acknowledged).toEqual([])

    const [, second] = within(answers).getAllByRole("status")
    await user.click(within(second).getByRole("button", { name: "Dismiss" }))
    await waitFor(() => expect(acknowledged).toEqual([declined.id]))
    await waitFor(() =>
      expect(screen.queryByText(/was declined/)).not.toBeInTheDocument(),
    )
    expect(screen.getByText(/Linear was added/)).toBeVisible()
  })

  it("clears the answer once the member's account is connected", async () => {
    const approved = request()
    const acknowledged = mockApi([approved])
    renderAnswers({ connected: [LINEAR.id] })

    await waitFor(() => expect(acknowledged).toEqual([approved.id]))
    expect(screen.queryByText(/Linear was added/)).not.toBeInTheDocument()
  })

  it("folds a long stack behind Show all", async () => {
    mockApi(
      ["Asana", "Box", "Canva", "Deel", "Egnyte"].map((title) =>
        request({ status: "declined", server_id: null, title }),
      ),
    )
    const user = userEvent.setup()
    renderAnswers()

    expect(await screen.findAllByRole("status")).toHaveLength(3)
    await user.click(screen.getByRole("button", { name: "Show all (5)" }))
    expect(screen.getAllByRole("status")).toHaveLength(5)
  })

  it("puts a dismissed answer back when the gateway refuses", async () => {
    mockApi([request({ status: "declined", server_id: null })], {
      refuses: true,
    })
    const user = userEvent.setup()
    renderAnswers()

    await user.click(await screen.findByRole("button", { name: "Dismiss" }))
    expect(await screen.findByRole("alert")).toBeVisible()
    expect(await screen.findByText(/was declined/)).toBeVisible()
  })

  it("renders nothing with nothing to say", async () => {
    mockApi([request({ status: "pending", server_id: null })])
    renderAnswers()

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalled())
    expect(
      screen.queryByRole("region", { name: "Answers to your requests" }),
    ).not.toBeInTheDocument()
  })
})
