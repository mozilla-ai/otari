import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import type {
  ModelListResponse,
  OrganizationContext,
  PlaygroundComparisons,
  PlaygroundConsent,
  PlaygroundConversations,
  PlaygroundFavoriteModels,
  PlaygroundMessages,
  PlaygroundTools,
} from "@/client"
import { PlaygroundPage } from "@/features/playground/PlaygroundPage"
import * as apiClient from "@/shared/api/client"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { organizationContext } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

const WORKSPACE_ID = "44444444-4444-4444-4444-444444444444"

const CATALOG: ModelListResponse = {
  object: "list",
  data: [
    {
      id: "openai:gpt-4o",
      object: "model",
      created: 0,
      owned_by: "openai",
      pricing_source: "none",
    },
    {
      id: "anthropic:claude-sonnet-4",
      object: "model",
      created: 0,
      owned_by: "anthropic",
      pricing_source: "none",
    },
    // Not a chat model: it must not reach the picker.
    {
      id: "openai:text-embedding-3-small",
      object: "model",
      created: 0,
      owned_by: "openai",
      pricing_source: "none",
    },
  ],
} as ModelListResponse

const NO_TOOLS: PlaygroundTools = {
  web_search: {
    configured: false,
    enabled: false,
    reason: "No backend is configured on this deployment.",
  },
  code_execution: {
    configured: false,
    enabled: false,
    reason: "No backend is configured on this deployment.",
  },
  mcp_servers: [],
}

function context(): OrganizationContext {
  return organizationContext({
    workspace_memberships: [
      { workspace_id: WORKSPACE_ID, name: "Default", role: "owner" },
    ],
  }) as OrganizationContext
}

interface ApiState {
  consent?: PlaygroundConsent
  conversations?: PlaygroundConversations
  comparisons?: PlaygroundComparisons
  favorites?: PlaygroundFavoriteModels
  catalog?: ModelListResponse
  messages?: PlaygroundMessages
}

/**
 * The transport, and only the transport.
 *
 * Every hook, query key, derivation and formatter below it is the real one,
 * which is the point: mocking `usePlaygroundConsent` would hide a changed key
 * or an unrendered loading state, which are the regressions worth catching
 * here.
 */
function mockApi(state: ApiState = {}) {
  const writes: { url: string; method: string; body: unknown }[] = []
  const fetchSpy = vi
    .spyOn(apiClient, "apiFetch")
    .mockImplementation(async (path, init) => {
      const method = (init?.method ?? "GET").toUpperCase()
      if (method !== "GET") {
        writes.push({
          url: path,
          method,
          body: init?.body ? JSON.parse(String(init.body)) : undefined,
        })
      }
      if (path.startsWith("/organizations/me")) return context() as never
      if (path.startsWith("/models")) {
        return (state.catalog ?? CATALOG) as never
      }
      if (path.startsWith("/playground/tools")) return NO_TOOLS as never
      if (path.startsWith("/playground/consent")) {
        return (state.consent ?? {
          store_conversations: false,
          store_comparisons: false,
        }) as never
      }
      if (path.includes("/messages")) {
        return (state.messages ?? { data: [] }) as never
      }
      if (path.startsWith("/playground/conversations")) {
        return (state.conversations ?? { data: [] }) as never
      }
      if (path.startsWith("/playground/comparisons")) {
        return (state.comparisons ?? { data: [] }) as never
      }
      if (path.startsWith("/playground/favorite-models")) {
        return (state.favorites ?? { model_keys: [] }) as never
      }
      throw new Error(`unexpected read: ${path}`)
    })
  return { fetchSpy, writes }
}

/**
 * A streaming response built from the frames a gateway would emit.
 *
 * Two details are load-bearing, and both were added after a test passed
 * against a bug it was written to catch.
 *
 * Frames arrive one `pull` at a time with a macrotask between them rather than
 * all enqueued up front, so two streams are genuinely in flight at once: a
 * stream that completes before its caller returns makes every concurrency
 * assertion here vacuous.
 *
 * And the abort signal is honored, the way `fetch` honors it, by erroring the
 * body. A mock that accepts a signal and ignores it cannot fail a test about
 * cancellation, which is exactly what the compare test below needs.
 */
function sseResponse(frames: string[], signal?: AbortSignal | null): Response {
  const encoder = new TextEncoder()
  let next = 0
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      await new Promise((resolve) => setTimeout(resolve, 0))
      if (signal?.aborted) {
        controller.error(new DOMException("Aborted", "AbortError"))
        return
      }
      if (next >= frames.length) {
        controller.close()
        return
      }
      controller.enqueue(encoder.encode(`data: ${frames[next]}\n\n`))
      next += 1
    },
  })
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  })
}

function mockStream(frames: string[]) {
  return vi
    .spyOn(apiClient, "apiStream")
    .mockImplementation(async (_path, init) =>
      sseResponse(frames, init?.signal),
    )
}

const delta = (content: string) =>
  JSON.stringify({ choices: [{ delta: { content } }] })

function renderPage() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return render(
    <QueryClientProvider client={client}>
      <SelectedWorkspaceProvider>
        <PlaygroundPage />
      </SelectedWorkspaceProvider>
    </QueryClientProvider>,
    { wrapper: withRouter({ url: "/playground" }) },
  )
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
})

afterEach(() => {
  vi.restoreAllMocks()
  localStorage.clear()
})

describe("the Playground before the first question", () => {
  it("greets, and offers the catalog's first chat model", async () => {
    mockApi()
    renderPage()

    expect(await screen.findByText("What can I help with?")).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Model" })).toHaveTextContent(
        "openai:gpt-4o",
      )
    })
  })

  it("offers no model the gateway could not chat with", async () => {
    mockApi()
    renderPage()
    await screen.findByText("What can I help with?")

    await userEvent.click(await screen.findByRole("button", { name: "Model" }))
    expect(await screen.findByText("gpt-4o")).toBeInTheDocument()
    expect(screen.queryByText("text-embedding-3-small")).not.toBeInTheDocument()
  })

  it("says what to do when the deployment serves no models", async () => {
    mockApi({ catalog: { object: "list", data: [] } as ModelListResponse })
    renderPage()

    expect(
      await screen.findByText("No models to chat with"),
    ).toBeInTheDocument()
  })
})

describe("sending a question", () => {
  it("renders the reply as it streams in", async () => {
    // The whole reason this is SSE rather than one JSON response: the frames
    // arrive separately and the page has to fold them into one answer.
    mockApi()
    mockStream([delta("Hel"), delta("lo there"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))

    expect(await screen.findByText("Hello there")).toBeInTheDocument()
    // And the question is on screen above it, so the exchange reads in order.
    expect(screen.getByText("hi")).toBeInTheDocument()
  })

  it("sends the workspace and the selected model", async () => {
    mockApi()
    const stream = mockStream([delta("ok"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("ok")

    const [path, init] = stream.mock.calls[0] ?? []
    expect(path).toBe(
      `/playground/chat/completions?workspace_id=${WORKSPACE_ID}`,
    )
    expect(JSON.parse(String(init?.body))).toMatchObject({
      model: "openai:gpt-4o",
      stream: true,
      messages: [{ role: "user", content: "hi" }],
    })
  })

  it("shows the per-turn stats the final frame reported", async () => {
    mockApi()
    mockStream([
      delta("ok"),
      JSON.stringify({
        choices: [],
        usage: {
          prompt_tokens: 1200,
          completion_tokens: 34,
          total_tokens: 1234,
        },
      }),
      "[DONE]",
    ])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))

    expect(await screen.findByText(/1,200 in · 34 out/)).toBeInTheDocument()
  })

  it("stops on request, keeping what had arrived and reporting no failure", async () => {
    // Somebody pressing Stop already knows why it stopped, so the partial reply
    // stays as it is and nothing is reported as an error.
    mockApi()
    mockStream([delta("partial "), delta("and more"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))

    await user.click(
      await screen.findByRole("button", { name: "Stop generating" }),
    )

    // Send is offered again, which is how the page says the request is over.
    expect(
      await screen.findByRole("button", { name: "Send message" }),
    ).toBeInTheDocument()
    expect(screen.getByText(/partial/)).toBeInTheDocument()
    expect(screen.queryByText(/Aborted/)).not.toBeInTheDocument()
  })

  it("reports a mid-stream failure on the conversation", async () => {
    // A failure after the headers arrives as a frame, because the status is
    // already 200 by then. It belongs where the answer would have been.
    mockApi()
    mockStream([
      delta("partial"),
      JSON.stringify({ error: "Upstream refused" }),
    ])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))

    expect(await screen.findByText("Upstream refused")).toBeInTheDocument()
    expect(screen.getByText("partial")).toBeInTheDocument()
  })
})

describe("retention consent", () => {
  it("asks before the first save, then saves", async () => {
    // The page asks at the moment it needs the grant. The server refuses a save
    // without one either way, so this is the prompt rather than the gate.
    const { writes } = mockApi()
    mockStream([delta("ok"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("ok")

    await user.click(screen.getByRole("button", { name: "Save conversation" }))
    expect(
      await screen.findByText("Save this conversation?"),
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => {
      expect(
        writes.some(
          (write) =>
            write.url === "/playground/consent" &&
            (write.body as { store_conversations?: boolean })
              .store_conversations === true,
        ),
      ).toBe(true)
    })
    await waitFor(() => {
      const save = writes.find(
        (write) => write.url === "/playground/conversations",
      )
      expect(save?.body).toMatchObject({
        workspace_id: WORKSPACE_ID,
        model: "openai:gpt-4o",
        title: "hi",
      })
    })
  })

  it("does not ask again once it has been granted", async () => {
    const { writes } = mockApi({
      consent: { store_conversations: true, store_comparisons: false },
    })
    mockStream([delta("ok"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("ok")

    await user.click(screen.getByRole("button", { name: "Save conversation" }))

    await waitFor(() => {
      expect(
        writes.some((write) => write.url === "/playground/conversations"),
      ).toBe(true)
    })
    expect(
      screen.queryByText("Save this conversation?"),
    ).not.toBeInTheDocument()
  })
})

describe("comparing two models", () => {
  it("asks both models and offers a rating once both answer", async () => {
    mockApi()
    // Two frames, so an answer cut short is observable: a panel aborted after
    // the first would show "an " and never "an answer".
    mockStream([delta("an "), delta("answer"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.click(
      await screen.findByRole("button", { name: "Compare two models" }),
    )
    expect(
      await screen.findByRole("button", { name: "Model A" }),
    ).toBeInTheDocument()
    // The second column starts on a different model: comparing a model with
    // itself produces two answers nobody can tell apart.
    expect(
      screen.getByRole("button", { name: "Model B" }),
    ).not.toHaveTextContent("openai:gpt-4o")

    await user.type(screen.getByLabelText("Message"), "which?")
    await user.click(screen.getByRole("button", { name: "Send message" }))

    // Both columns answer. Asserted directly rather than only through the
    // rating bar, because the failure this catches is specifically one panel
    // cancelling the other: the two streams run at once, and an earlier version
    // shared one abort controller, so starting B stopped A mid-sentence.
    expect(await screen.findAllByText("an answer")).toHaveLength(2)
    expect(
      await screen.findByRole("button", { name: "Model A answered better" }),
    ).toBeInTheDocument()
  })

  it("records the rating with both answers", async () => {
    const { writes } = mockApi({
      consent: { store_conversations: false, store_comparisons: true },
    })
    mockStream([delta("an answer"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.click(
      await screen.findByRole("button", { name: "Compare two models" }),
    )
    await user.type(screen.getByLabelText("Message"), "which?")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await user.click(
      await screen.findByRole("button", { name: "Model B answered better" }),
    )

    await waitFor(() => {
      const rating = writes.find(
        (write) => write.url === "/playground/comparisons",
      )
      expect(rating?.body).toMatchObject({
        workspace_id: WORKSPACE_ID,
        user_question: "which?",
        model_a: "openai:gpt-4o",
        model_a_answer: "an answer",
        model_b_answer: "an answer",
        preference: "model_b",
      })
    })
  })

  it("starts both columns level, clearing what single view had", async () => {
    // The two models are each sent their own column's history, so a column
    // carrying an earlier conversation answers a different prompt from the one
    // beside it, and a rating over that pair is a judgment on an unequal
    // contest. The hosted original kept the first panel's transcript; this
    // deliberately does not.
    mockApi()
    mockStream([delta("an answer"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.type(await screen.findByLabelText("Message"), "asked before")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    expect(await screen.findByText("asked before")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Compare two models" }))
    await screen.findByRole("button", { name: "Model A" })

    expect(screen.queryByText("asked before")).not.toBeInTheDocument()
    expect(await screen.findAllByText("Send a message to start.")).toHaveLength(
      2,
    )
  })

  it("offers no Save while comparing, because there is no one conversation", async () => {
    mockApi()
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.click(
      await screen.findByRole("button", { name: "Compare two models" }),
    )
    await screen.findByRole("button", { name: "Model A" })
    expect(
      screen.queryByRole("button", { name: "Save conversation" }),
    ).not.toBeInTheDocument()
  })
})

describe("history", () => {
  it("hides the history control until there is history", async () => {
    mockApi()
    renderPage()
    await screen.findByText("What can I help with?")

    expect(
      screen.queryByRole("button", { name: "Conversation history" }),
    ).not.toBeInTheDocument()
  })

  it("loads a saved transcript into the conversation", async () => {
    const SAVED = {
      id: "conv-1",
      workspace_id: WORKSPACE_ID,
      title: "How does OAuth work",
      model: "openai:gpt-4o",
      message_count: 2,
      created_at: "2026-01-01T00:00:00Z",
    }
    mockApi({
      consent: { store_conversations: true, store_comparisons: true },
      conversations: { data: [SAVED] },
      messages: {
        data: [
          { role: "user", content: "How does OAuth work" },
          { role: "assistant", content: "It delegates authorization." },
        ],
      },
    })
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("What can I help with?")

    await user.click(
      await screen.findByRole("button", { name: "Conversation history" }),
    )
    const dialog = await screen.findByRole("dialog")
    // Anchored, so it picks the row and not the delete beside it, whose name
    // is `Delete conversation "How does OAuth work"`.
    await user.click(
      within(dialog).getByRole("button", { name: /^How does OAuth work/ }),
    )

    expect(
      await screen.findByText("It delegates authorization."),
    ).toBeInTheDocument()
  })
})
