import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, render, screen, waitFor, within } from "@testing-library/react"
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
import {
  SelectedWorkspaceProvider,
  useSelectedWorkspace,
} from "@/shared/hooks/SelectedWorkspace"
import { organizationContext } from "@/tests/fixtures"
import { withRouter } from "@/tests/router"

/** Mirrors `RATING_ACKNOWLEDGEMENT_MS` in `hooks/usePlayground.ts`. */
const RATING_ACKNOWLEDGEMENT_MS = 3000

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

const SECOND_WORKSPACE_ID = "55555555-5555-5555-5555-555555555555"

function twoWorkspaces(): OrganizationContext {
  return organizationContext({
    workspace_memberships: [
      { workspace_id: WORKSPACE_ID, name: "Default", role: "owner" },
      { workspace_id: SECOND_WORKSPACE_ID, name: "Second", role: "owner" },
    ],
  }) as OrganizationContext
}

/**
 * The switcher's own `select`, exposed as buttons.
 *
 * The real `SelectedWorkspaceProvider` and the real `select` the sidebar's
 * switcher calls, so this drives the actual code path rather than a stub of it;
 * what it skips is the switcher's popover, which belongs to the shell and has
 * its own tests.
 */
function WorkspacePicker() {
  const { memberships, select } = useSelectedWorkspace()
  return (
    <div>
      {memberships.map((membership) => (
        <button
          key={membership.workspace_id}
          type="button"
          onClick={() => select(membership.workspace_id)}
        >
          Select {membership.name}
        </button>
      ))}
    </div>
  )
}

interface ApiState {
  consent?: PlaygroundConsent
  conversations?: PlaygroundConversations
  comparisons?: PlaygroundComparisons
  favorites?: PlaygroundFavoriteModels
  catalog?: ModelListResponse
  messages?: PlaygroundMessages
  context?: OrganizationContext
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
      if (path.startsWith("/organizations/me")) {
        return (state.context ?? context()) as never
      }
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

/**
 * A stream the test drives frame by frame.
 *
 * What a self-closing stream cannot test: anything about a reply still in
 * flight, which is every cancellation rule below. `push` delivers one frame and
 * resolves once the page has folded it in, so a test can put a frame *after* a
 * transition and assert it never lands. The abort signal is honored the way
 * fetch honors it.
 */
function openStream() {
  const queue: string[] = []
  let closed = false
  let wake = () => {}
  const wait = () =>
    new Promise<void>((resolve) => {
      wake = resolve
    })
  let pending = wait()

  const spy = vi
    .spyOn(apiClient, "apiStream")
    .mockImplementation(async (_path, init) => {
      const encoder = new TextEncoder()
      const body = new ReadableStream<Uint8Array>({
        async pull(controller) {
          while (queue.length === 0 && !closed) {
            if (init?.signal?.aborted) break
            await pending
            pending = wait()
          }
          if (init?.signal?.aborted) {
            controller.error(new DOMException("Aborted", "AbortError"))
            return
          }
          const frame = queue.shift()
          if (frame === undefined) {
            controller.close()
            return
          }
          controller.enqueue(encoder.encode(`data: ${frame}\n\n`))
        },
      })
      return new Response(body, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      })
    })

  return {
    spy,
    async push(frame: string) {
      queue.push(frame)
      wake()
      // Let the reader drain it and React commit the fragment.
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 10))
      })
    },
    async close() {
      closed = true
      wake()
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 10))
      })
    },
  }
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
        <WorkspacePicker />
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
  it("identifies the provider when model labels are shared", async () => {
    mockApi({
      catalog: {
        ...CATALOG,
        data: [...CATALOG.data, { ...CATALOG.data[0], id: "backup:gpt-4o" }],
      } as ModelListResponse,
    })
    renderPage()
    await screen.findByText("Try a prompt.")
    expect(screen.getByRole("button", { name: "Model" })).toHaveTextContent(
      "openai:gpt-4o",
    )
  })

  it("filters comparison history from the shared History control", async () => {
    mockApi()
    renderPage()
    await screen.findByText("Try a prompt.")
    await userEvent.click(screen.getByRole("button", { name: "History" }))
    await userEvent.click(screen.getByRole("radio", { name: "Comparisons" }))
    expect(screen.getByRole("radio", { name: "Comparisons" })).toBeChecked()
    expect(await screen.findByText("No saved history yet.")).toBeInTheDocument()
  })

  it("greets, and offers the catalog's first chat model", async () => {
    mockApi()
    renderPage()

    expect(await screen.findByText("Try a prompt.")).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Model" })).toHaveTextContent(
        "gpt-4o",
      )
    })
  })

  it("offers no model the gateway could not chat with", async () => {
    mockApi()
    renderPage()
    await screen.findByText("Try a prompt.")

    await userEvent.click(await screen.findByRole("button", { name: "Model" }))
    expect(
      await screen.findByRole("button", { name: /gpt-4o.*Selected/ }),
    ).toBeInTheDocument()
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
    await screen.findByText("Try a prompt.")

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
    await screen.findByText("Try a prompt.")

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
    await screen.findByText("Try a prompt.")

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
    await screen.findByText("Try a prompt.")

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

  it("offers Regenerate on a reply that failed before its first token", async () => {
    // The moment somebody most wants to retry. Gating the action row on the
    // response alone hid it exactly then, because a failure before the first
    // token leaves the content empty.
    mockApi()
    mockStream([JSON.stringify({ error: "Upstream refused" })])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))

    expect(await screen.findByText("Upstream refused")).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Regenerate response" }),
    ).toBeInTheDocument()
    // And no Copy, because there is nothing to copy.
    expect(
      screen.queryByRole("button", { name: "Copy response" }),
    ).not.toBeInTheDocument()
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
    await screen.findByText("Try a prompt.")

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
    await screen.findByText("Try a prompt.")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("ok")

    await user.click(screen.getByRole("button", { name: "History" }))
    await user.click(screen.getByRole("button", { name: "Save current chat" }))
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
    await screen.findByText("Try a prompt.")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("ok")

    await user.click(screen.getByRole("button", { name: "History" }))
    await user.click(screen.getByRole("button", { name: "Save current chat" }))

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
  it("waits for model B even when Enter is pressed", async () => {
    mockApi()
    const stream = mockStream([delta("answer"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")
    await user.click(screen.getByRole("radio", { name: "Compare" }))
    await user.type(
      screen.getByLabelText("Message"),
      "Compare these answers{Enter}",
    )
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled()
    expect(stream).not.toHaveBeenCalled()
    expect(screen.getByLabelText("Message")).toHaveValue(
      "Compare these answers",
    )
    await user.click(screen.getByRole("button", { name: "Model B" }))
    expect(screen.getByRole("button", { name: "gpt-4o" })).toBeDisabled()
    await user.click(await screen.findByText("claude-sonnet-4"))
    expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled()
  })

  it("asks both models and offers a rating once both answer", async () => {
    mockApi()
    // Two frames, so an answer cut short is observable: a panel aborted after
    // the first would show "an " and never "an answer".
    mockStream([delta("an "), delta("answer"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.click(await screen.findByRole("radio", { name: "Compare" }))
    expect(
      await screen.findByRole("button", { name: "Model A" }),
    ).toBeInTheDocument()
    // Model B is an explicit choice before either request can start.
    expect(screen.getByRole("button", { name: "Model B" })).toHaveTextContent(
      "Choose a model",
    )

    await user.click(screen.getByRole("button", { name: "Model B" }))
    await user.click(await screen.findByText("claude-sonnet-4"))
    await user.type(screen.getByLabelText("Message"), "which?")
    await user.click(screen.getByRole("button", { name: "Send message" }))

    // Both columns answer. Asserted directly rather than only through the
    // rating bar, because the failure this catches is specifically one panel
    // cancelling the other: the two streams run at once, and an earlier version
    // shared one abort controller, so starting B stopped A mid-sentence.
    // `waitFor` on the count, not `findAllByText` then a length assertion:
    // the two panels stream independently, so a `find` resolves on whichever
    // answered first and the length is then read one short.
    await waitFor(() =>
      expect(screen.getAllByText("an answer")).toHaveLength(2),
    )
    expect(
      await screen.findByRole("button", { name: "Model A answered better" }),
    ).toBeInTheDocument()
  })

  it("can record a tie, which the stored vocabulary has always had", async () => {
    const { writes } = mockApi({
      consent: { store_conversations: false, store_comparisons: true },
    })
    mockStream([delta("an answer"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.click(screen.getByRole("radio", { name: "Compare" }))
    await user.click(screen.getByRole("button", { name: "Model B" }))
    await user.click(await screen.findByText("claude-sonnet-4"))
    await user.type(screen.getByLabelText("Message"), "which?")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await user.click(
      await screen.findByRole("button", {
        name: "Both answered equally well",
      }),
    )

    await waitFor(() => {
      expect(
        writes.find((write) => write.url === "/playground/comparisons")?.body,
      ).toMatchObject({ preference: "tie" })
    })
  })

  it("records the rating with both answers", async () => {
    const { writes } = mockApi({
      consent: { store_conversations: false, store_comparisons: true },
    })
    mockStream([delta("an answer"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.click(await screen.findByRole("radio", { name: "Compare" }))
    await user.click(screen.getByRole("button", { name: "Model B" }))
    await user.click(await screen.findByText("claude-sonnet-4"))
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

  it("does not dismiss a fresh rating with the previous one's timer", async () => {
    // The acknowledgement clears itself after RATING_ACKNOWLEDGEMENT_MS. Anything
    // that moves the rating state in the meantime (regenerating an answer, which
    // makes the exchange different from the one that was rated) has to cancel it,
    // or the pending dismissal lands on a fresh unrated exchange and takes the
    // rating controls away from under the operator.
    vi.useFakeTimers({ shouldAdvanceTime: true })
    try {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime })
      mockApi({
        consent: { store_conversations: false, store_comparisons: true },
      })
      mockStream([delta("an answer"), "[DONE]"])
      renderPage()
      await screen.findByText("Try a prompt.")

      await user.click(await screen.findByRole("radio", { name: "Compare" }))
      await user.click(screen.getByRole("button", { name: "Model B" }))
      await user.click(await screen.findByText("claude-sonnet-4"))
      await user.type(screen.getByLabelText("Message"), "which?")
      await user.click(screen.getByRole("button", { name: "Send message" }))
      await user.click(
        await screen.findByRole("button", { name: "Model B answered better" }),
      )
      await screen.findByText("Recorded. Thanks for the feedback.")

      // Regenerate: the exchange on screen is no longer the one that was rated,
      // so the rating controls come back and the pending dismissal is stale.
      mockStream([delta("a different answer"), "[DONE]"])
      await user.click(
        (
          await screen.findAllByRole("button", { name: "Regenerate response" })
        )[0],
      )
      await screen.findByRole("button", { name: "Model B answered better" })

      await act(async () => {
        await vi.advanceTimersByTimeAsync(RATING_ACKNOWLEDGEMENT_MS + 500)
      })

      expect(
        screen.getByRole("button", { name: "Model B answered better" }),
      ).toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
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
    await screen.findByText("Try a prompt.")

    await user.type(await screen.findByLabelText("Message"), "asked before")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    expect(await screen.findByText("asked before")).toBeInTheDocument()

    await user.click(screen.getByRole("radio", { name: "Compare" }))
    await screen.findByRole("button", { name: "Model A" })

    expect(screen.queryByText("asked before")).not.toBeInTheDocument()
    expect(
      await screen.findByText("Choose model B to start comparing."),
    ).toBeInTheDocument()
  })

  it("offers no Save while comparing, because there is no one conversation", async () => {
    mockApi()
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.click(await screen.findByRole("radio", { name: "Compare" }))
    await screen.findByRole("button", { name: "Model A" })
    expect(
      screen.queryByRole("button", { name: "Save current chat" }),
    ).not.toBeInTheDocument()
  })
})

describe("state that must not outlive what produced it", () => {
  it("does not start a second reply when Enter is pressed mid-stream", async () => {
    // The send control becomes Stop while a reply is in flight, so the button
    // path is safe and Enter is the one that is not. Two streams into one panel
    // interleave their fragments into a single turn.
    mockApi()
    const stream = openStream()
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await stream.push(delta("still going"))
    await screen.findByText("still going")
    // The reply is in flight: the composer offers Stop rather than Send.
    expect(
      screen.getByRole("button", { name: "Stop generating" }),
    ).toBeInTheDocument()

    // Re-queried, not reused: the welcome screen and the conversation render
    // their own composer, so the node captured before the first reply is
    // detached by now and typing into it would reach nothing. That is what made
    // an earlier version of this test pass against the missing guard.
    await user.type(screen.getByLabelText("Message"), "again")
    await user.keyboard("{Enter}")

    expect(stream.spy).toHaveBeenCalledTimes(1)
    await stream.close()
  })

  it("cancels a reply in flight before it can land in another layout", async () => {
    // A fragment arriving after the switch would repopulate a column that
    // entering comparison had just cleared.
    mockApi()
    const stream = openStream()
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await stream.push(delta("mid-flight"))
    await screen.findByText("mid-flight")

    await user.click(screen.getByRole("radio", { name: "Compare" }))
    await screen.findByRole("button", { name: "Model A" })

    // The frame that matters: one delivered *after* the switch. An uncancelled
    // stream folds it into the panel that was just cleared.
    await stream.push(delta(" and more"))
    await stream.close()

    expect(screen.queryByText(/mid-flight/)).not.toBeInTheDocument()
    expect(
      screen.getByText("Choose model B to start comparing."),
    ).toBeInTheDocument()
  })

  it("starts over when the selected workspace changes", async () => {
    // A transcript belongs to the workspace whose credentials answered it.
    // Carrying it across would save it under the new workspace and send its
    // history to a model the new catalog may not serve.
    mockApi({ context: twoWorkspaces() })
    mockStream([delta("first workspace answer"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("first workspace answer")

    await user.click(screen.getByRole("button", { name: "Select Second" }))

    expect(await screen.findByText("Try a prompt.")).toBeInTheDocument()
    expect(screen.queryByText("first workspace answer")).not.toBeInTheDocument()
  })

  it("asks before switching model with a transcript on screen", async () => {
    // Each request carries the panel's whole history, so the new model would be
    // answering the old one's conversation, and a save would label that
    // transcript with one model when two answered it.
    mockApi()
    mockStream([delta("first answer"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("first answer")

    await user.click(screen.getByLabelText("Model", { exact: true }))
    await user.click(await screen.findByText("claude-sonnet-4"))

    expect(
      await screen.findByText("Switch model and start over?"),
    ).toBeInTheDocument()
    // Still the old model until they confirm.
    expect(screen.getByLabelText("Model", { exact: true })).toHaveTextContent(
      "gpt-4o",
    )

    await user.click(screen.getByRole("button", { name: "Switch model" }))
    expect(screen.getByLabelText("Model", { exact: true })).toHaveTextContent(
      "claude-sonnet-4",
    )
    expect(screen.queryByText("first answer")).not.toBeInTheDocument()
  })

  it("switches straight away on an empty panel", async () => {
    mockApi()
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.click(await screen.findByLabelText("Model", { exact: true }))
    await user.click(await screen.findByText("claude-sonnet-4"))

    expect(
      screen.queryByText("Switch model and start over?"),
    ).not.toBeInTheDocument()
    expect(screen.getByLabelText("Model", { exact: true })).toHaveTextContent(
      "claude-sonnet-4",
    )
  })
})

describe("when a write fails", () => {
  it("reports a refused save rather than doing nothing", async () => {
    // The Save control has no state of its own to report a failure with, so
    // without this the press looks like it worked.
    const { fetchSpy } = mockApi({
      consent: { store_conversations: true, store_comparisons: false },
    })
    mockStream([delta("ok"), "[DONE]"])
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.type(await screen.findByLabelText("Message"), "hi")
    await user.click(screen.getByRole("button", { name: "Send message" }))
    await screen.findByText("ok")

    const previous = fetchSpy.getMockImplementation()
    fetchSpy.mockImplementation(async (path, init) => {
      if (path === "/playground/conversations" && init?.method === "POST") {
        throw new apiClient.ApiError(
          403,
          "Saving conversations requires consent.",
        )
      }
      return previous?.(path, init) as never
    })

    await user.click(screen.getByRole("button", { name: "History" }))
    await user.click(screen.getByRole("button", { name: "Save current chat" }))

    expect(
      await screen.findByText("Saving conversations requires consent."),
    ).toBeInTheDocument()
  })

  it("reports a transcript that would not load", async () => {
    // The click handler cannot throw: it would be an unhandled rejection and
    // the dialog would simply do nothing.
    const SAVED = {
      id: "conv-1",
      workspace_id: WORKSPACE_ID,
      title: "How does OAuth work",
      model: "openai:gpt-4o",
      message_count: 2,
      created_at: "2026-01-01T00:00:00Z",
    }
    const { fetchSpy } = mockApi({
      consent: { store_conversations: true, store_comparisons: true },
      conversations: { data: [SAVED] },
    })
    const previous = fetchSpy.getMockImplementation()
    fetchSpy.mockImplementation(async (path, init) => {
      if (path.includes("/messages")) {
        throw new apiClient.ApiError(404, "Conversation not found")
      }
      return previous?.(path, init) as never
    })
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.click(await screen.findByRole("button", { name: "History" }))
    const dialog = await screen.findByRole("dialog")
    await user.click(
      within(dialog).getByRole("button", { name: /^How does OAuth work/ }),
    )

    expect(
      await screen.findByText("Conversation not found"),
    ).toBeInTheDocument()
  })
})

describe("history", () => {
  it("opens an empty history before the first save", async () => {
    mockApi()
    renderPage()
    await screen.findByText("Try a prompt.")

    await userEvent.click(screen.getByRole("button", { name: "History" }))
    expect(await screen.findByText("No saved history yet.")).toBeInTheDocument()
  })

  it("falls back to a servable model when the saved one is gone", async () => {
    mockApi({
      consent: { store_conversations: true, store_comparisons: true },
      conversations: {
        data: [
          {
            id: "conv-2",
            workspace_id: WORKSPACE_ID,
            title: "Asked a retired model",
            // Not in CATALOG, which is what a transcript outliving its model
            // looks like: restoring the key would leave the picker blank while
            // Send still dispatched it.
            model: "openai:gpt-4-retired",
            message_count: 1,
            created_at: "2026-01-01T00:00:00Z",
          },
        ],
      },
      messages: { data: [{ role: "user", content: "Asked a retired model" }] },
    })
    const user = userEvent.setup()
    renderPage()
    await screen.findByText("Try a prompt.")

    await user.click(await screen.findByRole("button", { name: "History" }))
    const dialog = await screen.findByRole("dialog")
    await user.click(
      within(dialog).getByRole("button", { name: /^Asked a retired model/ }),
    )

    const picker = await screen.findByRole("button", { name: "Model" })
    expect(picker).toHaveTextContent("gpt-4o")
    expect(picker).not.toHaveTextContent("gpt-4-retired")
  })

  it("loads a saved transcript with its model and exits comparison", async () => {
    const SAVED = {
      id: "conv-1",
      workspace_id: WORKSPACE_ID,
      title: "How does OAuth work",
      model: "anthropic:claude-sonnet-4",
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
    await screen.findByText("Try a prompt.")
    await user.click(screen.getByRole("radio", { name: "Compare" }))

    await user.click(await screen.findByRole("button", { name: "History" }))
    const dialog = await screen.findByRole("dialog")
    // Anchored, so it picks the row and not the delete beside it, whose name
    // is `Delete conversation "How does OAuth work"`.
    await user.click(
      within(dialog).getByRole("button", { name: /^How does OAuth work/ }),
    )

    expect(
      await screen.findByText("It delegates authorization."),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Model" })).toHaveTextContent(
      "claude-sonnet-4",
    )
    expect(
      screen.queryByRole("button", { name: "Model B" }),
    ).not.toBeInTheDocument()
    expect(screen.getByRole("radio", { name: "Compare" })).toBeInTheDocument()
  })
})
