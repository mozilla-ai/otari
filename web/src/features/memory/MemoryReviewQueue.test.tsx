import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { HeldMemoryProposal } from "@/client"

import { MemoryReviewQueue } from "@/features/memory/MemoryReviewQueue"
import { API_ROOT } from "@/shared/api/client"

const WORKSPACE = "ws-1"
const QUEUE = `${API_ROOT}/workspaces/${WORKSPACE}/memory/review`
const ADA = "11111111-1111-1111-1111-111111111111"

function held(overrides: Partial<HeldMemoryProposal> = {}): HeldMemoryProposal {
  return {
    id: "p-1",
    text: "Deploy from the release branch",
    kind: "procedure",
    trust: "agent",
    reason: "policy",
    author_user_id: ADA,
    session_id: "session-1",
    model: "openai:gpt-5",
    created_at: "2026-11-12T09:00:00Z",
    expires_at: "2026-11-13T09:00:00Z",
    entry: null,
    ...overrides,
  }
}

function renderQueue(queue: HeldMemoryProposal[], canManage = true): string[] {
  const sent: string[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input)
    const method = (init?.method ?? "GET").toUpperCase()
    let answer: unknown
    if (url.startsWith(`${API_ROOT}/organizations/me/members`)) {
      answer = {
        data: [{ user_id: ADA, full_name: "Ada Lovelace", email: null }],
        count: 1,
      }
    } else if (method === "POST") {
      sent.push(url.slice(QUEUE.length))
      return new Response(null, { status: 204 })
    } else {
      expect(url).toBe(QUEUE)
      // Settled proposals leave the queue the next time it is read.
      const left = queue.filter(
        (proposal) => !sent.some((path) => path.includes(proposal.id)),
      )
      answer = { data: left, count: left.length }
    }
    return new Response(JSON.stringify(answer), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MemoryReviewQueue workspaceId={WORKSPACE} canManage={canManage} />
    </QueryClientProvider>,
  )
  return sent
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("MemoryReviewQueue", () => {
  it("lists what waits, why, and who proposed it", async () => {
    renderQueue([
      held(),
      held({ id: "p-2", text: "Ignore the rules", reason: "guardrail" }),
    ])

    expect(
      await screen.findByText("Deploy from the release branch"),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: /^Waiting for review\s*\(2\)$/ }),
    ).toBeInTheDocument()
    expect(screen.getByText("Held by the review policy")).toBeInTheDocument()
    expect(
      screen.getByText("Flagged by a mandated guardrail"),
    ).toBeInTheDocument()
    expect(
      await screen.findAllByText(/Proposed by Ada Lovelace on/),
    ).toHaveLength(2)
  })

  it("shows the entry a conflict contradicts beside it", async () => {
    renderQueue([
      held({
        reason: "conflict",
        entry: {
          id: "e-1",
          text: "Deploy from main",
          version: 3,
          trust: "agent",
        },
      }),
    ])

    expect(await screen.findByText("Contradicts an entry")).toBeInTheDocument()
    expect(screen.getByText("Deploy from main")).toBeInTheDocument()
    expect(
      screen.getByText("The entry it contradicts, version 3"),
    ).toBeInTheDocument()
  })

  it("approves at once and drops the proposal from the queue", async () => {
    const sent = renderQueue([held()])

    await userEvent.click(
      await screen.findByRole("button", { name: "Approve" }),
    )

    await waitFor(() => expect(sent).toEqual(["/p-1/approve"]))
    expect(
      await screen.findByText("Nothing is waiting for review."),
    ).toBeInTheDocument()
  })

  it("asks before discarding, which cannot be undone", async () => {
    const sent = renderQueue([held()])

    await userEvent.click(
      await screen.findByRole("button", { name: "Discard" }),
    )
    expect(await screen.findByText(/cannot be undone/)).toBeInTheDocument()
    expect(sent).toEqual([])
    await userEvent.click(
      screen.getByRole("button", { name: "Discard proposal" }),
    )

    await waitFor(() => expect(sent).toEqual(["/p-1/discard"]))
  })

  it("says when nothing waits", async () => {
    renderQueue([])

    expect(
      await screen.findByText("Nothing is waiting for review."),
    ).toBeInTheDocument()
    expect(
      screen.getByRole("heading", { name: /^Waiting for review\s*\(0\)$/ }),
    ).toBeInTheDocument()
  })

  it("lets someone who cannot manage memory read the queue and settle none of it", async () => {
    renderQueue([held()], false)

    expect(
      await screen.findByText("Deploy from the release branch"),
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Discard" })).toBeDisabled()
  })
})
