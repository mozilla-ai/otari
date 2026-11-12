import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceMemory } from "@/client"

import { WorkspaceMemorySection } from "@/features/memory/WorkspaceMemorySection"
import { API_ROOT } from "@/shared/api/client"

const WORKSPACE = "ws-1"
const URL = `${API_ROOT}/workspaces/${WORKSPACE}/memory`

interface Sent {
  method: string
  body: unknown
}

function memory(overrides: Partial<WorkspaceMemory> = {}): WorkspaceMemory {
  return {
    workspace_id: WORKSPACE,
    deployment_enabled: true,
    organization_enabled: true,
    enabled: true,
    workspace_scope_enabled: false,
    write_policy: "review",
    offer_to_every_request: false,
    hide_external: false,
    retention_days: null,
    deployment_retention_days: 180,
    auto_recall: false,
    recall_min_similarity: 0.75,
    recall_max_entries: 3,
    recall_rerank: false,
    recall_min_relevance: 0.5,
    rerank_available: true,
    experiment_holdout_percent: 0,
    experiment_started_at: null,
    experiment_ended_at: null,
    entries: { active: 3, quarantined: 1 },
    proposals_waiting: 2,
    ...overrides,
  }
}

function mockApi(served: WorkspaceMemory): Sent[] {
  const sent: Sent[] = []
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const method = (init?.method ?? "GET").toUpperCase()
    expect(String(input)).toBe(URL)
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    if (method !== "GET") sent.push({ method, body })
    const answer =
      method === "DELETE"
        ? {
            workspace_id: WORKSPACE,
            entries: 3,
            proposals: 2,
            reads: 4,
            keys: 1,
          }
        : method === "PUT"
          ? { ...served, ...body }
          : served
    return new Response(JSON.stringify(answer), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
  return sent
}

async function renderSection(served: WorkspaceMemory, canManage = true) {
  const sent = mockApi(served)
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <WorkspaceMemorySection workspaceId={WORKSPACE} canManage={canManage} />
    </QueryClientProvider>,
  )
  await screen.findByRole("switch", { name: "Agent memory" })
  return sent
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("WorkspaceMemorySection", () => {
  it("turns on automatic recall with every other switch as it was", async () => {
    const sent = await renderSection(memory({ offer_to_every_request: true }))
    expect(screen.queryByLabelText("How alike an entry must be")).toBeNull()

    await userEvent.click(
      screen.getByRole("switch", { name: "Recall automatically" }),
    )

    await waitFor(() => expect(sent).toHaveLength(1))
    expect(sent[0]).toEqual({
      method: "PUT",
      body: expect.objectContaining({
        enabled: true,
        offer_to_every_request: true,
        auto_recall: true,
        recall_min_similarity: 0.75,
        recall_max_entries: 3,
        experiment_holdout_percent: 0,
      }),
    })
    expect(
      await screen.findByText("Stricter adds less, and less often."),
    ).toBeInTheDocument()
  })

  it("shows the switches and what the workspace holds", async () => {
    await renderSection(memory())

    expect(screen.getByRole("switch", { name: "Agent memory" })).toBeChecked()
    expect(
      screen.getByRole("switch", { name: "Review shared proposals" }),
    ).toBeChecked()
    expect(
      screen.getByText(
        "3 entries in recall, 2 proposals not decided yet, 1 flagged out of recall.",
      ),
    ).toBeInTheDocument()
    expect(
      screen.queryByText(/off for this organization/),
    ).not.toBeInTheDocument()
  })

  it("says when the organization keeps memory off", async () => {
    await renderSection(memory({ organization_enabled: false }))

    expect(
      screen.getByText(/Memory is off for this organization/),
    ).toBeInTheDocument()
  })

  it("asks before turning memory on, then sends every switch", async () => {
    const sent = await renderSection(memory({ enabled: false }))

    await userEvent.click(screen.getByRole("switch", { name: "Agent memory" }))
    expect(sent).toEqual([])
    await userEvent.click(
      await screen.findByRole("button", { name: "Turn on memory" }),
    )

    await waitFor(() => expect(sent).toHaveLength(1))
    expect(sent[0]).toEqual({
      method: "PUT",
      body: {
        enabled: true,
        workspace_scope_enabled: false,
        write_policy: "review",
        offer_to_every_request: false,
        hide_external: false,
        retention_days: null,
        auto_recall: false,
        recall_min_similarity: 0.75,
        recall_max_entries: 3,
        recall_rerank: false,
        recall_min_relevance: 0.5,
        experiment_holdout_percent: 0,
      },
    })
  })

  it("checks relevance, and then asks how relevant rather than how alike", async () => {
    const sent = await renderSection(memory({ auto_recall: true }))
    expect(
      screen.getByLabelText("How alike an entry must be"),
    ).toBeInTheDocument()

    await userEvent.click(
      screen.getByRole("switch", { name: "Check relevance" }),
    )

    await waitFor(() => expect(sent).toHaveLength(1))
    expect(sent[0]).toEqual({
      method: "PUT",
      body: expect.objectContaining({
        auto_recall: true,
        recall_rerank: true,
        recall_min_relevance: 0.5,
        recall_min_similarity: 0.75,
      }),
    })
    expect(
      await screen.findByLabelText("How relevant an entry must be"),
    ).toBeInTheDocument()
    expect(screen.queryByLabelText("How alike an entry must be")).toBeNull()
  })

  it("says when the deployment has no rerank model to check relevance with", async () => {
    await renderSection(
      memory({
        auto_recall: true,
        recall_rerank: true,
        rerank_available: false,
      }),
    )

    expect(
      screen.getByRole("switch", { name: "Check relevance" }),
    ).toBeDisabled()
    expect(
      screen.getByText(/names no rerank model \(memory_rerank_model\)/),
    ).toBeInTheDocument()
    expect(
      screen.getByLabelText("How alike an entry must be"),
    ).toBeInTheDocument()
  })

  it("applies a switch as it is flipped", async () => {
    const sent = await renderSection(memory())

    await userEvent.click(
      screen.getByRole("switch", { name: "Offer memory to every request" }),
    )

    await waitFor(() => expect(sent).toHaveLength(1))
    expect(sent[0]?.body).toMatchObject({
      offer_to_every_request: true,
      enabled: true,
    })
  })

  it("keeps a retention outside its bounds from being saved", async () => {
    const sent = await renderSection(memory())
    const field = screen.getByRole("textbox", { name: /Keep unused entries/ })

    await userEvent.type(field, "0")
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled()
    await userEvent.clear(field)
    await userEvent.type(field, "30")
    await userEvent.click(screen.getByRole("button", { name: "Save" }))

    await waitFor(() => expect(sent).toHaveLength(1))
    expect(sent[0]?.body).toMatchObject({ retention_days: 30 })
  })

  it("asks before erasing, which cannot be undone", async () => {
    const sent = await renderSection(memory())

    await userEvent.click(
      screen.getByRole("button", { name: "Erase this workspace's memory" }),
    )
    expect(await screen.findByText(/cannot be undone/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole("button", { name: "Erase memory" }))

    await waitFor(() =>
      expect(sent).toEqual([{ method: "DELETE", body: undefined }]),
    )
  })

  it("lets a member read the switches and change none", async () => {
    await renderSection(memory(), false)

    expect(screen.getByRole("switch", { name: "Agent memory" })).toBeDisabled()
    expect(
      screen.getByRole("button", { name: "Erase this workspace's memory" }),
    ).toBeDisabled()
    expect(
      screen.getByText(/Only the organization's owners and admins/),
    ).toBeInTheDocument()
  })
})
