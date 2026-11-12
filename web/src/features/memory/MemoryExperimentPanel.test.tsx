import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"
import type { MemoryExperiment, WorkspaceMemory } from "@/client"

import { MemoryExperimentPanel } from "@/features/memory/MemoryExperimentPanel"
import { API_ROOT } from "@/shared/api/client"

const WORKSPACE = "ws-1"
const STARTED = "2026-11-11T09:00:00Z"

function memory(overrides: Partial<WorkspaceMemory> = {}): WorkspaceMemory {
  return {
    workspace_id: WORKSPACE,
    deployment_enabled: true,
    organization_enabled: true,
    enabled: true,
    workspace_scope_enabled: true,
    write_policy: "auto",
    offer_to_every_request: false,
    hide_external: false,
    retention_days: null,
    deployment_retention_days: 180,
    auto_recall: true,
    recall_min_similarity: 0.65,
    recall_max_entries: 3,
    recall_rerank: false,
    recall_min_relevance: 0.8,
    rerank_available: false,
    experiment_holdout_percent: 0,
    experiment_started_at: null,
    experiment_ended_at: null,
    entries: {},
    proposals_waiting: 0,
    ...overrides,
  }
}

const arm = (sessions: number, tokens: number) => ({
  sessions,
  requests_per_session: 4,
  tokens_per_session: tokens,
  median_tokens_per_session: tokens,
  input_tokens_per_session: tokens - 100,
  output_tokens_per_session: 100,
  cache_read_tokens_per_session: 0,
  cost_per_session: null,
})

function report(overrides: Partial<MemoryExperiment> = {}): MemoryExperiment {
  return {
    workspace_id: WORKSPACE,
    holdout_percent: 20,
    started_at: STARTED,
    ended_at: null,
    min_sessions: 20,
    control: arm(3, 50_000),
    memory: arm(9, 41_000),
    tokens: null,
    cost: null,
    memory_calls: {},
    memory_call_tokens_per_session: 120,
    recalls: 10,
    recalls_with_results: 4,
    searches: 2,
    ...overrides,
  }
}

function renderPanel(served: WorkspaceMemory, answer?: MemoryExperiment) {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    expect(String(input)).toBe(
      `${API_ROOT}/workspaces/${WORKSPACE}/memory/experiment`,
    )
    return new Response(JSON.stringify(answer), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  })
  const onChange = vi.fn()
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  render(
    <QueryClientProvider client={client}>
      <MemoryExperimentPanel
        workspaceId={WORKSPACE}
        memory={served}
        canManage
        isSaving={false}
        onChange={onChange}
      />
    </QueryClientProvider>,
  )
  return onChange
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("MemoryExperimentPanel", () => {
  it("starts an experiment with the share chosen, and reads nothing before one ran", async () => {
    const onChange = renderPanel(memory())

    await userEvent.click(
      screen.getByRole("button", { name: "Start experiment" }),
    )

    expect(onChange).toHaveBeenCalledWith({ experiment_holdout_percent: 20 })
    expect(globalThis.fetch).not.toHaveBeenCalled()
  })

  it("shows a running experiment's arms, says how many sessions it still needs, and stops it", async () => {
    const onChange = renderPanel(
      memory({
        experiment_holdout_percent: 20,
        experiment_started_at: STARTED,
      }),
      report(),
    )

    expect(await screen.findByText("3 / 9")).toBeInTheDocument()
    expect(
      screen.getByText("Needs 20 sessions in each arm"),
    ).toBeInTheDocument()
    expect(screen.getByText("40%")).toBeInTheDocument()

    await userEvent.click(screen.getByRole("button", { name: "Stop" }))
    expect(onChange).toHaveBeenCalledWith({ experiment_holdout_percent: 0 })
  })

  it("says memory saves once the interval leaves out no change", async () => {
    renderPanel(
      memory({
        experiment_started_at: STARTED,
        experiment_ended_at: "2026-11-18T09:00:00Z",
      }),
      report({
        ended_at: "2026-11-18T09:00:00Z",
        tokens: {
          absolute: -9_000,
          relative: -0.18,
          low: -12_000,
          high: -6_000,
          conclusive: true,
        },
      }),
    )

    expect(await screen.findByText("-18%")).toBeInTheDocument()
    expect(screen.getByText(/^Memory saves;/)).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Start a new experiment" }),
    ).toBeInTheDocument()
  })
})
