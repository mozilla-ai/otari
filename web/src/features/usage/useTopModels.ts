import type { SpendWindow } from "@/features/spend/period"
import type { SpendRow } from "@/features/spend/useSpend"
import { pickTopModel, type TopModel } from "@/features/usage/topModel"
import { useModelSummariesBy } from "@/shared/api/usage"

export type TopModelState =
  | { status: "loading" }
  | { status: "failed" }
  /** No spend, or no one to ask about: nothing to read. */
  | { status: "none" }
  | { status: "ready"; top: TopModel | undefined }

const NONE: TopModelState = { status: "none" }

/**
 * Each shown row's top model. A person's, a key's or a provider's models take
 * one summary request per row with spend, for the rows on screen only; a
 * workspace's came with its own summary.
 */
export function useTopModels({
  rows,
  cut,
  window,
  workspaceId,
  userId,
}: {
  rows: readonly SpendRow[]
  cut: "people" | "keys" | "models" | "providers" | "workspaces"
  window: SpendWindow
  workspaceId: string
  /** Narrows each row's models to one person's requests, for a member's Home. */
  userId?: string
}): ReadonlyMap<string, TopModelState> {
  const param =
    cut === "people"
      ? "user_id"
      : cut === "providers"
        ? "provider"
        : "api_key_id"
  const isAskable = (row: SpendRow) => row.cost > 0 && row.filterKey !== null
  const queries = useModelSummariesBy(
    param,
    rows.map((row) =>
      (cut === "people" || cut === "keys" || cut === "providers") &&
      isAskable(row)
        ? row.filterKey
        : null,
    ),
    window.start,
    { workspaceId, userId, end: window.end },
  )

  const states = new Map<string, TopModelState>()
  rows.forEach((row, index) => {
    if (cut === "models" || row.cost <= 0) {
      states.set(row.id, NONE)
      return
    }
    if (cut === "workspaces") {
      states.set(
        row.id,
        row.byModel
          ? { status: "ready", top: pickTopModel(row.byModel) }
          : { status: "loading" },
      )
      return
    }
    if (!isAskable(row)) {
      states.set(row.id, NONE)
      return
    }
    const query = queries[index]
    states.set(
      row.id,
      query.data
        ? { status: "ready", top: pickTopModel(query.data.by_model) }
        : query.isError
          ? { status: "failed" }
          : { status: "loading" },
    )
  })
  return states
}
