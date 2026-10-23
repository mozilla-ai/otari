import { useMemo } from "react"

import type { SummaryDimension, UsageFilters, UsageGroupRow } from "@/client"
import {
  type FilterChip,
  FilterChips,
} from "@/design-system/navigation/FilterChips"
import { FilterMultiComboBox } from "@/design-system/navigation/FilterMultiComboBox"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { STATUS_OPTIONS } from "@/features/activity/activityModel"
import { useMemberAttributionLabels } from "@/features/organization/attribution"
import { userDisplay } from "@/features/users/userDisplay"
import { useUsageSummary } from "@/shared/api/usage"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { type ObserveView, statusApplies, windowFor } from "./observeSearch"
import { useObserveScope } from "./useObserveScope"

// The pickers read these two summaries. by_user and by_api_key carry each
// entity's display name, so naming an option costs nothing beyond the breakdown.
const MODEL_BREAKDOWN: SummaryDimension[] = ["model"]
const ENTITY_BREAKDOWNS: SummaryDimension[] = ["user", "api_key"]

const URL_DEFAULTS = {
  range: "",
  start_date: "",
  end_date: "",
  status: "",
  model: "",
  user_id: "",
  api_key_id: "",
  page: "0",
} as const

const realGroups = (rows: UsageGroupRow[] | undefined) =>
  (rows ?? []).filter((group) => !group.is_other && group.key !== null)

const labelFrom = (
  options: { value: string; label: string }[],
  value: string,
) => options.find((option) => option.value === value)?.label ?? value

/**
 * The filters every Overview tab reads under one name: model, user, API key and
 * status. Their options come from the usage summary over the window the current
 * view shows, each picker's own filter left out so its list stays complete.
 * A filter the current view cannot apply stays visible and removable, marked as
 * not applied, rather than being dropped silently.
 */
export function ObserveFilterBar({
  view,
  usageAvailable = true,
}: {
  view: ObserveView
  // Where the usage surface is not published its summary is not there to read.
  usageAvailable?: boolean
}) {
  const url = useUrlState(URL_DEFAULTS)
  const memberLabels = useMemberAttributionLabels()
  const { selected: workspace } = useSelectedWorkspace()

  const status = url.get("status")
  const models = url.getAll("model")
  const { scope, userIds: users, isResolved } = useObserveScope()
  const isMine = scope === "mine"
  const apiKeys = url.getAll("api_key_id")

  const range = url.get("range")
  const startParam = url.get("start_date")
  const endParam = url.get("end_date")
  // One reading of the clock per selection, so the query key holds between
  // renders. Under the request log's "All" this is a year (`windowFor`), so the
  // pickers offer what was used in the last 365 days, not all time.
  const win = useMemo(
    () =>
      windowFor(view, {
        range,
        start_date: startParam,
        end_date: endParam,
      }),
    [view, range, startParam, endParam],
  )

  const applied = statusApplies(view, status)
  const base: UsageFilters = useMemo(
    () => ({
      workspace_id: workspace?.workspace_id,
      start_date: win.start || undefined,
      end_date: win.end || undefined,
      status: applied && status ? status : undefined,
    }),
    [workspace?.workspace_id, win, applied, status],
  )
  const modelFilters: UsageFilters = useMemo(
    () => ({
      ...base,
      user_id: users.length > 0 ? users : undefined,
      api_key_id: apiKeys.length > 0 ? apiKeys : undefined,
    }),
    [base, users, apiKeys],
  )
  // Under Mine the key list narrows to the caller's own; otherwise the user
  // picker's own filter stays out so its list is complete.
  const entityFilters: UsageFilters = useMemo(
    () => ({
      ...base,
      model: models.length > 0 ? models : undefined,
      user_id: isMine && users.length > 0 ? users : undefined,
    }),
    [base, models, isMine, users],
  )
  const modelSummary = useUsageSummary(
    modelFilters,
    "day",
    MODEL_BREAKDOWN,
    usageAvailable && isResolved,
  )
  const entitySummary = useUsageSummary(
    entityFilters,
    "day",
    ENTITY_BREAKDOWNS,
    usageAvailable && isResolved,
  )

  const modelOptions = realGroups(modelSummary.data?.by_model).map((group) => ({
    value: group.key as string,
    label: group.key as string,
  }))
  // Name first, id in parentheses: the id is what the filter submits, and two
  // people can share a name.
  const userOptions = realGroups(entitySummary.data?.by_user).map((group) => {
    const name = userDisplay(group.key as string, group.label, memberLabels)
    return {
      value: group.key as string,
      label: name.id ? `${name.label} (${name.id})` : name.label,
    }
  })
  const keyOptions = realGroups(entitySummary.data?.by_api_key).map(
    (group) => ({
      value: group.key as string,
      label: group.label ?? `${(group.key as string).slice(0, 8)}…`,
    }),
  )

  const patch = (updates: Parameters<typeof url.patch>[0]) =>
    url.patch({ ...updates, page: 0 })

  const valueChips = (
    dimension: string,
    label: string,
    param: "model" | "user_id" | "api_key_id",
    values: string[],
    display: (value: string) => string,
  ): FilterChip[] =>
    values.map((value) => ({
      key: `${dimension}:${value}`,
      label,
      value: display(value),
      // Several chips share a dimension, so the value has to be part of the name.
      clearLabel: `Remove ${label} filter ${display(value)}`,
      onClear: () =>
        patch({ [param]: values.filter((picked) => picked !== value) }),
    }))
  const chips: FilterChip[] = [
    ...(status
      ? [
          {
            key: "status",
            label: "Status",
            value: applied
              ? labelFrom(STATUS_OPTIONS, status)
              : `${status} (not applied to ${view})`,
            onClear: () => patch({ status: "" }),
          },
        ]
      : []),
    ...(isMine
      ? []
      : valueChips("user", "User", "user_id", users, (value) =>
          labelFrom(userOptions, value),
        )),
    ...valueChips("model", "Model", "model", models, (value) => value),
    ...valueChips("key", "API key", "api_key_id", apiKeys, (value) =>
      labelFrom(keyOptions, value),
    ),
  ]

  return (
    <div className="pb-3">
      <FilterChips
        chips={chips}
        onClearAll={() =>
          patch({ status: "", model: [], user_id: [], api_key_id: [] })
        }
      >
        {/* Usage reads no status, so the select would do nothing there. */}
        {view === "usage" ? null : (
          <FilterSelect
            label="Status"
            value={status}
            onChange={(value) => patch({ status: value })}
            options={STATUS_OPTIONS}
          />
        )}
        {/* allowsCustom on all three: the options are the in-window top spenders
          (a breakdown capped at 100), so an entity below that rank, or with no
          traffic in the window, is not offered. Enter commits a pasted id. */}
        <FilterMultiComboBox
          label="API key"
          values={apiKeys}
          onChange={(values) => patch({ api_key_id: values })}
          allowsCustom
          placeholder="All keys"
          options={keyOptions}
        />
        {isMine ? null : (
          <FilterMultiComboBox
            label="User"
            values={users}
            onChange={(values) => patch({ user_id: values })}
            allowsCustom
            placeholder="All users"
            options={userOptions}
          />
        )}
        <FilterMultiComboBox
          label="Model"
          values={models}
          onChange={(values) => patch({ model: values })}
          allowsCustom
          placeholder="Any model"
          options={modelOptions}
        />
      </FilterChips>
    </div>
  )
}
