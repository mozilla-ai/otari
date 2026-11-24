import { Link, useNavigate } from "@tanstack/react-router"
import { type ReactNode, useState } from "react"
import { FiDollarSign, FiKey, FiLayers, FiUsers } from "react-icons/fi"
import type { UsageSummary } from "@/client"
import { RowMenu, type RowMenuAction } from "@/design-system/actions/RowMenu"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { useIsMobile } from "@/design-system/hooks/useIsMobile"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { Sparkline } from "@/design-system/metrics/charts"
import { KpiCell } from "@/design-system/metrics/KpiCell"
import { KpiStrip } from "@/design-system/metrics/KpiStrip"
import { SpendMeter } from "@/design-system/metrics/SpendMeter"
import { TrendChip } from "@/design-system/metrics/TrendChip"
import { FilterSelect } from "@/design-system/navigation/FilterSelect"
import { Segmented } from "@/design-system/navigation/Segmented"
import { MemberAvatar } from "@/features/account/avatar/MemberAvatar"
import { breakdownCut } from "@/features/home/breakdownCut"
import { busiestWorkspace } from "@/features/home/busiestWorkspace"
import { canOpenInKeys } from "@/features/home/keyAccess"
import { useHomeAccount } from "@/features/home/overlayHomeAccount"
import { hasProviderSlot, ProviderSlot } from "@/features/home/ProviderSlot"
import { ShareCell } from "@/features/home/ShareCell"
import { RouterSavingsTile } from "@/features/routerSavings/RouterSavingsTile"
import { limitDialogCopy } from "@/features/spend/limitCopy"
import {
  type AppliedLimit,
  type LimitTarget,
  limitLine,
  usedShare,
  wholePct,
} from "@/features/spend/limits"
import { paceOf } from "@/features/spend/pace"
import {
  comparedWith,
  DEFAULT_PERIOD,
  figureLabel,
  periodFromUrl,
  sessionsSearchFor,
  usageRange,
  windowOf,
  windowWords,
} from "@/features/spend/period"
import { SetLimitDialog } from "@/features/spend/SetLimitDialog"
import { SpendPeriodPicker } from "@/features/spend/SpendPeriodPicker"
import { usePageClock } from "@/features/spend/usePageClock"
import {
  filterValues,
  type SpendRow,
  useSpend,
  useWorkspaceSpend,
} from "@/features/spend/useSpend"
import { WorkspaceDefaultLimit } from "@/features/spend/WorkspaceDefaultLimit"
import { TopModelCell } from "@/features/usage/TopModelCell"
import { billedTokenTotal, cacheSums } from "@/features/usage/usageTotals"
import { useTopModels } from "@/features/usage/useTopModels"
import { useKeys } from "@/shared/api/apiKeys"
import { apiFetch } from "@/shared/api/client"
import { useTraceCount } from "@/shared/api/traces"
import { useUsageScope } from "@/shared/api/usage"
import { useWorkspaces } from "@/shared/api/workspaces"
import {
  deltaFraction,
  formatNumber,
  formatPct,
  formatTokens,
  formatUsd,
  formatUsdHeadline,
} from "@/shared/helpers/format"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { useSurfaces } from "@/shared/hooks/useDeployment"

/** Rows shown before the rest folds behind "Show all". */
const TOP_ROWS = 5

type Cut = "people" | "keys" | "models" | "providers" | "workspaces"

const CUTS: {
  value: Cut
  label: string
  /** The first column's head. */
  noun: string
  /** What "Show all 19 …" counts. */
  plural: string
  /**
   * The filter a row opens on, where a row is one: Sessions', or Requests' for
   * a provider, which Sessions cannot filter on.
   */
  filter?: "user_id" | "api_key_id" | "model" | "provider"
}[] = [
  {
    value: "people",
    label: "People",
    noun: "Person",
    plural: "people",
    filter: "user_id",
  },
  {
    value: "keys",
    label: "API keys",
    noun: "API key",
    plural: "API keys",
    filter: "api_key_id",
  },
  {
    value: "models",
    label: "Models",
    noun: "Model",
    plural: "models",
    filter: "model",
  },
  {
    value: "providers",
    label: "Providers",
    noun: "Provider",
    plural: "providers",
    filter: "provider",
  },
  {
    value: "workspaces",
    label: "Workspaces",
    noun: "Workspace",
    plural: "workspaces",
  },
]

const URL_DEFAULTS = { period: DEFAULT_PERIOD, by: "", workspace: "" }

/** The limit figure's label, by how often the organization's limit resets. */
const LIMIT_LABELS: Record<string, string | undefined> = {
  "a day": "Of the daily limit",
  "a week": "Of the weekly limit",
  "a month": "Of the monthly limit",
}

/** Where a limit lands if its period keeps this pace (`paceOf`). */
function paceLine(limit: AppliedLimit, now: number): string {
  const line = `of ${formatUsdHeadline(limit.max)}`
  const pace = paceOf({
    spent: limit.spent,
    max: limit.max,
    periodStart: limit.ceiling.period_start,
    periodEnd: limit.ceiling.period_end,
    now,
  })
  return pace
    ? `${line}, on pace for ${formatUsd(pace.projected).replace(/\.\d+$/, "")}`
    : line
}

function LimitCell({ row }: { row: SpendRow }) {
  if (row.limits.isUnknown) {
    return <span className="text-caption">Not loaded</span>
  }
  const limit = row.limits.binding
  if (!limit) return <span className="text-muted">No limit</span>
  const more = row.limits.count - 1
  // Whose limit it is when a person has one of their own or shares one, then
  // how many others reach the row: the one shown is the closest to its cap.
  const note = [
    limit.origin
      ? `everyone ${limit.origin}`
      : row.id.startsWith("person:")
        ? "their own"
        : undefined,
    more > 0 ? `${more} more` : undefined,
  ]
    .filter(Boolean)
    .join(" · ")
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <span className="flex min-w-0 flex-wrap items-baseline gap-x-2.5">
        <span className="text-mono-caption whitespace-nowrap text-foreground">
          {limitLine(limit)}
        </span>
        {note ? (
          <span className="text-caption whitespace-nowrap">{note}</span>
        ) : null}
      </span>
      <SpendMeter
        spent={limit.spent}
        allocated={limit.max}
        ariaLabel={`${row.name}: limit spent`}
        className="max-w-[16rem]"
      />
    </div>
  )
}

/**
 * Home for the people who run the organization: what the window cost against
 * the limit, and where it went, with the limit each person, key and workspace
 * works under on its own row.
 *
 * Each band answers one question. Needs attention (`TeamAttentionBand`) is
 * held back until it reads each row's scope right; it is not mounted.
 * - The figures: how much, against what limit, how many requests failed (or
 *   a figure the edition adds, `overlayHomeAccount.tsx`), and how much of the
 *   team is using it.
 * - Where it went: who or what spent it, and what stops each of them.
 */
export function TeamHome({
  notice,
}: {
  /** A band under the title, such as the first step on a fresh gateway. */
  notice?: ReactNode
}) {
  // One clock for the page's life, so its windows and query keys hold still.
  const [now, refreshClock] = usePageClock()
  const navigate = useNavigate()
  const { get, patch } = useUrlState(URL_DEFAULTS)
  const { select } = useSelectedWorkspace()
  const workspaces = useWorkspaces()
  // A tile the edition that holds the organization's money adds, in place of
  // Failures; none in this build.
  const account = useHomeAccount()
  const usageScope = useUsageScope("organization")
  const [editing, setEditing] = useState<LimitTarget>()
  const [isShowingAll, setShowingAll] = useState(false)
  const isMobile = useIsMobile()
  // The list `useSpend` reads, so this asks nothing new.
  const keys = useKeys()

  const period = periodFromUrl(get("period"))
  const window = windowOf(period, now)
  const workspaceId = get("workspace")
  const hasWorkspaces = (workspaces.data?.length ?? 0) > 1

  const spend = useSpend(window, {
    workspaceId: workspaceId || undefined,
    resetEndedAt: now,
  })
  const members = spend.people.filter((row) => row.isMember)
  // Alone in the organization, "who spent it" has one answer: lead with keys.
  const isTeam = members.length > 1
  const cuts = CUTS.filter((one) => one.value !== "workspaces" || hasWorkspaces)
  const shownCut = breakdownCut({
    chosen: cuts.find((one) => one.value === get("by"))?.value,
    roster: spend.reads.roster,
    isTeam,
  })
  // Until there is one, the table under it is loading whichever it is.
  const cut =
    cuts.find((one) => one.value === shownCut) ??
    cuts.find((one) => one.value === "keys") ??
    cuts[0]
  const byWorkspace = useWorkspaceSpend(window, cut.value === "workspaces", now)
  // A deployment that records no sessions serves no route to count them, and
  // has none to open. Over the page's own scope: a workspace it picked, or
  // every one.
  const hasSurface = useSurfaces()
  const hasSessions = hasSurface("traces")
  const hasRequests = hasSurface("usage")
  const sessions = useTraceCount(
    { workspaceId, start: window.start, end: window.end || undefined, q: "" },
    hasSessions,
  )

  const total = spend.totals?.cost ?? 0
  const rows: SpendRow[] =
    cut.value === "people"
      ? spend.people
      : cut.value === "keys"
        ? spend.keys
        : cut.value === "models"
          ? spend.models
          : cut.value === "providers"
            ? spend.providers
            : byWorkspace.rows
  const isLoading =
    cut.value === "workspaces" ? byWorkspace.isLoading : spend.isLoading
  // A workspace's own total is the whole organization's when none is picked.
  const tableTotal =
    cut.value === "workspaces"
      ? byWorkspace.rows.reduce((sum, row) => sum + row.cost, 0)
      : total
  // One unread row leaves every share unknown, not only its own.
  const isTotalKnown = !rows.some((row) => row.isUnread)
  const visible = isShowingAll ? rows : rows.slice(0, TOP_ROWS)
  const hidden = rows.length - visible.length
  const topModels = useTopModels({
    rows: visible,
    cut: cut.value,
    window,
    workspaceId,
  })
  const pickCut = (next: string) => {
    setShowingAll(false)
    patch({ by: next })
  }
  const hasMarkSlot = cut.value === "providers" && hasProviderSlot(rows)
  // What a row opens: a workspace narrows this table, a provider opens its
  // requests, and anything else its sessions.
  const canOpenRows =
    cut.value === "workspaces" ||
    (cut.value === "providers" ? hasRequests : hasSessions)

  /**
   * Sessions lists one workspace, so a row opens in the workspace its
   * sessions are in: the one filtered to, a key's own, or, for someone in
   * several, the one they spent the most in over this window.
   */
  const workspaceFor = async (row: SpendRow): Promise<string | undefined> => {
    if (workspaceId) return workspaceId
    // A model's or a provider's spend is in any workspace, so every one is a
    // candidate; a person's or a key's are the ones they are in.
    const candidates =
      row.workspaceIds ?? (workspaces.data ?? []).map((one) => one.id)
    const { filter } = cut
    if (candidates.length <= 1 || !filter) return candidates[0]
    const costs = await Promise.all(
      candidates.map((candidate) => {
        const params = new URLSearchParams({
          workspace_id: candidate,
          ...usageRange(window),
          bucket: "day",
          dimensions: "none",
        })
        for (const value of filterValues(row)) params.append(filter, value)
        return apiFetch<UsageSummary>(`${usageScope.base}/summary?${params}`)
          .then((summary) => summary.totals.cost)
          .catch(() => undefined)
      }),
    )
    return busiestWorkspace(candidates, costs)
  }

  const open = async (row: SpendRow) => {
    if (cut.value === "workspaces") {
      // A workspace opens as its own people, which is the next question.
      patch({ workspace: row.filterKey ?? "", by: "people" })
      return
    }
    if (!cut.filter || !row.filterKey) return
    const target = await workspaceFor(row)
    if (target) select(target)
    void navigate({
      to: cut.value === "providers" ? "/requests" : "/sessions",
      search: {
        ...sessionsSearchFor(window),
        [cut.filter]: filterValues(row),
      },
    })
  }

  const menuFor = (row: SpendRow): RowMenuAction[] => {
    const actions: RowMenuAction[] = []
    const target = row.target
    if (target) {
      actions.push({
        id: "limit",
        label: target.own ? "Change spend limit…" : "Set spend limit…",
        hint: target.own
          ? limitLine(target.own)
          : cut.value === "people"
            ? "On everything they spend"
            : cut.value === "keys"
              ? "On this key alone"
              : "On everyone in it, together",
        icon: FiDollarSign,
        run: () => setEditing(target),
      })
    }
    if (cut.value === "workspaces") {
      actions.push({
        id: "open",
        label: "See its people",
        hint: "This table, filtered to that workspace",
        icon: FiUsers,
        run: () => void open(row),
      })
    } else if (row.filterKey && canOpenRows) {
      actions.push({
        id: "open",
        label:
          cut.value === "people"
            ? "See their sessions"
            : cut.value === "providers"
              ? "See its requests"
              : "See its sessions",
        hint: `What ran ${windowWords(window)}`,
        icon: FiLayers,
        run: () => void open(row),
      })
    }
    // Only a key the caller's keys endpoint lists, which is every key for the
    // operator and their own and service keys for anyone else.
    if (cut.value === "keys" && canOpenInKeys(row.filterKey, keys.data ?? [])) {
      actions.push({
        id: "keys",
        label: "Open in API keys",
        hint: "Disable, edit or regenerate it",
        icon: FiKey,
        run: () => {
          const home = row.workspaceIds?.[0]
          if (home) select(home)
          void navigate({ to: "/keys" })
        },
      })
    }
    return actions
  }

  const columns: DataTableColumn<SpendRow>[] = [
    {
      id: "name",
      header: cut.noun,
      isRowHeader: true,
      cell: (row) => (
        <span
          className={`flex min-w-0 items-center gap-3 md:max-w-[13rem] ${cut.value === "models" ? "xl:max-w-[24rem]" : ""}`}
        >
          {cut.value === "people" ? (
            <MemberAvatar userId={row.avatarUserId} face={row.avatarFace} />
          ) : hasMarkSlot ? (
            <ProviderSlot row={row} />
          ) : null}
          <span className="flex min-w-0 flex-col">
            {/* A model is told apart by its whole name, so a long id wraps,
                to two lines; the title holds every id it ran as. */}
            <span
              className={`text-foreground ${cut.value === "models" ? "line-clamp-2 [overflow-wrap:anywhere]" : "truncate"}`}
              title={row.spellings?.join("\n")}
            >
              {row.name}
            </span>
            {row.detail && cut.value !== "people" ? (
              <span className="truncate text-caption">{row.detail}</span>
            ) : null}
          </span>
        </span>
      ),
    },
    {
      id: "spend",
      header: "Spent",
      align: "end",
      cell: (row) => (
        <span className="text-mono-caption text-foreground">
          {row.isUnread || row.isFolded ? "—" : formatUsd(row.cost)}
          {isMobile && isTotalKnown && !row.isFolded && tableTotal > 0 ? (
            <span className="text-muted">
              {" · "}
              {wholePct(row.cost / tableTotal)}
            </span>
          ) : null}
        </span>
      ),
    },
    {
      id: "tokens",
      header: "Tokens",
      align: "end",
      cell: (row) => (
        <span className="text-mono-caption text-foreground">
          {row.isUnread || row.isFolded ? "—" : formatTokens(row.tokens)}
        </span>
      ),
    },
    ...(cut.value !== "models"
      ? ([
          {
            id: "model",
            header: "Top model",
            cell: (row) => (
              <TopModelCell
                state={topModels.get(row.id)}
                rowCost={row.cost}
                isInline={isMobile}
              />
            ),
          },
        ] satisfies DataTableColumn<SpendRow>[])
      : []),
    ...(isMobile
      ? []
      : ([
          {
            id: "share",
            header: "Share",
            cell: (row) => (
              <ShareCell
                share={
                  isTotalKnown && !row.isFolded && tableTotal > 0
                    ? row.cost / tableTotal
                    : undefined
                }
              />
            ),
          },
        ] satisfies DataTableColumn<SpendRow>[])),
    // A provider's limit is set on a stored key, not on the name a request
    // records, so no row here has one to show.
    ...(cut.value !== "models" && cut.value !== "providers"
      ? ([
          {
            id: "limit",
            header: "Limit",
            cell: (row) => <LimitCell row={row} />,
          },
        ] satisfies DataTableColumn<SpendRow>[])
      : []),
    {
      id: "actions",
      header: <span className="sr-only">Actions</span>,
      align: "end",
      cell: (row) => (
        <RowMenu label={`Actions for ${row.name}`} actions={menuFor(row)} />
      ),
    },
  ]

  const organization = spend.organization.limits.binding
  const orgTarget = spend.organization.target
  const limitsRead = spend.reads.limits
  const delta = deltaFraction(total, spend.previousCost)
  // The people of the scope the figures cover: a picked workspace's members,
  // or the whole organization's.
  const inScope = workspaceId
    ? members.filter((row) => row.workspaceIds?.includes(workspaceId))
    : members
  const active = inScope.filter((row) => row.cost > 0).length
  const scopeName = workspaceId
    ? (workspaces.data?.find((one) => one.id === workspaceId)?.name ??
      "this workspace")
    : "all workspaces"
  const isActiveKnown =
    spend.reads.roster === "ready" && spend.reads.usage === "ready"
  // A figure with no read behind it: still coming, or not coming.
  const unreadLine =
    spend.reads.usage === "failed" ? "Could not load" : "Loading"
  const isTableFailed =
    cut.value === "workspaces"
      ? workspaces.isError && workspaces.data === undefined
      : spend.reads.usage === "failed"
  const errorRate = spend.totals?.request_count
    ? spend.totals.error_count / spend.totals.request_count
    : 0
  const sessionCount = sessions.data?.count
  const cacheTotals = cacheSums(spend.series)

  return (
    <div className="flex flex-col gap-6">
      <PageIntro
        title="Home"
        action={
          <div className="flex flex-wrap items-center justify-end gap-2">
            <SpendPeriodPicker
              value={period}
              onChange={(next) => {
                refreshClock()
                patch({ period: next })
              }}
            />
            {hasWorkspaces ? (
              <FilterSelect
                ariaLabel="Workspace"
                value={workspaceId}
                onChange={(next) => patch({ workspace: next })}
                options={[
                  { value: "", label: "All workspaces" },
                  ...(workspaces.data ?? []).map((one) => ({
                    value: one.id,
                    label: one.name,
                  })),
                ]}
              />
            ) : null}
          </div>
        }
      />
      {notice}
      <ErrorBanner
        error={
          spend.error ??
          (cut.value === "workspaces" ? byWorkspace.error : undefined)
        }
      />
      <KpiStrip isEmpty={false} columns={5}>
        <KpiCell
          label={figureLabel("Spent", window)}
          value={spend.totals ? formatUsd(total) : "—"}
          delta={
            delta !== null ? (
              <TrendChip
                fraction={delta}
                polarity="neutral"
                caption={comparedWith(window)}
              />
            ) : undefined
          }
          subline={
            !spend.totals
              ? unreadLine
              : delta === null
                ? `${formatNumber(spend.totals.request_count)} requests`
                : undefined
          }
          graphic={
            spend.series.length > 1 ? (
              <Sparkline
                values={spend.series.map((point) => point.cost)}
                ariaLabel={`Spend per ${window.bucket} over the window`}
                height={40}
              />
            ) : undefined
          }
        />
        <KpiCell
          label={
            organization
              ? (LIMIT_LABELS[organization.per] ?? "Of the team budget")
              : "Team budget"
          }
          value={
            organization
              ? wholePct(usedShare(organization))
              : limitsRead === "ready"
                ? "None"
                : "—"
          }
          subline={
            organization
              ? paceLine(organization, now)
              : limitsRead === "ready"
                ? "No team budget is set"
                : limitsRead === "loading"
                  ? "Loading"
                  : "Could not load"
          }
          graphic={
            <span className="flex w-full flex-col gap-2">
              {organization ? (
                <SpendMeter
                  spent={organization.spent}
                  allocated={organization.max}
                  ariaLabel="Team budget spent"
                />
              ) : null}
              {orgTarget ? (
                <button
                  type="button"
                  onClick={() => setEditing(orgTarget)}
                  className="self-start text-xs text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
                >
                  {organization ? "Change spend limit" : "Set a team budget"}
                </button>
              ) : null}
            </span>
          }
        />
        {account.tile ? (
          <KpiCell
            label={account.tile.label}
            value={account.tile.value}
            subline={account.tile.subline}
            graphic={account.tile.action}
          />
        ) : (
          <KpiCell
            label="Failures"
            value={spend.totals ? formatPct(errorRate) : "—"}
            subline={
              spend.totals
                ? `${formatNumber(spend.totals.error_count)} failed ${spend.totals.error_count === 1 ? "request" : "requests"}`
                : unreadLine
            }
          />
        )}
        <KpiCell
          label={figureLabel("Tokens", window)}
          value={
            spend.totals
              ? formatTokens(billedTokenTotal(spend.totals) ?? 0)
              : "—"
          }
          subline={
            !spend.totals
              ? unreadLine
              : cacheTotals.input > 0
                ? `${formatPct(cacheTotals.read / cacheTotals.input)} of input from cache`
                : `${formatNumber(spend.totals.request_count)} requests`
          }
          graphic={
            spend.series.length > 1 ? (
              <Sparkline
                values={spend.series.map(
                  (point) => point.input_tokens + point.output_tokens,
                )}
                ariaLabel={`Billed tokens per ${window.bucket} over the window`}
                height={40}
              />
            ) : undefined
          }
        />
        <KpiCell
          label="People active"
          value={
            isActiveKnown ? `${active} of ${Math.max(inScope.length, 1)}` : "—"
          }
          subline={
            sessionCount === undefined
              ? `in ${scopeName}`
              : `${formatNumber(sessionCount)} ${sessionCount === 1 ? "session" : "sessions"} ${windowWords(window)}`
          }
          graphic={
            hasSessions ? (
              <Link
                to="/sessions"
                search={sessionsSearchFor(window)}
                className="self-start text-xs text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
              >
                See the sessions
              </Link>
            ) : undefined
          }
        />
      </KpiStrip>
      <RouterSavingsTile window={window} workspaceId={workspaceId} />
      <section aria-label="Where it went" className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
            <h2 className="text-title">Where it went</h2>
            {/* A segmented track holds four choices at most, and a phone
              fits three at a larger text size. */}
            {cuts.length > (isMobile ? 3 : 4) ? (
              <FilterSelect
                ariaLabel="Break down by"
                value={cut.value}
                onChange={pickCut}
                options={cuts}
              />
            ) : (
              <Segmented
                label="Break down by"
                size="md"
                options={cuts}
                value={shownCut ?? ""}
                onChange={pickCut}
              />
            )}
          </div>
        </div>
        {isTeam && cut.value === "people" ? <WorkspaceDefaultLimit /> : null}
        <TableScrollFrame className="otari-breakdown otari-spend-table flex flex-col gap-2">
          {/* Which columns is the roster's answer, so none show before it. */}
          {shownCut === undefined ? (
            <PageLoading />
          ) : (
            <DataTable
              key={cut.value}
              ariaLabel={`Spend by ${cut.label.toLowerCase()}`}
              columns={columns}
              rows={isLoading ? [] : visible}
              getRowKey={(row) => row.id}
              isLoading={isLoading}
              emptyContent={
                <EmptyMessage>
                  {isTableFailed
                    ? "Could not load"
                    : period === "6m"
                      ? "Nothing was spent in this window."
                      : "Nothing was spent in this window. Pick a longer one above."}
                </EmptyMessage>
              }
              onRowAction={
                canOpenRows
                  ? (key) => {
                      const row = rows.find((one) => one.id === key)
                      if (row?.filterKey) void open(row)
                    }
                  : undefined
              }
              // A row that opens nothing (the rest, summed) keeps the arrow.
              rowClassName={(row) =>
                row.filterKey ? undefined : "cursor-default!"
              }
              mobileLayout="cards"
            />
          )}
        </TableScrollFrame>
        <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2 text-sm">
          {!isLoading && hidden > 0 ? (
            <button
              type="button"
              onClick={() => setShowingAll(true)}
              className="text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
            >
              Show all {rows.length} {cut.plural}
            </button>
          ) : (
            <span />
          )}
          <span className="text-muted">
            {!canOpenRows
              ? null
              : cut.value === "workspaces"
                ? "A row opens that workspace’s people."
                : cut.value === "providers"
                  ? "A row opens its requests."
                  : "A row opens its sessions."}
          </span>
        </div>
        {isTeam || spend.isLoading || spend.reads.roster !== "ready" ? null : (
          <p className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4 text-sm">
            <span className="text-muted">
              It is just you so far. Each person you add shows up here with what
              they spend and the limit they work under.
            </span>
            <Link
              to="/organization/members"
              className="shrink-0 text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
            >
              Invite people
            </Link>
          </p>
        )}
      </section>
      {editing ? (
        <SetLimitDialog
          key={`${editing.scope}:${editing.scopeId}`}
          target={editing}
          {...limitDialogCopy(editing)}
          onClose={() => setEditing(undefined)}
        />
      ) : null}
    </div>
  )
}
