import { Link, useNavigate } from "@tanstack/react-router"
import { type ReactNode, useState } from "react"
import { FiKey, FiLayers } from "react-icons/fi"
import { RowMenu, type RowMenuAction } from "@/design-system/actions/RowMenu"
import { DataTable, type DataTableColumn } from "@/design-system/data/DataTable"
import { EmptyMessage } from "@/design-system/feedback/EmptyMessage"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { useIsMobile } from "@/design-system/hooks/useIsMobile"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { TableScrollFrame } from "@/design-system/layout/TableScrollFrame"
import { Sparkline } from "@/design-system/metrics/charts"
import { KpiCell } from "@/design-system/metrics/KpiCell"
import { KpiStrip } from "@/design-system/metrics/KpiStrip"
import { TrendChip } from "@/design-system/metrics/TrendChip"
import { Segmented } from "@/design-system/navigation/Segmented"
import { canOpenInKeys } from "@/features/home/keyAccess"
import { MemberFirstRun } from "@/features/home/MemberFirstRun"
import {
  hasActiveKey,
  isSpendRefusal,
  lowerLimitNote,
  memberLimit,
  memberLimitLabel,
  rosterRowOf,
  unseenLimitLines,
} from "@/features/home/memberHomeModel"
import { hasProviderSlot, ProviderSlot } from "@/features/home/ProviderSlot"
import { ShareCell } from "@/features/home/ShareCell"
import { useCanCreateKey } from "@/features/organization/access"
import { useCurrentOwner } from "@/features/organization/attribution"
import { RouterSavingsTile } from "@/features/routerSavings/RouterSavingsTile"
import { wholePct } from "@/features/spend/limits"
import { paidSplitLine } from "@/features/spend/paidSplit"
import {
  comparedWith,
  DEFAULT_PERIOD,
  figureLabel,
  periodFromUrl,
  sessionsSearchFor,
  windowOf,
  windowSpan,
  windowWords,
} from "@/features/spend/period"
import { SpendPeriodPicker } from "@/features/spend/SpendPeriodPicker"
import { usePageClock } from "@/features/spend/usePageClock"
import {
  filterValues,
  type ReadState,
  readState,
  type SpendRow,
  useSpend,
} from "@/features/spend/useSpend"
import { TopModelCell } from "@/features/usage/TopModelCell"
import { billedTokenTotal, cacheSums } from "@/features/usage/usageTotals"
import { useTopModels } from "@/features/usage/useTopModels"
import { useWorkspaceActivation } from "@/shared/api/activation"
import { useKeys } from "@/shared/api/apiKeys"
import { useModels } from "@/shared/api/models"
import { useOrganizationMembers } from "@/shared/api/organizations"
import { useTraceCount } from "@/shared/api/traces"
import { useRefusedUsage, useUsageSummary } from "@/shared/api/usage"
import { useWorkspaceBudgetDefaults } from "@/shared/api/workspaces"
import {
  deltaFraction,
  formatNumber,
  formatPct,
  formatRelative,
  formatTokens,
  formatUsd,
} from "@/shared/helpers/format"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSurfaces } from "@/shared/hooks/useDeployment"

type Cut = "keys" | "models" | "providers"

const CUTS: {
  value: Cut
  label: string
  noun: string
  /** What the empty table says shows up in it once there is spend. */
  each: string
}[] = [
  { value: "keys", label: "API keys", noun: "API key", each: "key you make" },
  { value: "models", label: "Models", noun: "Model", each: "model you use" },
  {
    value: "providers",
    label: "Providers",
    noun: "Provider",
    each: "provider you use",
  },
]

const URL_DEFAULTS = { period: DEFAULT_PERIOD, by: "" }

/**
 * Home for someone who uses the team rather than runs it: the owner's Home,
 * for one person. Four figures (spent, their limit, tokens, and sessions, or
 * failures where the deployment records no sessions), and where it went by
 * their keys or their models, all over their own requests in the selected
 * team. Nothing about anyone else, and no control the server would refuse them.
 */
export function MemberHome({
  workspaceId,
  workspace,
  notice,
}: {
  /** A band under the title, such as the first step on a fresh gateway. */
  notice?: ReactNode
  workspaceId: string
  workspace: string
}) {
  // One clock for the page's life, so its windows and query keys hold still.
  const [now, refreshClock] = usePageClock()
  const navigate = useNavigate()
  const { get, patch } = useUrlState(URL_DEFAULTS)
  const isMobile = useIsMobile()
  const owner = useCurrentOwner()
  const me = owner.ownerId || undefined
  const canCreateKey = useCanCreateKey()

  const period = periodFromUrl(get("period"))
  const window = windowOf(period, now)
  const cut = CUTS.find((one) => one.value === get("by")) ?? CUTS[0]

  const spend = useSpend(window, {
    workspaceId,
    userId: me,
    isEnabled: me !== undefined,
    resetEndedAt: now,
  })
  const keys = useKeys(workspaceId)
  const members = useOrganizationMembers()
  const defaults = useWorkspaceBudgetDefaults(workspaceId)
  // A deployment that records no sessions serves no route to count them, and
  // has none to open.
  const hasSurface = useSurfaces()
  const hasSessions = hasSurface("traces")
  // Sessions cannot filter on a provider, so a provider's row opens Requests.
  const canOpenRows =
    cut.value === "providers" ? hasSurface("usage") : hasSessions
  const sessions = useTraceCount(
    {
      workspaceId,
      start: window.start,
      end: window.end || undefined,
      q: "",
      userIds: me ? [me] : undefined,
    },
    hasSessions,
  )
  const ownKeys = new Map((keys.data ?? []).map((key) => [key.id, key]))
  // Rows the window spent on: keys with nothing spent are on Keys.
  const rows: SpendRow[] = (
    cut.value === "keys"
      ? spend.keys
      : cut.value === "providers"
        ? spend.providers
        : spend.models
  )
    .filter((row) => row.requests > 0 || row.cost > 0)
    .map((row) => {
      if (cut.value !== "keys") return row
      const key = row.filterKey ? ownKeys.get(row.filterKey) : undefined
      if (!key) return row
      return {
        ...row,
        detail: key.last_used_at
          ? `Used ${formatRelative(key.last_used_at)}`
          : "Never used",
      }
    })
  const topModels = useTopModels({
    rows,
    cut: cut.value,
    window,
    workspaceId,
    userId: me,
  })
  const hasMarkSlot = cut.value === "providers" && hasProviderSlot(rows)

  const total = spend.totals?.cost ?? 0
  const requestCount = spend.totals?.request_count ?? 0
  const delta = deltaFraction(total, spend.previousCost)
  const cacheTotals = cacheSums(spend.series)
  // The count is the caller's once their id is in the request. Until one has
  // answered for it, an answer standing in from before is the workspace's.
  const [hasOwnCount, setHasOwnCount] = useState(false)
  if (!hasOwnCount && me && sessions.data && !sessions.isPlaceholderData) {
    setHasOwnCount(true)
  }
  const sessionCount =
    me && (hasOwnCount || !sessions.isPlaceholderData)
      ? sessions.data?.count
      : undefined
  const sessionsFailed =
    owner.isFailed ||
    (me !== undefined && sessions.isError && sessions.data === undefined)
  // What the spend figures stand on. A caller with no billing id has no
  // spend read for them at all.
  const usage: ReadState | "none" = owner.isFailed
    ? "failed"
    : owner.isPending
      ? "loading"
      : me
        ? spend.reads.usage
        : "none"
  const unreadLine =
    usage === "failed"
      ? "Could not load"
      : usage === "loading"
        ? "Loading"
        : undefined
  const words = windowWords(window)
  // A cell with no graphic still holds the row, or its figure drops off the
  // baseline of the cells beside it that have one.
  const hasGraphicRow = spend.series.length > 1 || Boolean(sessionCount)
  const slot = (graphic: ReactNode) =>
    graphic ?? (hasGraphicRow ? <span aria-hidden /> : undefined)
  const isFirstRun =
    !keys.isLoading && !hasActiveKey(keys.data ?? [], now) && sessionCount === 0
  // While the setup guide is offered to this member it is their first run, so
  // the band below stays away: the same reads the guide opens on (the layout
  // mounts it over Home), a model to call and the server's offer.
  const asksGuide = isFirstRun && canCreateKey
  const activation = useWorkspaceActivation(workspaceId, asksGuide, {
    poll: false,
  })
  const models = useModels(asksGuide, workspaceId)
  const isGuideOffered =
    activation.data?.experience_eligible === true &&
    (models.data?.data.length ?? 0) > 0
  // Until both answer, the band could flash in and out under the guide.
  const isGuideSettled =
    !asksGuide || (!activation.isPending && !models.isPending)
  const mySessions = {
    ...sessionsSearchFor(window),
    ...(me ? { user_id: me } : {}),
  }

  const limitReads = [readState(members), readState(defaults)]
  const limitRead: ReadState =
    owner.isFailed || limitReads.includes("failed")
      ? "failed"
      : limitReads.includes("loading")
        ? "loading"
        : "ready"
  const limit =
    limitRead === "ready"
      ? memberLimit({
          member: rosterRowOf(members.data ?? [], me),
          workspaceId,
          defaults: defaults.data ?? [],
          now,
        })
      : undefined
  // The note weighs the per-person figure against its own period, not the
  // window shown. Read from the refused requests themselves: the summary
  // counts every 403 alike, and only a spend refusal says the limit is why.
  // An unseen limit is only said to have refused the member over the window.
  const periodStart = limit?.kind === "default" ? limit.periodStart : ""
  const refusedSince = limit?.kind === "unseen" ? window.start : periodStart
  const refused = useRefusedUsage(
    { userIds: me ? [me] : undefined, workspaceId },
    refusedSince,
    100,
    me !== undefined && refusedSince !== "",
    limit?.kind === "unseen" ? window.end : "",
  )
  const spendRefusals =
    limit?.kind === "default"
      ? (refused.data ?? []).filter((row) => isSpendRefusal(row.error_message))
          .length
      : 0
  const periodSpend = useUsageSummary(
    {
      workspace_id: workspaceId,
      user_id: me ? [me] : undefined,
      start_date: periodStart,
    },
    "day",
    [],
    me !== undefined && spendRefusals > 0,
    "organization",
  )

  const unseen = unseenLimitLines({
    limit,
    // A caller with no billing id has no refused requests to read.
    refusals:
      me === undefined || refused.isError
        ? []
        : refused.data?.map((row) => row.error_message),
    words,
    workspace,
  })
  // An unseen limit reads as loading until its lines are known.
  const shown = limit?.kind === "unseen" && !unseen ? undefined : limit

  const limitWords =
    shown === undefined
      ? limitRead === "failed"
        ? "Could not load"
        : "Loading"
      : shown.kind === "own"
        ? isMobile
          ? `in ${workspace}, set by an owner or admin`
          : "Set by an owner or admin"
        : shown.kind === "default"
          ? `${shown.per ? `${shown.per}, ` : ""}for everyone in ${workspace}`
          : (unseen?.subline ?? "")

  const limitNote =
    (periodSpend.data && !periodSpend.isPlaceholderData
      ? lowerLimitNote({
          limit,
          since: periodSpend.data.start_date,
          spent: periodSpend.data.totals.cost,
          spendRefusals,
        })
      : undefined) ?? unseen?.note

  const open = (row: SpendRow) => {
    if (!row.filterKey) return
    void navigate({
      to: cut.value === "providers" ? "/requests" : "/sessions",
      search: {
        ...mySessions,
        [cut.value === "keys"
          ? "api_key_id"
          : cut.value === "providers"
            ? "provider"
            : "model"]: filterValues(row),
      },
    })
  }

  const menuFor = (row: SpendRow): RowMenuAction[] => {
    const actions: RowMenuAction[] = []
    if (row.filterKey && canOpenRows) {
      actions.push({
        id: "open",
        label:
          cut.value === "providers" ? "See its requests" : "See its sessions",
        hint: `What ran ${words}`,
        icon: FiLayers,
        run: () => open(row),
      })
    }
    if (cut.value === "keys" && canOpenInKeys(row.filterKey, keys.data ?? [])) {
      actions.push({
        id: "keys",
        label: "Open in API keys",
        hint: canCreateKey
          ? "Disable, edit or regenerate it"
          : "See its details",
        icon: FiKey,
        run: () => void navigate({ to: "/keys" }),
      })
    }
    return actions
  }

  const columns: DataTableColumn<SpendRow>[] = [
    {
      id: "name",
      header: cut.noun,
      isRowHeader: true,
      cell: (row) =>
        cut.value === "models" ? (
          <span
            className="line-clamp-2 text-mono-caption text-foreground [overflow-wrap:anywhere] md:max-w-[13rem] xl:max-w-[24rem]"
            title={row.spellings?.join("\n")}
          >
            {row.name}
          </span>
        ) : (
          <span className="flex min-w-0 items-center gap-3 md:max-w-[13rem]">
            {hasMarkSlot ? <ProviderSlot row={row} /> : null}
            <span className="flex min-w-0 flex-col">
              <span className="truncate text-foreground">{row.name}</span>
              {row.detail ? (
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
          {formatUsd(row.cost)}
          {isMobile && total > 0 ? (
            <span className="text-muted">
              {" · "}
              {wholePct(row.cost / total)}
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
          {formatTokens(row.tokens)}
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
              <ShareCell share={total > 0 ? row.cost / total : undefined} />
            ),
          },
        ] satisfies DataTableColumn<SpendRow>[])),
    {
      id: "actions",
      header: <span className="sr-only">Actions</span>,
      align: "end",
      cell: (row) => (
        <RowMenu label={`Actions for ${row.name}`} actions={menuFor(row)} />
      ),
    },
  ]

  // Only a window still under way can be spent in "yet".
  const nothingYet = `Nothing spent ${words}${
    period === "today" ? " yet" : ""
  }.`

  return (
    <div className="flex flex-col gap-6">
      <PageIntro
        title="Home"
        action={
          <SpendPeriodPicker
            value={period}
            onChange={(next) => {
              refreshClock()
              patch({ period: next })
            }}
          />
        }
      />
      {notice}
      <ErrorBanner error={spend.error} />
      {/* Needs attention (`MemberAttentionBand`) is held back until it reads
          each row's scope right; only the first-run band shows here. */}
      {isFirstRun && isGuideSettled && !isGuideOffered ? (
        <MemberFirstRun workspace={workspace} canCreateKey={canCreateKey} />
      ) : null}
      <KpiStrip isEmpty={false} columns={4}>
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
            paidSplitLine(spend.totals) ||
            (delta !== null
              ? undefined
              : !spend.totals
                ? unreadLine
                : requestCount > 0
                  ? `${formatNumber(requestCount)} requests`
                  : "No requests yet")
          }
          graphic={slot(
            spend.series.length > 1 ? (
              <Sparkline
                values={spend.series.map((point) => point.cost)}
                ariaLabel={`Your spend per ${window.bucket} over the window`}
                height={40}
              />
            ) : undefined,
          )}
        />
        <KpiCell
          label={memberLimitLabel(limit, workspace)}
          value={
            shown === undefined
              ? "—"
              : shown.kind === "unseen"
                ? "Not shown"
                : formatUsd(shown.amount)
          }
          subline={isMobile ? undefined : limitWords}
          // A half-width cell would cut the subline short, and it is where the
          // workspace is named now that the label is not: on a phone the
          // words and the hint wrap together, under the figure, as far apart
          // as the desktop cell sets them (18px between the two lines, there
          // by the graphic row centering the hint).
          delta={
            isMobile && limitWords ? (
              <span className="flex flex-col gap-4.5 whitespace-normal">
                <span>{limitWords}</span>
                {limitNote ? <span>{limitNote}</span> : null}
              </span>
            ) : undefined
          }
          // On its own line, wrapping: the subline truncates, and this is the
          // sentence that must be read whole.
          graphic={
            limitNote && !isMobile ? (
              <span className="whitespace-normal text-caption">
                {limitNote}
              </span>
            ) : (
              slot(undefined)
            )
          }
        />
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
                ? // The split took the request count off the Spent tile.
                  `${paidSplitLine(spend.totals) ? `${formatNumber(requestCount)} requests · ` : ""}${formatPct(cacheTotals.read / cacheTotals.input)} of input from cache`
                : requestCount > 0
                  ? `${formatNumber(requestCount)} requests`
                  : "No requests yet"
          }
          graphic={slot(
            spend.series.length > 1 ? (
              <Sparkline
                values={spend.series.map(
                  (point) => point.input_tokens + point.output_tokens,
                )}
                ariaLabel={`Your billed tokens per ${window.bucket} over the window`}
                height={40}
              />
            ) : undefined,
          )}
        />
        {hasSessions ? (
          <KpiCell
            label="Your sessions"
            value={
              sessionCount === undefined ? "—" : formatNumber(sessionCount)
            }
            subline={
              sessionsFailed
                ? "Could not load"
                : `${windowSpan(window)} in ${workspace}`
            }
            graphic={slot(
              sessionCount ? (
                <Link
                  to="/sessions"
                  search={mySessions}
                  className="self-start text-xs text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"
                >
                  See your sessions
                </Link>
              ) : undefined,
            )}
          />
        ) : (
          // The owner's Home has this cell, so the strip keeps its four.
          <KpiCell
            label="Failures"
            value={
              spend.totals
                ? formatPct(
                    requestCount > 0
                      ? spend.totals.error_count / requestCount
                      : 0,
                  )
                : "—"
            }
            subline={
              spend.totals
                ? `${formatNumber(spend.totals.error_count)} failed ${spend.totals.error_count === 1 ? "request" : "requests"}`
                : unreadLine
            }
            graphic={slot(undefined)}
          />
        )}
      </KpiStrip>
      <RouterSavingsTile
        window={window}
        workspaceId={workspaceId}
        userId={me}
        isEnabled={me !== undefined}
      />
      <section aria-label="Where it went" className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
          <h2 className="text-title">Where it went</h2>
          <Segmented
            label="Break down by"
            size="md"
            options={CUTS}
            value={cut.value}
            onChange={(next) => patch({ by: next })}
            isFullWidth={isMobile}
          />
        </div>
        <TableScrollFrame className="otari-breakdown otari-spend-table flex flex-col gap-2">
          <DataTable
            key={cut.value}
            ariaLabel={`Your spend by ${cut.label.toLowerCase()}`}
            columns={columns}
            rows={rows}
            getRowKey={(row) => row.id}
            isLoading={spend.isLoading}
            emptyContent={
              <EmptyMessage>
                {usage === "failed"
                  ? "Could not load"
                  : canCreateKey
                    ? `${nothingYet} Each ${cut.each} shows up here with what it spent.`
                    : nothingYet}
              </EmptyMessage>
            }
            onRowAction={
              canOpenRows
                ? (key) => {
                    const row = rows.find((one) => one.id === key)
                    if (row) open(row)
                  }
                : undefined
            }
            rowClassName={(row) =>
              row.filterKey ? undefined : "cursor-default!"
            }
            mobileLayout="cards"
          />
        </TableScrollFrame>
        {rows.length > 0 &&
        (canOpenRows || (cut.value === "keys" && canCreateKey)) ? (
          <p className="text-sm text-muted">
            {canOpenRows
              ? cut.value === "providers"
                ? "A row opens its requests."
                : "A row opens its sessions."
              : null}
            {cut.value === "keys" && canCreateKey ? (
              <>
                {canOpenRows ? " " : null}
                Keys that spent nothing {words} are on{" "}
                <Link to="/keys" className="text-link underline">
                  API keys
                </Link>
                .
              </>
            ) : null}
          </p>
        ) : null}
      </section>
    </div>
  )
}
