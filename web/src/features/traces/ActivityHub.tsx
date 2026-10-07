import { Segmented } from "@/design-system/navigation/Segmented"
import { ActivityPage } from "@/features/activity/ActivityPage"
import { TracesPage } from "@/features/traces/TracesPage"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSurfaces } from "@/shared/hooks/useDeployment"

const VIEWS = [
  { value: "sessions", label: "Sessions" },
  { value: "requests", label: "Requests" },
]

// What a link into the request log carries (Usage's drill-down, a pricing warning).
const REQUEST_LOG_FILTERS = [
  "status",
  "model",
  "user_id",
  "api_key_id",
  "priced",
  "source",
  "source_label",
  "endpoint",
  "provider",
  "tool",
  "start_date",
  "end_date",
] as const

// Activity opens on agent sessions where this deployment records traces, with the
// per-request log one switch away. Where it records none, the log is the page.
export function ActivityHub() {
  const hosts = useSurfaces()
  const url = useUrlState({
    view: "",
    status: "",
    model: "",
    user_id: "",
    api_key_id: "",
    priced: "",
    source: "",
    source_label: "",
    endpoint: "",
    provider: "",
    tool: "",
    start_date: "",
    end_date: "",
  })
  if (!hosts("traces")) {
    return <ActivityPage />
  }
  // An explicit choice wins; otherwise a link that carries request-log filters
  // means the log, and anything else opens on sessions.
  const chosen = url.get("view")
  const view =
    chosen === "requests" || chosen === "sessions"
      ? chosen
      : REQUEST_LOG_FILTERS.some((key) => url.get(key) !== "")
        ? "requests"
        : "sessions"
  const viewSwitch = (
    <Segmented
      label="Activity view"
      options={VIEWS}
      value={view}
      onChange={(next) => url.patch({ view: next })}
      size="sm"
    />
  )
  return view === "requests" ? (
    <ActivityPage viewSwitch={viewSwitch} />
  ) : (
    <TracesPage viewSwitch={viewSwitch} />
  )
}
