import { Segmented } from "@/design-system/navigation/Segmented"
import { ActivityPage } from "@/features/activity/ActivityPage"
import { TracesPage } from "@/features/traces/TracesPage"
import { useUrlState } from "@/shared/helpers/urlState"
import { useSurfaces } from "@/shared/hooks/useDeployment"

const VIEWS = [
  { value: "sessions", label: "Sessions" },
  { value: "requests", label: "Requests" },
]

// Activity opens on agent sessions where this deployment records traces, with the
// per-request log one switch away. Where it records none, the log is the page.
export function ActivityHub() {
  const hosts = useSurfaces()
  const url = useUrlState({ view: "" })
  if (!hosts("traces")) {
    return <ActivityPage />
  }
  // The view is named in the URL, never inferred from the filters it carries, so
  // clearing the request log's filters leaves the request log on screen. A link
  // into the log says so (`view=requests`).
  const view = url.get("view") === "requests" ? "requests" : "sessions"
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
