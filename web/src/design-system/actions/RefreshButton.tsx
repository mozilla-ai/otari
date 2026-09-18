import { useEffect, useState } from "react"
import { FiRotateCw } from "react-icons/fi"
import { IconButton } from "@/design-system/actions/IconButton"
import { formatRelative } from "@/design-system/helpers/format"

// A slowly ticking wall clock, only to keep a relative "updated Ns ago" label
// current between renders. This is a display timer, not the prohibited data
// polling (which belongs on a TanStack Query `refetchInterval`): it fetches
// nothing. Paused while the tab is hidden so a backgrounded dashboard is idle.
function useDisplayClock(intervalMs: number): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | undefined
    const start = () => {
      if (timer === undefined) {
        timer = setInterval(() => setNow(Date.now()), intervalMs)
      }
    }
    const stop = () => {
      if (timer !== undefined) {
        clearInterval(timer)
        timer = undefined
      }
    }
    const sync = () => {
      setNow(Date.now())
      if (document.visibilityState === "visible") start()
      else stop()
    }
    sync()
    document.addEventListener("visibilitychange", sync)
    return () => {
      stop()
      document.removeEventListener("visibilitychange", sync)
    }
  }, [intervalMs])
  return now
}

// A refresh control paired with a "last updated" timestamp, so an operator can
// tell stale numbers from fresh ones. The icon spins while a refetch is in
// flight. `updatedAt` is a TanStack Query `dataUpdatedAt` (ms epoch; 0 before
// the first successful load, which reads as "never" and is hidden).
export function RefreshButton({
  onRefresh,
  isFetching = false,
  updatedAt,
  label = "Refresh",
}: {
  onRefresh: () => void
  isFetching?: boolean
  updatedAt?: number
  label?: string
}) {
  const now = useDisplayClock(15_000)
  const freshness = updatedAt
    ? formatRelative(new Date(updatedAt).toISOString(), now)
    : null
  return (
    <span className="inline-flex items-center gap-2">
      {freshness ? (
        <span className="text-caption">Updated {freshness}</span>
      ) : null}
      <IconButton
        label={label}
        variant="ghost"
        size="sm"
        isIconOnly
        // `size="sm"` is 32px, under the 44px floor. The box grows below `md`
        // and settles back to the button's own size where a pointer is doing
        // the pressing, so the desktop density of the four toolbars this sits
        // in is unchanged.
        className="md:min-h-8 md:min-w-8"
        isDisabled={isFetching}
        onPress={onRefresh}
      >
        <FiRotateCw
          aria-hidden="true"
          className={`h-4 w-4 ${isFetching ? "animate-spin" : ""}`}
        />
      </IconButton>
    </span>
  )
}
