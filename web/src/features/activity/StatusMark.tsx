import { Dot } from "@/design-system/indicators/Dot"
import { describeStatus } from "./activityModel"

/**
 * Status as a square dot and a word, failure-forward.
 *
 * `error` is the only one that colors its text, which is what keeps a failure
 * findable in a scan of fifty rows. `absorbed` is deliberately neutral rather
 * than caution: a routing policy recovered from it, so the request was served,
 * and painting it as a warning made a working gateway look like a failing one.
 * It reads as the third thing it is, on the subtle rung in both channels.
 */
export function StatusMark({ status }: { status: string }) {
  const { dot, ink } =
    status === "error"
      ? { dot: "bg-danger", ink: "text-danger" }
      : status === "absorbed"
        ? { dot: "bg-text-subtle", ink: "text-subtle" }
        : { dot: "bg-success", ink: "text-muted" }
  return (
    <span className={`flex items-center gap-2 text-mono-caption ${ink}`}>
      <Dot className={dot} />
      {describeStatus(status)}
    </span>
  )
}
