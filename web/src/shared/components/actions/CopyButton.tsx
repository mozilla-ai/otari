import { Button, Tooltip } from "@heroui/react"
import { useEffect, useRef, useState } from "react"
import { FiCopy } from "react-icons/fi"
import { copyToClipboard } from "@/shared/helpers/clipboard"

// A compact copy control for an identifier an operator has to paste elsewhere (a
// model id, an alias target). Table rows own click-drag for selection, so the
// text in a cell cannot be highlighted by hand (issue #478); this is how it gets
// out. copyToClipboard covers the plain-HTTP origins this dashboard is routinely
// served from, where the async Clipboard API does not exist; if even the legacy
// path fails, this says so rather than claiming a copy it did not make (the same
// rule as the Keys page's CopyField). Unlike that one, the value here is not a
// form field this can select for the operator, so the failure message asks them
// to select it rather than implying something already is.
// The acknowledgement is a tooltip over the icon that was pressed, so the answer
// appears where the operator is looking in a column of identical buttons. It is
// controlled (never hover-opened) because it reports an event, not a hint, and it
// renders in an overlay so it is not clipped by the table's scroll container and
// does not reflow the row it reports on.
export function CopyButton({ value, label }: { value: string; label: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle")
  const resetTimer = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  )

  useEffect(() => () => clearTimeout(resetTimer.current), [])

  const copy = async () => {
    const copied = await copyToClipboard(value)
    setState(copied ? "copied" : "failed")
    clearTimeout(resetTimer.current)
    // A failure has something to read and act on, so it lingers longer.
    resetTimer.current = setTimeout(
      () => setState("idle"),
      copied ? 1_500 : 5_000,
    )
  }

  return (
    <Tooltip.Root isOpen={state !== "idle"}>
      <Button
        size="sm"
        variant="ghost"
        isIconOnly
        aria-label={`Copy ${label}`}
        onPress={copy}
      >
        <FiCopy aria-hidden="true" className="h-3.5 w-3.5" />
      </Button>
      <Tooltip.Content placement="top" showArrow>
        {state === "failed"
          ? "Copy blocked, select the value and press Ctrl/Cmd-C"
          : "Copied!"}
      </Tooltip.Content>
      {/* A tooltip opened by a press rather than by focus is not announced, and
          the outcome is the whole point of the press. `CopyField` says its own
          the same way. */}
      <span aria-live="polite" className="sr-only">
        {state === "copied"
          ? `Copied ${label} to clipboard.`
          : state === "failed"
            ? `Could not copy ${label}. Select the value and press Ctrl/Cmd-C.`
            : ""}
      </span>
    </Tooltip.Root>
  )
}
