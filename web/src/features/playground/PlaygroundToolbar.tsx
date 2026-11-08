import type { ReactNode } from "react"
import { FiPlus } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { Segmented } from "@/design-system/navigation/Segmented"

export function PlaygroundToolbar({
  isComparing,
  onToggleCompare,
  onNewChat,
  hasTranscript,
  history,
  files,
  capture,
}: {
  isComparing: boolean
  onToggleCompare: () => void
  onNewChat: () => void
  hasTranscript: boolean
  history: ReactNode
  /** The uploaded-files control, where the deployment serves uploads. */
  files?: ReactNode
  /** The prompt-capture switch, where the deployment can keep trace content. */
  capture?: ReactNode
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <Segmented
        size="md"
        label="Playground mode"
        value={isComparing ? "compare" : "single"}
        onChange={(next) => {
          if ((next === "compare") !== isComparing) onToggleCompare()
        }}
        options={[
          { value: "single", label: "Single" },
          { value: "compare", label: "Compare" },
        ]}
      />
      {history}
      {files}
      {capture}
      <Button
        onPress={onNewChat}
        isDisabled={!hasTranscript}
        className="min-h-11 md:min-h-9"
      >
        <FiPlus aria-hidden className="size-4" /> New chat
      </Button>
    </div>
  )
}
