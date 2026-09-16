import {
  FiBarChart2,
  FiClock,
  FiColumns,
  FiMinimize2,
  FiSave,
} from "react-icons/fi"

import { Button } from "@/design-system/actions/Button"
import { Section } from "@/design-system/layout/Section"
import { Toolbar } from "@/design-system/layout/Toolbar"

export function PlaygroundToolbar({
  isComparing,
  onOpenHistory,
  onSaveConversation,
  isSaveDisabled,
  isSavePending,
  onOpenComparisonHistory,
  onToggleCompare,
  hasTranscript,
}: {
  isComparing: boolean
  onOpenHistory: () => void
  onSaveConversation: () => void
  isSaveDisabled: boolean
  isSavePending: boolean
  onOpenComparisonHistory: () => void
  onToggleCompare: () => void
  hasTranscript: boolean
}) {
  return (
    <Section className="border-y border-border">
      <Toolbar className="min-h-15 py-2">
        <Button
          aria-label={isComparing ? "Single view" : "Compare models"}
          aria-pressed={isComparing}
          onPress={onToggleCompare}
        >
          {isComparing ? (
            <FiMinimize2 aria-hidden className="size-4" />
          ) : (
            <FiColumns aria-hidden className="size-4" />
          )}
          <span className="hidden md:inline">
            {isComparing ? "Single view" : "Compare models"}
          </span>
          <span className="md:hidden">
            {isComparing ? "Single" : "Compare"}
          </span>
        </Button>
        <Button
          aria-label="Conversation history"
          className="min-h-11 min-w-11"
          onPress={onOpenHistory}
        >
          <FiClock aria-hidden className="size-4" />
          <span className="hidden lg:inline">Conversation history</span>
        </Button>
        <Button
          aria-label="Comparison history"
          className="min-h-11 min-w-11"
          onPress={onOpenComparisonHistory}
        >
          <FiBarChart2 aria-hidden className="size-4" />
          <span className="hidden lg:inline">Comparison history</span>
        </Button>
        {!isComparing && hasTranscript ? (
          <Button
            aria-label="Save conversation"
            className="ml-auto min-h-11 min-w-11"
            onPress={onSaveConversation}
            isDisabled={isSaveDisabled}
            isPending={isSavePending}
          >
            <FiSave aria-hidden className="size-4" />
            <span className="hidden md:inline">Save conversation</span>
          </Button>
        ) : null}
      </Toolbar>
    </Section>
  )
}
