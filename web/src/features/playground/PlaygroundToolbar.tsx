import {
  FiBarChart2,
  FiBookOpen,
  FiColumns,
  FiEdit,
  FiMinimize2,
  FiSave,
} from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { Tooltip } from "@/design-system/overlays/Tooltip"

import type { PlaygroundModel } from "./helpers/playgroundModels"
import { ModelSelect } from "./ModelSelect"

/**
 * The row above the conversation: the model picker, and the actions over the
 * whole page.
 *
 * The picker is here in single view and moves above each column while
 * comparing, which is not a layout preference: two pickers in one row cannot
 * say which column each belongs to, and a comparison whose picker-to-panel
 * mapping is ambiguous is a comparison nobody can trust.
 *
 * Two actions appear only when they have something to act on, rather than
 * sitting disabled. A history control with no history explains nothing by being
 * dim, and both are counts the page already knows.
 */
export function PlaygroundToolbar({
  isComparing,
  modelA,
  models,
  pinnedKeys,
  onTogglePin,
  onModelAChange,
  conversationCount,
  onOpenHistory,
  onSaveConversation,
  isSaveDisabled,
  isSavePending,
  comparisonCount,
  onOpenComparisonHistory,
  onToggleCompare,
  onNewChat,
  isNewChatDisabled,
}: {
  isComparing: boolean
  modelA: string
  models: readonly PlaygroundModel[]
  pinnedKeys: readonly string[]
  onTogglePin: (key: string) => void
  onModelAChange: (key: string) => void
  conversationCount: number
  onOpenHistory: () => void
  onSaveConversation: () => void
  isSaveDisabled: boolean
  isSavePending: boolean
  comparisonCount: number
  onOpenComparisonHistory: () => void
  onToggleCompare: () => void
  onNewChat: () => void
  isNewChatDisabled: boolean
}) {
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-3 border-border border-b pb-4">
      {isComparing ? null : (
        <ModelSelect
          label="Model"
          value={modelA}
          models={models}
          pinnedKeys={pinnedKeys}
          onTogglePin={onTogglePin}
          onChange={onModelAChange}
          className="w-72 max-w-full"
        />
      )}

      <div className="flex-1" />

      <div className="flex items-center gap-1">
        {/* Not gated on retention consent: withdrawing it blocks new saves, so
            what was already saved stays listable and deletable. */}
        {conversationCount > 0 ? (
          <Tooltip content="Conversation history">
            <IconButton label="Conversation history" onPress={onOpenHistory}>
              <FiBookOpen aria-hidden className="size-4" />
            </IconButton>
          </Tooltip>
        ) : null}

        {comparisonCount > 0 ? (
          <Tooltip content="Saved comparisons">
            <IconButton
              label="Saved comparisons"
              onPress={onOpenComparisonHistory}
            >
              <FiBarChart2 aria-hidden className="size-4" />
            </IconButton>
          </Tooltip>
        ) : null}

        {/* Single view only. Comparing has two models and so no single "the
            conversation" to save, and a comparison is recorded through an
            explicit rating rather than a Save. */}
        {isComparing ? null : (
          <Tooltip content="Save conversation">
            <IconButton
              label="Save conversation"
              onPress={onSaveConversation}
              isDisabled={isSaveDisabled}
              isPending={isSavePending}
            >
              <FiSave aria-hidden className="size-4" />
            </IconButton>
          </Tooltip>
        )}

        <Tooltip content={isComparing ? "Single view" : "Compare models"}>
          <IconButton
            label={isComparing ? "Switch to single view" : "Compare two models"}
            onPress={onToggleCompare}
          >
            {isComparing ? (
              <FiMinimize2 aria-hidden className="size-4" />
            ) : (
              <FiColumns aria-hidden className="size-4" />
            )}
          </IconButton>
        </Tooltip>

        <Tooltip content="New chat">
          <IconButton
            label="Start a new chat"
            onPress={onNewChat}
            isDisabled={isNewChatDisabled}
          >
            <FiEdit aria-hidden className="size-4" />
          </IconButton>
        </Tooltip>
      </div>
    </div>
  )
}
