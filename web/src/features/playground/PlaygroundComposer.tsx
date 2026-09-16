import type { FormEvent, KeyboardEvent, ReactNode } from "react"
import { FiArrowUp, FiSquare } from "react-icons/fi"

import type { PlaygroundTools } from "@/client"
import { Button } from "@/design-system/actions/Button"

import { ActiveToolChips } from "./ActiveToolChips"
import { ToolsMenu } from "./ToolsMenu"

export interface ComposerProps {
  modelPicker?: ReactNode
  draft: string
  onDraftChange: (value: string) => void
  onSubmit: (event: FormEvent) => void
  onKeyDown: (event: KeyboardEvent<HTMLTextAreaElement>) => void
  onStop: () => void
  canChat: boolean
  isBusy: boolean
  tools: PlaygroundTools | undefined
  isWebSearchOn: boolean
  isCodeExecutionOn: boolean
  selectedMcpIds: string[]
  toggleWebSearch: (isOn: boolean) => void
  toggleCodeExecution: (isOn: boolean) => void
  toggleMcpServer: (id: string, isOn: boolean) => void
}

/**
 * The frame groups tools, the model picker, and send with the input. A shared
 * TextArea would add another field border and label row inside that frame, so
 * a labeled native textarea supplies the editable area here.
 */
export function PlaygroundComposer({
  modelPicker,
  draft,
  onDraftChange,
  onSubmit,
  onKeyDown,
  onStop,
  canChat,
  isBusy,
  tools,
  isWebSearchOn,
  isCodeExecutionOn,
  selectedMcpIds,
  toggleWebSearch,
  toggleCodeExecution,
  toggleMcpServer,
}: ComposerProps) {
  return (
    <form onSubmit={onSubmit} className="w-full">
      <div className="flex flex-col gap-3 border border-[var(--field-border)] bg-[var(--field-background)] p-3 has-[textarea:focus-visible]:otari-focus-ring">
        <ActiveToolChips
          isWebSearchOn={isWebSearchOn}
          isCodeExecutionOn={isCodeExecutionOn}
          selectedMcpIds={selectedMcpIds}
          tools={tools}
          onToggleWebSearch={toggleWebSearch}
          onToggleCodeExecution={toggleCodeExecution}
          onToggleMcpServer={toggleMcpServer}
        />
        <textarea
          aria-label="Message"
          value={draft}
          onChange={(event) => onDraftChange(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            canChat
              ? "Write a message to test your model…"
              : "Pick a model to start"
          }
          rows={2}
          disabled={!canChat}
          // `field-sizing-content` grows the box with what is typed and caps it,
          // which is what a chat composer does; without the cap a pasted essay
          // pushes the conversation off the screen.
          className="min-h-13 max-h-48 w-full resize-none bg-transparent text-base leading-[1.625rem] text-foreground placeholder:text-subtle focus:outline-none disabled:cursor-not-allowed md:min-h-16 [field-sizing:content]"
        />
        <div className="otari-actions flex min-h-11 items-center gap-2 md:gap-3">
          <ToolsMenu
            tools={tools}
            isWebSearchOn={isWebSearchOn}
            isCodeExecutionOn={isCodeExecutionOn}
            selectedMcpIds={selectedMcpIds}
            onToggleWebSearch={toggleWebSearch}
            onToggleCodeExecution={toggleCodeExecution}
            onToggleMcpServer={toggleMcpServer}
            isDisabled={isBusy}
          />
          <div className="min-w-0 flex-1">{modelPicker}</div>
          {isBusy ? (
            <Button
              aria-label="Stop generating"
              variant="primary"
              className="w-[5.6875rem] min-w-[5.6875rem] shrink-0"
              onPress={onStop}
            >
              <FiSquare aria-hidden className="size-4" />
              Stop
            </Button>
          ) : (
            <Button
              aria-label="Send message"
              variant="primary"
              type="submit"
              className="w-[5.6875rem] min-w-[5.6875rem] shrink-0"
              isDisabled={!draft.trim() || !canChat}
            >
              <FiArrowUp aria-hidden className="size-4" />
              Send
            </Button>
          )}
        </div>
      </div>
      <p className="mt-2 hidden text-caption md:block">
        Enter to send · Shift + Enter for a new line
      </p>
    </form>
  )
}
