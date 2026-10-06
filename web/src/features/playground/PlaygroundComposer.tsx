import type { FormEvent, KeyboardEvent, ReactNode } from "react"
import { FileTrigger } from "react-aria-components"
import { FiArrowUp, FiPaperclip, FiSquare } from "react-icons/fi"

import type { PlaygroundTools } from "@/client"
import { IconButton } from "@/design-system/actions/IconButton"
import { errorMessage } from "@/design-system/feedback/errorMessage"

import { ActiveToolChips } from "./ActiveToolChips"
import { PendingAttachmentChips } from "./AttachmentChips"
import type { PendingAttachment } from "./hooks/usePlayground"
import { ToolsMenu } from "./ToolsMenu"

export interface ComposerProps {
  modelPicker?: ReactNode
  hasTranscript?: boolean
  /** Set while comparing and B is unchosen, which is the only panel that can be. */
  missingModel?: "B"
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
  /** False where the deployment serves no uploads, which hides the control. */
  canAttachFiles: boolean
  attachments: PendingAttachment[]
  onAttachFiles: (files: File[]) => void
  onRemoveAttachment: (key: string) => void
  isUploading: boolean
  uploadError: unknown
}

/**
 * The frame groups tools, the model picker, and send with the input. A shared
 * TextArea would add another field border and label row inside that frame, so
 * a labeled native textarea supplies the editable area here.
 */
export function PlaygroundComposer({
  modelPicker,
  hasTranscript = false,
  missingModel,
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
  canAttachFiles,
  attachments,
  onAttachFiles,
  onRemoveAttachment,
  isUploading,
  uploadError,
}: ComposerProps) {
  // Narrow enough to fit beside the controls from `lg` up, and under the frame
  // below it, so the two nodes say the same thing at one width each.
  const hint = missingModel
    ? `Choose model ${missingModel} to send`
    : "Enter to send · Shift + Enter for a new line"

  return (
    <form onSubmit={onSubmit} className="w-full">
      <div className="otari-composer flex flex-col gap-2 px-3 pt-3 pb-2 has-[textarea:focus-visible]:otari-focus-ring">
        <ActiveToolChips
          isWebSearchOn={isWebSearchOn}
          isCodeExecutionOn={isCodeExecutionOn}
          selectedMcpIds={selectedMcpIds}
          tools={tools}
          onToggleWebSearch={toggleWebSearch}
          onToggleCodeExecution={toggleCodeExecution}
          onToggleMcpServer={toggleMcpServer}
        />
        <PendingAttachmentChips
          attachments={attachments}
          onRemove={onRemoveAttachment}
        />
        {uploadError ? (
          <p role="alert" className="px-1 text-caption text-danger">
            {errorMessage(uploadError)}
          </p>
        ) : null}
        <textarea
          aria-label="Message"
          value={draft}
          onChange={(event) => onDraftChange(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            canChat
              ? hasTranscript
                ? "Ask a follow-up…"
                : "Ask anything"
              : "Pick a model to start"
          }
          rows={2}
          disabled={!canChat}
          // `field-sizing-content` grows the box with what is typed and caps it,
          // which is what a chat composer does; without the cap a pasted essay
          // pushes the conversation off the screen.
          className="min-h-13 max-h-48 w-full resize-none px-1 bg-transparent text-base leading-[1.625rem] text-foreground placeholder:text-subtle focus:outline-none disabled:cursor-not-allowed [field-sizing:content]"
        />
        <div className="otari-actions flex min-h-8 items-center gap-1">
          {canAttachFiles ? (
            <FileTrigger
              allowsMultiple
              onSelect={(files) => {
                if (files) onAttachFiles(Array.from(files))
              }}
            >
              <IconButton
                label="Attach files"
                size="sm"
                isIconOnly
                className="shrink-0 md:min-h-8 md:min-w-8"
                isDisabled={!canChat || isBusy}
              >
                <FiPaperclip aria-hidden className="size-4" />
              </IconButton>
            </FileTrigger>
          ) : null}
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
          <div className="min-w-0">{modelPicker}</div>
          <span className="ml-auto hidden pr-2 text-caption lg:block">
            {hint}
          </span>
          {isBusy ? (
            <IconButton
              label="Stop generating"
              variant="primary"
              size="sm"
              isIconOnly
              className="ml-auto shrink-0 lg:ml-0 md:min-h-8 md:min-w-8"
              onPress={onStop}
            >
              <FiSquare aria-hidden className="size-4" />
            </IconButton>
          ) : (
            <IconButton
              label="Send message"
              variant="primary"
              type="submit"
              size="sm"
              isIconOnly
              className="ml-auto shrink-0 lg:ml-0 md:min-h-8 md:min-w-8"
              isDisabled={
                !draft.trim() || !canChat || !!missingModel || isUploading
              }
            >
              <FiArrowUp aria-hidden className="size-4" />
            </IconButton>
          )}
        </div>
      </div>
      {missingModel ? (
        <p className="mt-2 text-caption lg:hidden">{hint}</p>
      ) : null}
    </form>
  )
}
