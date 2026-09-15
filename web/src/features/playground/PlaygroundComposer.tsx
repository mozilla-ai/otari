import type { FormEvent, KeyboardEvent } from "react"
import { FiSend, FiSquare } from "react-icons/fi"

import type { PlaygroundTools } from "@/client"
import { IconButton } from "@/design-system/actions/IconButton"

import { ActiveToolChips } from "./ActiveToolChips"
import { ToolsMenu } from "./ToolsMenu"

export interface ComposerProps {
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
 * The composer: attached-tool chips, the message field, the tools menu, and
 * send.
 *
 * One component used in two places, centred under the greeting before the first
 * question and pinned to the bottom afterwards, which is why it takes no
 * position of its own: the page places it.
 *
 * The field is a bare `<textarea>` rather than the shared `TextArea`, and this
 * is the exception the components rule allows for. That primitive is a labelled
 * form field: it renders its own label, its own bordered box, and reserves a
 * row for a validation message. Here the *frame* is the control, holding the
 * chips above and the tool and send controls below, so the primitive's box would
 * be a second border inside it and its label row would sit above the chips. The
 * field carries an `aria-label` and the frame draws the focus ring with
 * `focus-within`.
 *
 * While a reply is streaming, send becomes Stop. The same slot rather than a
 * second control beside it: there is exactly one thing to do to a request in
 * flight, and a disabled send next to a stop is two controls for one decision.
 */
export function PlaygroundComposer({
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
      <div className="flex flex-col gap-2 rounded-3xl border border-border bg-surface p-2.5 transition-colors focus-within:border-accent">
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
          placeholder={canChat ? "Ask anything" : "Pick a model to start"}
          rows={1}
          disabled={!canChat}
          // `field-sizing-content` grows the box with what is typed and caps it,
          // which is what a chat composer does; without the cap a pasted essay
          // pushes the conversation off the screen.
          className="max-h-48 w-full resize-none bg-transparent px-2 py-1.5 text-sm text-foreground placeholder:text-subtle focus:outline-none disabled:cursor-not-allowed [field-sizing:content]"
        />
        <div className="flex items-center justify-between gap-2 px-1">
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
          {isBusy ? (
            <IconButton
              label="Stop generating"
              variant="primary"
              className="rounded-full"
              onPress={onStop}
            >
              <FiSquare aria-hidden className="size-4" />
            </IconButton>
          ) : (
            <IconButton
              label="Send message"
              variant="primary"
              type="submit"
              className="rounded-full"
              isDisabled={!draft.trim() || !canChat}
            >
              <FiSend aria-hidden className="size-4" />
            </IconButton>
          )}
        </div>
      </div>
    </form>
  )
}
