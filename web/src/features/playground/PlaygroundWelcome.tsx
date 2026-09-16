import { FiChevronRight } from "react-icons/fi"
import { Button } from "@/design-system/actions/Button"
import { type ComposerProps, PlaygroundComposer } from "./PlaygroundComposer"
import { CHAT_COLUMN } from "./playgroundLayout"

const EXAMPLE_PROMPTS = [
  "Compare two approaches to API retries",
  "Write a Python function with tests",
]

export function PlaygroundWelcome({
  composerProps,
  onSelectPrompt,
}: {
  composerProps: ComposerProps
  onSelectPrompt: (prompt: string) => void
}) {
  return (
    <div className={`${CHAT_COLUMN} flex flex-1 flex-col gap-8 pt-12 md:pt-24`}>
      <div className="flex flex-col gap-2">
        <h2 className="text-display-sub">What can I help with?</h2>
        <p className="text-body text-muted">
          Test a prompt with your workspace’s models.
        </p>
      </div>
      <PlaygroundComposer {...composerProps} />
      {composerProps.canChat ? (
        <div className="flex flex-col gap-2">
          <p className="text-caption">Try a starting point</p>
          <ul className="otari-actions flex flex-col divide-y divide-border border-b border-border">
            {EXAMPLE_PROMPTS.map((prompt) => (
              <li key={prompt}>
                <Button
                  className="min-h-12 w-full justify-between whitespace-normal text-left"
                  onPress={() => onSelectPrompt(prompt)}
                >
                  {prompt}
                  <FiChevronRight aria-hidden className="size-4 shrink-0" />
                </Button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <p className="mt-auto pt-8 text-caption">
        Runs use the current workspace. Conversations are stored only when you
        save.
      </p>
    </div>
  )
}
