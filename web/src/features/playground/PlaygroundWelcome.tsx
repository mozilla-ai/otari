import { Button } from "@/design-system/actions/Button"
import { type ComposerProps, PlaygroundComposer } from "./PlaygroundComposer"
import { CHAT_COLUMN } from "./playgroundLayout"

const EXAMPLE_PROMPTS = [
  "Compare two API retry approaches",
  "Write a Python function with tests",
  "Summarize an incident",
]

export function PlaygroundWelcome({
  composerProps,
  onSelectPrompt,
}: {
  composerProps: ComposerProps
  onSelectPrompt: (prompt: string) => void
}) {
  return (
    <div className={`${CHAT_COLUMN} flex flex-1 flex-col gap-7 pt-16 md:pt-36`}>
      <div className="flex flex-col gap-1.5">
        <h2 className="text-display-sub">Try a prompt.</h2>
        <p className="text-base text-muted">
          Answers, tokens, latency and cost from any model in this workspace.
        </p>
      </div>
      <PlaygroundComposer {...composerProps} />
      {composerProps.canChat ? (
        <div className="flex flex-col gap-2.5">
          <p className="text-overline">Starting points</p>
          <ul className="flex flex-wrap gap-2">
            {EXAMPLE_PROMPTS.map((prompt) => (
              <li key={prompt}>
                <Button
                  size="sm"
                  className="min-h-11 whitespace-normal text-left md:min-h-8"
                  onPress={() => onSelectPrompt(prompt)}
                >
                  {prompt}
                </Button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      <p className="mt-auto pt-8 text-caption">
        Runs use the current workspace and count toward its usage. Conversations
        are stored only when you save.
      </p>
    </div>
  )
}
