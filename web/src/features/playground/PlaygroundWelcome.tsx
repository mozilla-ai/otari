import { Button } from "@/design-system/actions/Button"

import { type ComposerProps, PlaygroundComposer } from "./PlaygroundComposer"

/**
 * Starter prompts, for somebody whose first question is "what is this for".
 *
 * Chosen to be answerable by any model rather than to show the gateway off: the
 * point of the screen is that one click produces a reply, which is also the
 * fastest way to find out a provider key is misconfigured.
 */
const EXAMPLE_PROMPTS = [
  "Explain how OAuth 2.0 works",
  "Write a regex that matches an email address",
  "Draft a polite meeting follow-up",
  "Suggest five blog post ideas about AI",
]

/**
 * The empty state before the first question: a greeting, the composer under it,
 * and one-click prompts.
 *
 * The composer is centred here and drops to the bottom once a conversation
 * starts, which is the same control in two positions rather than two controls.
 * A prompt fills the field rather than sending immediately, so a reader can edit
 * it, and so a stray click does not spend money.
 */
export function PlaygroundWelcome({
  composerProps,
  onSelectPrompt,
}: {
  composerProps: ComposerProps
  onSelectPrompt: (prompt: string) => void
}) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-6 overflow-y-auto px-4 py-8">
      <h2 className="text-center text-display-sub text-heading">
        What can I help with?
      </h2>
      <div className="flex w-full max-w-3xl flex-col gap-3">
        <PlaygroundComposer {...composerProps} />
        {composerProps.canChat ? (
          <div className="flex flex-wrap justify-center gap-2 pt-1">
            {EXAMPLE_PROMPTS.map((prompt) => (
              <Button
                key={prompt}
                size="sm"
                className="rounded-full"
                onPress={() => onSelectPrompt(prompt)}
              >
                {prompt}
              </Button>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}
