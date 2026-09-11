import { Link } from "@tanstack/react-router"
import { useState } from "react"

import { Button } from "@/design-system/actions/Button"
import { CONCEALED_SECRET, CopyField } from "@/design-system/actions/CopyField"
import { CodeBlock } from "@/design-system/content/CodeBlock"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { ListeningPanel } from "@/features/onboarding/ListeningPanel"
import type { SetupFailure } from "@/features/onboarding/setupFailureCopy"
import {
  buildSetupSnippets,
  carriesKey,
  DEFAULT_SETUP_TAB,
  SETUP_TABS,
  type SetupSnippetId,
} from "@/features/onboarding/setupSnippets"
import { MissingGatewayAddressNotice } from "@/shared/components/access/MissingGatewayAddressNotice"
import {
  SNIPPET_KEY_ENV_VAR,
  SNIPPET_MODEL_PLACEHOLDER,
} from "@/shared/helpers/requestSnippets"

/**
 * The first-run sheet: one screen, one job, which is to get a request into a
 * new workspace.
 *
 * **A sheet rather than a panel on the page**, which is what the platform's
 * flow was and what this replaces. Nothing else on a dashboard with no traffic
 * in it is worth reading, so the guide is allowed to be the screen; it offers a
 * way out in the footer and never comes back once taken.
 *
 * **Divided rather than stacked.** Each part is a `DialogSection`, so the key,
 * the example and the wait are separated by hairlines that run the width of the
 * frame. A column of gaps would have read as three cards floating in a sheet,
 * which is the thing this design system does not do.
 *
 * Presentational, and controlled by `SetupGuide` above it: every piece of state
 * here is either the tab somebody picked or whether the key is on screen.
 */
export function SetupSheet({
  workspaceName,
  apiKey,
  baseUrl,
  model,
  failure,
  attemptAt,
  isChecking,
  checkFailed,
  keyError,
  skipError,
  isSkipping,
  onCheckNow,
  onSkip,
  onDismiss,
}: {
  workspaceName?: string
  /** The issued key's plaintext, or undefined while it is being minted. */
  apiKey?: string
  /** Where a request belongs, or undefined when the deployment names none. */
  baseUrl?: string
  /** The first model the gateway can serve, when it can serve one. */
  model?: string
  failure?: SetupFailure
  attemptAt?: string
  isChecking: boolean
  checkFailed: boolean
  keyError: unknown
  skipError: unknown
  isSkipping: boolean
  onCheckNow: () => void
  onSkip: () => void
  /** Closes the sheet without retiring the guide, so it is offered again. */
  onDismiss: () => void
}) {
  const [tab, setTab] = useState<SetupSnippetId>(DEFAULT_SETUP_TAB)
  // One reveal for the whole sheet: the key field and the example built around
  // it hide the same secret, so revealing one and not the other would be a
  // distinction with nothing behind it.
  const [isRevealed, setIsRevealed] = useState(false)

  const instruction = SETUP_TABS.find(({ id }) => id === tab)?.instruction ?? ""
  // Built from the stand-in whenever the key is not on screen, which includes
  // the moment before it exists: what is rendered needs no key, so the examples
  // are here from the first frame rather than appearing under the operator once
  // the mint lands. Only the copy waits for the real thing.
  const shown =
    baseUrl === undefined
      ? undefined
      : buildSetupSnippets({
          baseUrl,
          apiKey:
            isRevealed && apiKey !== undefined ? apiKey : CONCEALED_SECRET,
          model,
        })
  // What a copy yields, always the real key: an operator who copies without
  // revealing still gets something that runs. Undefined until there is a key,
  // and `CodeBlock` then offers no copy control rather than a dead one.
  const copied =
    baseUrl === undefined || apiKey === undefined
      ? undefined
      : buildSetupSnippets({ baseUrl, apiKey, model })

  return (
    <Dialog
      isOpen
      onOpenChange={(open) => {
        if (!open) onDismiss()
      }}
      size="lg"
      isAnnouncement
      title="Send your first request"
      description={
        <>
          It lands in{" "}
          <span className="text-foreground font-medium">
            {workspaceName ?? "this workspace"}
          </span>
          . We watch for it and finish setup for you.
        </>
      }
      status={
        <ListeningPanel
          failure={failure}
          attemptAt={attemptAt}
          isChecking={isChecking}
          checkFailed={checkFailed}
          onCheckNow={onCheckNow}
          onOpenTab={setTab}
          onLeave={onDismiss}
        />
      }
      footerStart={
        <p className="text-caption">
          Usage and the activity log stay empty until your first request lands.
        </p>
      }
      actions={
        <Button isPending={isSkipping} onPress={onSkip}>
          Skip
        </Button>
      }
    >
      <DialogSection>
        <ErrorBanner error={keyError} />
        <ErrorBanner error={skipError} />
        {apiKey === undefined ? (
          // Not a `CopyField` holding an empty string: every credential field
          // here conceals, so an empty one would show the same run of bullets a
          // real key does and invite a copy that yields nothing. The field
          // arrives with the key it is for.
          <div className="flex flex-col gap-2">
            <span className="text-emphasis">Your API key</span>
            <p
              aria-live="polite"
              className="border-border bg-surface-alt text-mono-caption text-subtle border px-3 py-2"
            >
              Creating your API key…
            </p>
          </div>
        ) : (
          <CopyField
            label="Your API key"
            value={apiKey}
            concealed={CONCEALED_SECRET}
            isRevealed={isRevealed}
            onRevealChange={setIsRevealed}
          />
        )}
        <p className="text-caption text-subtle">
          Shown once. Reveal it to read it, or copy it without. Set it as{" "}
          <code>{SNIPPET_KEY_ENV_VAR}</code> in your environment; reopening this
          guide issues a new key in its place.
        </p>
      </DialogSection>

      {baseUrl === undefined ? (
        <DialogSection>
          <MissingGatewayAddressNotice />
        </DialogSection>
      ) : null}

      {shown !== undefined ? (
        <DialogSection>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-emphasis">{instruction}</p>
            <TabRow>
              {SETUP_TABS.map(({ id, label }) => (
                <Tab key={id} isActive={tab === id} onPress={() => setTab(id)}>
                  {label}
                </Tab>
              ))}
            </TabRow>
          </div>
          {/* Bare: the tab row above already names the language, so a label row
              under it would say it twice. */}
          <CodeBlock
            label={tab}
            value={copied?.[tab]}
            arrangement="bare"
            isBounded
          >
            {shown[tab]}
          </CodeBlock>
          <p className="text-caption text-subtle">
            {tab === "agent"
              ? "Works with Claude Code, Codex, Cursor, and any agent that can edit files and run commands. It reads the key from your environment rather than carrying it."
              : carriesKey(tab) && !isRevealed
                ? "The key is hidden in the example above and copied in full. Reveal it with the control beside the field."
                : "Prefer to have an agent wire this up? The Agent tab is a paste-ready prompt."}
          </p>
          {model === undefined ? (
            <p className="text-caption text-subtle">
              No model is being served yet, so the examples name{" "}
              <code>{SNIPPET_MODEL_PLACEHOLDER}</code>. Replace it with one from
              the{" "}
              <Link
                to="/models"
                onClick={onDismiss}
                className="text-link hover:text-link-hover font-medium"
              >
                Models
              </Link>{" "}
              page.
            </p>
          ) : null}
        </DialogSection>
      ) : null}
    </Dialog>
  )
}
