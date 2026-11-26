import { Link } from "@tanstack/react-router"
import { useRef, useState } from "react"

import type { ActivationAttempt, ActivationIssuedKey } from "@/client"
import { Button } from "@/design-system/actions/Button"
import {
  CONCEALED_SECRET,
  CopyField,
  concealedFingerprint,
} from "@/design-system/actions/CopyField"
import { CodeBlock } from "@/design-system/content/CodeBlock"
import { Dialog, DialogSection } from "@/design-system/feedback/Dialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { ListeningPanel } from "@/features/onboarding/ListeningPanel"
import { setupSuccessFace } from "@/features/onboarding/SetupSuccess"
import type { SetupFailure } from "@/features/onboarding/setupFailureCopy"
import {
  buildSetupSnippets,
  DEFAULT_SETUP_TAB,
  SETUP_TABS,
  type SetupSnippetId,
} from "@/features/onboarding/setupSnippets"
import { useLandingTransition } from "@/features/onboarding/setupTransition"
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
 * in it is worth reading, so the guide is allowed to be the screen, offered
 * once per workspace.
 *
 * **Divided rather than stacked.** Each part is a `DialogSection`, so the key,
 * the example and the wait are separated by hairlines that run the width of the
 * frame. A column of gaps would have read as three cards floating in a sheet,
 * which is the thing this design system does not do.
 *
 * When the first request lands, the sheet turns into its success face
 * (`SetupSuccess`) in the same frame (`setupTransition`).
 *
 * Presentational, and controlled by `SetupGuide` above it: every piece of state
 * here is either the tab somebody picked or whether the key is on screen.
 */
export function SetupSheet({
  workspaceName,
  apiKey,
  keyPrefix,
  keySuffix,
  issuedKey,
  isUsingIssuedKey,
  onUseIssuedKey,
  baseUrl,
  model,
  failure,
  isMember = false,
  attemptAt,
  isChecking,
  checkFailed,
  keyError,
  isCreatingKey,
  skipError,
  isSkipping,
  onCreateKey,
  onCheckNow,
  onSkip,
  onLeave,
  landed,
}: {
  workspaceName?: string
  /** The issued key's plaintext, or undefined until one is created. */
  apiKey?: string
  /** The fingerprint the server stored for the issued key, shown while it is concealed. */
  keyPrefix?: string
  keySuffix?: string
  /** The setup key this workspace already holds, when this visit has not minted one. */
  issuedKey?: ActivationIssuedKey
  /** Whether the caller chose to keep `issuedKey` rather than replace it. */
  isUsingIssuedKey: boolean
  onUseIssuedKey: () => void
  /** Where a request belongs. */
  baseUrl: string
  /** The first model the gateway can serve, when it can serve one. */
  model?: string
  failure?: SetupFailure
  /** A member joining a workspace that may already have traffic, so its empty-dashboard note is left out. */
  isMember?: boolean
  attemptAt?: string
  isChecking: boolean
  checkFailed: boolean
  keyError: unknown
  isCreatingKey: boolean
  skipError: unknown
  isSkipping: boolean
  /** Mints the key. Nothing is minted until somebody asks, so leaving first leaves no key behind. */
  onCreateKey: () => void
  onCheckNow: () => void
  onSkip: () => void
  /** Retires the guide on the way to a page a guidance link names. */
  onLeave: () => void
  /** Set once the first request has landed: the sheet then shows its payoff. */
  landed?: {
    /** The request that succeeded. Absent if the gateway reported none. */
    attempt?: ActivationAttempt
    onDismiss: () => void
    onOpenActivity: () => void
  }
}) {
  const [tab, setTab] = useState<SetupSnippetId>(DEFAULT_SETUP_TAB)
  // One reveal for the whole sheet: the key field and the example built around
  // it hide the same secret, so revealing one and not the other would be a
  // distinction with nothing behind it.
  const [isRevealed, setIsRevealed] = useState(false)
  const faceRef = useRef<HTMLDivElement>(null)
  const showsSuccess = useLandingTransition(faceRef, landed !== undefined)

  const concealedKey =
    apiKey === undefined
      ? CONCEALED_SECRET
      : concealedFingerprint(keyPrefix, keySuffix)
  // A kept key's plaintext is gone, so the examples read the variable it was
  // exported as, and a copy of them still runs.
  const keyInUse =
    apiKey ??
    (issuedKey && isUsingIssuedKey ? `$${SNIPPET_KEY_ENV_VAR}` : undefined)

  const instruction = SETUP_TABS.find(({ id }) => id === tab)?.instruction ?? ""
  // A minted key shows its stand-in until revealed, a kept key the variable,
  // and no key yet the stand-in, so the examples are there from the first frame.
  const shown = buildSetupSnippets({
    baseUrl,
    apiKey:
      apiKey === undefined
        ? (keyInUse ?? concealedKey)
        : isRevealed
          ? apiKey
          : concealedKey,
    model,
  })
  // A copy always runs: the minted key itself or the kept key's variable. With
  // neither, `CodeBlock` offers no copy control rather than a dead one.
  const copied =
    keyInUse === undefined
      ? undefined
      : buildSetupSnippets({ baseUrl, apiKey: keyInUse, model })

  // The same `Dialog` in the same place, so the frame stays open across faces.
  if (showsSuccess && landed) {
    return (
      <Dialog
        isOpen
        onOpenChange={(isOpen) => {
          if (!isOpen) landed.onDismiss()
        }}
        size="lg"
        isAnnouncement
        faceRef={faceRef}
        {...setupSuccessFace({
          attempt: landed.attempt,
          workspaceName,
          onOpenActivity: landed.onOpenActivity,
        })}
      />
    )
  }

  return (
    <Dialog
      isOpen
      // Empty, because this sheet is not dismissable: `Dialog` gates the
      // callback on that, so a handler here would be a dismissal path that
      // cannot run. `onLeave` reaches it from the guidance links instead.
      onOpenChange={() => {}}
      size="lg"
      isAnnouncement
      isDismissable={false}
      faceRef={faceRef}
      isScanning={!checkFailed}
      scanTone={failure ? "danger" : "accent"}
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
          onLeave={onLeave}
        />
      }
      footerStart={
        isMember ? undefined : (
          <p className="text-caption">
            Usage and the activity log stay empty until your first request
            lands.
          </p>
        )
      }
      actions={
        // The wrapper names a place (see actions.md), which is all it does:
        // the footer's own rules reach Skip through it and still give the phone
        // sheet its full width and its press suppression.
        <div className="otari-setup-actions flex w-full">
          <Button isPending={isSkipping} onPress={onSkip}>
            Skip
          </Button>
        </div>
      }
    >
      <DialogSection>
        <ErrorBanner error={keyError} />
        <ErrorBanner error={skipError} />
        {apiKey !== undefined ? (
          <CopyField
            label="Your API key"
            value={apiKey}
            concealed={concealedKey}
            isRevealed={isRevealed}
            onRevealChange={setIsRevealed}
          />
        ) : issuedKey ? (
          <IssuedKey
            issuedKey={issuedKey}
            isUsing={isUsingIssuedKey}
            isReplacing={isCreatingKey}
            onUse={onUseIssuedKey}
            onReplace={onCreateKey}
          />
        ) : (
          // Not a `CopyField` holding an empty string: every credential field
          // here conceals, so an empty one would show the same run of bullets a
          // real key does and invite a copy that yields nothing. The field
          // arrives with the key it is for.
          <div className="flex flex-col items-start gap-2">
            <span className="text-emphasis">Your API key</span>
            <Button
              variant="primary"
              isPending={isCreatingKey}
              onPress={onCreateKey}
            >
              Create API key
            </Button>
          </div>
        )}
        {apiKey === undefined && issuedKey ? (
          isUsingIssuedKey ? (
            <p className="text-caption text-subtle">
              The examples read it from <code>{SNIPPET_KEY_ENV_VAR}</code> in
              your environment.
            </p>
          ) : null
        ) : (
          <p className="text-caption text-subtle">
            Shown once. Reveal it to read it, or copy it without. Set it as{" "}
            <code>{SNIPPET_KEY_ENV_VAR}</code> in your environment.
          </p>
        )}
      </DialogSection>

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
            ? "Works with Claude Code, Codex, and Cursor. Reads your key from the environment."
            : tab === "curl"
              ? "Hidden keys use a stand-in; copies include your real key."
              : `Reads your key from ${SNIPPET_KEY_ENV_VAR}; export it as the cURL tab shows.`}
        </p>
        {model === undefined ? (
          <p className="text-caption text-subtle">
            No model is being served yet, so the examples name{" "}
            <code>{SNIPPET_MODEL_PLACEHOLDER}</code>. Replace it with one from
            the{" "}
            <Link
              to="/models"
              onClick={onLeave}
              className="text-link hover:text-link-hover font-medium"
            >
              Models
            </Link>{" "}
            page.
          </p>
        ) : null}
      </DialogSection>
    </Dialog>
  )
}

/**
 * The setup key a previous visit issued. Its plaintext was shown once and is
 * gone, and replacing it rotates the row, so keeping it is the default and the
 * replacement says what it costs.
 */
function IssuedKey({
  issuedKey,
  isUsing,
  isReplacing,
  onUse,
  onReplace,
}: {
  issuedKey: ActivationIssuedKey
  isUsing: boolean
  isReplacing: boolean
  onUse: () => void
  onReplace: () => void
}) {
  const ending = issuedKey.key_suffix ? (
    <>
      {" "}
      ending <code>••{issuedKey.key_suffix}</code>
    </>
  ) : null
  return (
    <div className="flex flex-col items-start gap-2">
      <span className="text-emphasis">Your API key</span>
      <p className="text-body">
        {isUsing ? "Using your setup key" : "You already have a setup key"}
        {ending}.
      </p>
      {isUsing ? null : (
        <Button variant="primary" onPress={onUse}>
          Use it
        </Button>
      )}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-caption text-subtle">Lost it?</span>
        <Button size="sm" isPending={isReplacing} onPress={onReplace}>
          Replace it
        </Button>
        <span className="text-caption text-subtle">
          The old key stops working.
        </span>
      </div>
    </div>
  )
}
