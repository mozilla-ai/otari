import { FiArrowDown } from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageIntro } from "@/design-system/layout/PageIntro"
import { Section } from "@/design-system/layout/Section"
import { useDocumentTitle } from "@/shared/hooks/useDocumentTitle"

import { ComparisonRatingBar } from "./ComparisonRatingBar"
import { useFollowConversation } from "./hooks/useFollowConversation"
import { usePlayground } from "./hooks/usePlayground"
import { ModelSelect } from "./ModelSelect"
import { PlaygroundComposer } from "./PlaygroundComposer"
import { PlaygroundConversation } from "./PlaygroundConversation"
import { PlaygroundGateNotice } from "./PlaygroundGateNotice"
import { PlaygroundHistory } from "./PlaygroundHistory"
import { PlaygroundToolbar } from "./PlaygroundToolbar"
import { PlaygroundWelcome } from "./PlaygroundWelcome"
import { CHAT_COLUMN } from "./playgroundLayout"

export function PlaygroundPage() {
  useDocumentTitle("Playground")
  const playground = usePlayground()
  const follow = useFollowConversation({
    turnCount: playground.panelA.turns.length + playground.panelB.turns.length,
    lastRole: playground.panelA.turns[playground.panelA.turns.length - 1]?.role,
  })

  if (playground.gate !== "ready") {
    return (
      <PlaygroundGateNotice
        gate={playground.gate}
        error={playground.catalogError}
      />
    )
  }

  const composerProps = {
    hasTranscript: playground.panelA.turns.length > 0,
    missingModel:
      playground.isComparing && !playground.panelB.model
        ? ("B" as const)
        : undefined,
    modelPicker: playground.isComparing ? undefined : (
      <ModelSelect
        label="Model"
        value={playground.panelA.model}
        models={playground.models}
        pinnedKeys={playground.pinnedKeys}
        onTogglePin={playground.togglePin}
        onChange={playground.selectPanelAModel}
        size="sm"
        className="min-w-0 max-w-full"
      />
    ),
    draft: playground.draft,
    onDraftChange: playground.setDraft,
    onSubmit: playground.submit,
    onKeyDown: playground.handleComposerKeyDown,
    onStop: playground.stop,
    canChat: playground.canChat,
    isBusy: playground.isBusy,
    tools: playground.tools,
    isWebSearchOn: playground.isWebSearchOn,
    isCodeExecutionOn: playground.isCodeExecutionOn,
    selectedMcpIds: playground.selectedMcpIds,
    toggleWebSearch: playground.toggleWebSearch,
    toggleCodeExecution: playground.toggleCodeExecution,
    toggleMcpServer: playground.toggleMcpServer,
  }

  const hasTranscript =
    playground.panelA.turns.length > 0 || playground.panelB.turns.length > 0

  return (
    <div className="flex min-h-[calc(100dvh-6rem)] flex-col md:min-h-[calc(100dvh-6.5rem)]">
      <Section className="border-b border-border">
        <PageIntro
          title="Playground"
          action={
            <PlaygroundToolbar
              isComparing={playground.isComparing}
              hasTranscript={hasTranscript}
              onToggleCompare={playground.toggleCompare}
              onNewChat={() => playground.setIsNewChatConfirmOpen(true)}
              history={
                <PlaygroundHistory
                  conversations={playground.conversations}
                  comparisons={playground.comparisons}
                  isOpen={playground.isHistoryOpen}
                  onOpenChange={playground.setIsHistoryOpen}
                  onLoad={playground.loadConversation}
                  onDeleteConversation={playground.removeConversation}
                  onDeleteComparison={playground.removeComparison}
                  onSave={
                    !playground.isComparing && hasTranscript
                      ? playground.requestSaveConversation
                      : undefined
                  }
                  isSaveDisabled={
                    playground.isConversationSaved || playground.isBusy
                  }
                  isSavePending={playground.isSavePending}
                  isLoading={playground.isHistoryLoading}
                  error={playground.historyError}
                />
              }
            />
          }
        />
      </Section>
      {playground.actionError ? (
        <div className="pt-4">
          <ErrorBanner error={playground.actionError} />
        </div>
      ) : null}

      {playground.isShowingWelcome ? (
        <PlaygroundWelcome
          composerProps={composerProps}
          onSelectPrompt={playground.setDraft}
        />
      ) : (
        <>
          <PlaygroundConversation
            isComparing={playground.isComparing}
            panelA={playground.panelA}
            panelB={playground.panelB}
            setPanelA={playground.setPanelA}
            setPanelB={playground.setPanelB}
            models={playground.models}
            pinnedKeys={playground.pinnedKeys}
            onTogglePin={playground.togglePin}
            onRegenerate={playground.regenerate}
          />
          <div ref={follow.setEndMarker} aria-hidden className="h-0" />
          <div
            className={`${playground.isComparing ? "w-full" : CHAT_COLUMN} -mb-5 sticky bottom-0 z-10 mt-auto bg-background pt-7 pb-[max(1.25rem,env(safe-area-inset-bottom))] md:-mb-6 md:pb-6`}
          >
            {playground.haveBothAnswered &&
            playground.ratingState !== "dismissed" ? (
              <ComparisonRatingBar
                isAcknowledged={playground.ratingState === "acknowledged"}
                isPending={playground.isRatePending}
                onRate={playground.requestRate}
              />
            ) : null}
            <PlaygroundComposer {...composerProps} />
            {follow.isJumpVisible ? (
              <div className="-top-5 pointer-events-none absolute inset-x-0 flex justify-center">
                <IconButton
                  label="Scroll to latest"
                  variant="primary"
                  className="pointer-events-auto"
                  onPress={follow.jumpToLatest}
                >
                  <FiArrowDown aria-hidden className="size-4" />
                </IconButton>
              </div>
            ) : null}
          </div>
        </>
      )}
      <ConfirmDialog
        isOpen={playground.pendingModelChange !== undefined}
        onOpenChange={(next) => {
          if (!next) playground.cancelModelChange()
        }}
        heading="Switch model and start over?"
        body="A conversation is sent to one model, so switching clears what is on screen. Save it first if you want to keep it."
        confirmLabel="Switch model"
        confirmVariant="primary"
        isPending={false}
        onConfirm={playground.confirmModelChange}
      />

      <ConfirmDialog
        isOpen={playground.isNewChatConfirmOpen}
        onOpenChange={playground.setIsNewChatConfirmOpen}
        heading="Start a new chat?"
        body="This clears what is on screen. Anything you have not saved is lost."
        confirmLabel="New chat"
        confirmVariant="primary"
        isPending={false}
        onConfirm={() => {
          playground.clearConversation()
          playground.setIsNewChatConfirmOpen(false)
        }}
      />
      <ConfirmDialog
        isOpen={playground.pendingConsent !== undefined}
        onOpenChange={(next) => {
          if (!next) playground.cancelPendingConsent()
        }}
        heading={
          playground.pendingConsent?.kind === "comparison"
            ? "Save this rating?"
            : "Save this conversation?"
        }
        body={
          playground.pendingConsent?.kind === "comparison"
            ? "This stores the question and both models' full answers, visible only to you. You can delete saved comparisons from the Playground."
            : "This stores your messages and the model's replies, visible only to you. You can delete saved conversations from the Playground."
        }
        confirmLabel="Save"
        confirmVariant="primary"
        isPending={playground.isConfirmingConsent}
        error={playground.consentError}
        onConfirm={() => {
          void playground.confirmPendingConsent()
        }}
      />
    </div>
  )
}
