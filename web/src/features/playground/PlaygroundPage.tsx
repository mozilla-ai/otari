import { FiArrowDown } from "react-icons/fi"

import { IconButton } from "@/design-system/actions/IconButton"
import { ConfirmDialog } from "@/design-system/feedback/ConfirmDialog"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { useDocumentTitle } from "@/shared/hooks/useDocumentTitle"

import { ComparisonHistoryDialog } from "./ComparisonHistoryDialog"
import { ComparisonRatingBar } from "./ComparisonRatingBar"
import { ConversationHistoryDialog } from "./ConversationHistoryDialog"
import { useFollowConversation } from "./hooks/useFollowConversation"
import { usePlayground } from "./hooks/usePlayground"
import { PlaygroundComposer } from "./PlaygroundComposer"
import { PlaygroundConversation } from "./PlaygroundConversation"
import { PlaygroundGateNotice } from "./PlaygroundGateNotice"
import { PlaygroundToolbar } from "./PlaygroundToolbar"
import { PlaygroundWelcome } from "./PlaygroundWelcome"

/**
 * The Playground: chat with a model this gateway serves, or compare two.
 *
 * Composition only. Every decision it renders from is derived in
 * `hooks/usePlayground`, which is what keeps this file readable at the size the
 * page actually is: a toolbar, a conversation area that is one panel or two, a
 * composer in one of two positions, four dialogs, and a rating bar that appears
 * for one exchange at a time.
 *
 * The completions it sends are authorized by the dashboard session and billed to
 * the signed-in caller in the workspace the sidebar has selected; no API key is
 * involved and none is held in the browser. `src/gateway/api/routes/playground.py`
 * has the whole of that reasoning.
 */
export function PlaygroundPage() {
  useDocumentTitle("Playground")
  const playground = usePlayground()
  // Both panels feed one follow rule, because there is one scroll position.
  // The count is the signal: it changes when a turn is added, which is the only
  // thing that decides whether to follow, and it does not change on every
  // streamed fragment.
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
    // No padding and no height of its own. The shell's content wrapper already
    // pads every page, and it is auto-height, so a page cannot fill the
    // viewport from inside it: the conversation grows the page, the shell's
    // `<main>` scrolls it, and the composer sticks to the bottom of the
    // viewport so it stays reachable however long the transcript gets. One
    // scroll container, which is also what a phone wants.
    <div className="flex flex-col">
      <PlaygroundToolbar
        isComparing={playground.isComparing}
        modelA={playground.panelA.model}
        models={playground.models}
        pinnedKeys={playground.pinnedKeys}
        onTogglePin={playground.togglePin}
        onModelAChange={playground.selectPanelAModel}
        conversationCount={playground.conversations.length}
        onOpenHistory={() => playground.setIsHistoryOpen(true)}
        onSaveConversation={playground.requestSaveConversation}
        isSaveDisabled={
          playground.panelA.turns.length === 0 ||
          playground.isSavePending ||
          playground.isConversationSaved
        }
        isSavePending={playground.isSavePending}
        comparisonCount={playground.comparisons.length}
        onOpenComparisonHistory={() =>
          playground.setIsComparisonHistoryOpen(true)
        }
        onToggleCompare={playground.toggleCompare}
        onNewChat={() => playground.setIsNewChatConfirmOpen(true)}
        isNewChatDisabled={!hasTranscript}
      />

      {/* Under the toolbar, where the controls that can fail are: a refused
          save, a refused rating, or a transcript that would not load. */}
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

          {/* The end of the conversation, and what the follow rule scrolls
              to. Before the composer rather than after it, so "the bottom" is
              the newest answer and not a control that is always on screen. */}
          <div ref={follow.setEndMarker} aria-hidden className="h-0" />

          {/* `sticky` and not `fixed`: it stays inside the page's own column,
              so it keeps the shell's width and its left edge does not sit over
              the sidebar. The negative bottom margin cancels the wrapper's
              padding below it, so nothing shows between it and the edge. */}
          <div className="-mb-5 sticky bottom-0 z-10 bg-background pt-2 pb-5 md:-mb-6 md:pb-6">
            {/* Inside the sticky block, above the composer. In normal flow it
                sat at the document's end and the composer stuck over it, so the
                one control the comparison exists for was the one you could not
                see. Above the composer is also where it reads: the verdict on
                the last exchange, then the box for the next question. */}
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
                  className="pointer-events-auto rounded-full shadow-elevation-md"
                  onPress={follow.jumpToLatest}
                >
                  <FiArrowDown aria-hidden className="size-4" />
                </IconButton>
              </div>
            ) : null}
          </div>
        </>
      )}

      {/* Switching model with a transcript on screen. Its own dialog rather
          than the New chat one, because the copy has to say what is being
          traded: each request carries the panel's history, so the new model
          would answer the old one's conversation. */}
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

      {/* Retention consent, asked at the moment it is needed rather than on a
          settings page nobody visits first. The two bodies differ because the
          disclosures differ: a comparison stores both models' full answers. */}
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

      <ConversationHistoryDialog
        isOpen={playground.isHistoryOpen}
        onOpenChange={playground.setIsHistoryOpen}
        conversations={playground.conversations}
        onLoad={(id) => void playground.loadConversation(id)}
        onDelete={playground.removeConversation}
        isDeleting={playground.isDeletingConversation}
        deleteError={playground.deleteConversationError}
      />

      <ComparisonHistoryDialog
        isOpen={playground.isComparisonHistoryOpen}
        onOpenChange={playground.setIsComparisonHistoryOpen}
        comparisons={playground.comparisons}
        onDelete={playground.removeComparison}
        isDeleting={playground.isDeletingComparison}
        deleteError={playground.deleteComparisonError}
      />
    </div>
  )
}
