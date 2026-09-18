import type { FormEvent, KeyboardEvent } from "react"
import { useCallback, useEffect, useRef, useState } from "react"

import type { PlaygroundComparisonPreference } from "@/client"
import { useModels } from "@/shared/api/models"
import {
  fetchPlaygroundConversation,
  useDeletePlaygroundComparison,
  useDeletePlaygroundConversation,
  usePlaygroundComparisons,
  usePlaygroundConsent,
  usePlaygroundConversations,
  usePlaygroundFavoriteModels,
  usePlaygroundTools,
  useReplacePlaygroundFavoriteModels,
  useSavePlaygroundComparison,
  useSavePlaygroundConversation,
  useUpdatePlaygroundConsent,
} from "@/shared/api/playground"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"

import {
  buildPlaygroundModels,
  pickInitialModel,
} from "../helpers/playgroundModels"
import {
  buildComparisonRequest,
  buildConversationRequest,
  findRatedExchange,
  togglePinnedModel,
} from "../helpers/playgroundSave"
import { turnsForRegenerate } from "../helpers/playgroundTurns"
import {
  type ChatTurn,
  EMPTY_PANEL,
  type PanelState,
} from "../helpers/playgroundTypes"
import {
  derivePlaygroundGate,
  haveBothPanelsAnswered,
  isAnyPanelBusy,
  shouldShowWelcome,
} from "../helpers/playgroundView"
import { readRememberedModel, rememberModel } from "../helpers/rememberedModel"
import { usePlaygroundStream } from "./usePlaygroundStream"
import { usePlaygroundToolSelection } from "./usePlaygroundToolSelection"

/** What the page asked for and is waiting on a retention grant to do. */
export type PendingConsent =
  | { kind: "conversation" }
  | { kind: "comparison"; preference: PlaygroundComparisonPreference }

/** How long the "recorded, thanks" line stays up after a rating. */
const RATING_ACKNOWLEDGEMENT_MS = 3000

/**
 * The Playground's controller: every piece of state the page renders from, and
 * every call it makes.
 *
 * One hook, deliberately, and the reason is the coupling rather than
 * convenience. Sending a message clears the rating, disarms Save, and touches
 * both panels; a rating reads both panels' turns; loading a transcript replaces
 * one panel and closes a dialog. Splitting these into per-concern hooks would
 * mean lifting the same state back up to a component that coordinates them,
 * which is the god component otari-ai#1360 broke up. The page stays a
 * composition of presentational pieces because this is where the coordination
 * lives.
 */
export function usePlayground() {
  const { selected, isLoading: isLoadingWorkspace } = useSelectedWorkspace()
  const workspaceId = selected?.workspace_id

  const catalog = useModels()
  const models = buildPlaygroundModels(catalog.data)

  const consent = usePlaygroundConsent()
  const updateConsent = useUpdatePlaygroundConsent()
  const canStoreConversations = consent.data?.store_conversations ?? false
  const canStoreComparisons = consent.data?.store_comparisons ?? false

  const tools = usePlaygroundTools(workspaceId)
  const toolSelection = usePlaygroundToolSelection(tools.data)

  const conversations = usePlaygroundConversations(workspaceId)
  const saveConversation = useSavePlaygroundConversation(workspaceId)
  const deleteConversation = useDeletePlaygroundConversation(workspaceId)
  const comparisons = usePlaygroundComparisons(workspaceId)
  const saveComparison = useSavePlaygroundComparison(workspaceId)
  const deleteComparison = useDeletePlaygroundComparison(workspaceId)

  const pinned = usePlaygroundFavoriteModels(workspaceId)
  const replacePinned = useReplacePlaygroundFavoriteModels(workspaceId)
  const pinnedKeys = pinned.data?.model_keys ?? []

  const [isComparing, setIsComparing] = useState(false)
  const [draft, setDraft] = useState("")
  const [panelA, setPanelA] = useState<PanelState>(EMPTY_PANEL)
  const [panelB, setPanelB] = useState<PanelState>(EMPTY_PANEL)
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [isNewChatConfirmOpen, setIsNewChatConfirmOpen] = useState(false)
  const [isConversationSaved, setIsConversationSaved] = useState(false)
  // The id the last Save recorded, so deleting that row from history re-arms
  // Save. Without it the button stays disabled for a transcript that no longer
  // exists on the server.
  const [savedConversationId, setSavedConversationId] = useState<
    string | undefined
  >(undefined)
  const [ratingState, setRatingStateNow] = useState<
    "none" | "acknowledged" | "dismissed"
  >("none")
  // The acknowledgement dismisses itself on a timer, which has to die with the
  // hook and be cancelled by anything that moves the state itself. Without the
  // second half a pending dismissal lands after the state has already gone back
  // to "none" (editing the exchange, or loading a stored transcript) and
  // overwrites it three seconds later.
  const ratingDismissal = useRef<ReturnType<typeof setTimeout> | undefined>(
    undefined,
  )
  useEffect(() => () => clearTimeout(ratingDismissal.current), [])
  // Stable, which is what lets `invalidateSavedState` below name it as a
  // dependency without re-creating itself on every render.
  const setRatingState = useCallback(
    (next: "none" | "acknowledged" | "dismissed") => {
      clearTimeout(ratingDismissal.current)
      setRatingStateNow(next)
    },
    [],
  )
  const [pendingConsent, setPendingConsent] = useState<
    PendingConsent | undefined
  >(undefined)
  // Loading a saved transcript is an action rather than a query, so a failure
  // has no mutation state to read: it is kept here and reported beside the
  // page's other write failures.
  const [loadError, setLoadError] = useState<unknown>(undefined)
  // The model a reader picked while a transcript was on screen, held until they
  // confirm losing it.
  const [pendingModelChange, setPendingModelChange] = useState<
    string | undefined
  >(undefined)

  const { streamReply, stop } = usePlaygroundStream({
    workspaceId: workspaceId ?? "",
    tools: toolSelection.selection,
  })

  /** Anything that makes the on-screen exchange different from the saved one. */
  const invalidateSavedState = useCallback(() => {
    setRatingState("none")
    setIsConversationSaved(false)
    setSavedConversationId(undefined)
  }, [setRatingState])

  /** Put both panels back to empty, keeping whichever models they hold. */
  const clearPanels = useCallback(() => {
    stop()
    setPanelA((prev) => ({ ...prev, turns: [] }))
    setPanelB((prev) => ({ ...prev, turns: [] }))
    invalidateSavedState()
  }, [stop, invalidateSavedState])

  // Switching workspace in the sidebar starts over. Nothing on screen survives
  // it, and it cannot: a transcript belongs to the workspace whose credentials
  // answered it, so carrying it across would save it under the new workspace
  // and send its history to a model the new catalog may not even serve. The
  // pinned models and the saved history are per workspace too, and their
  // queries are already keyed on it.
  //
  // Keyed on the id rather than done in the seeding effect below, because that
  // one runs whenever the catalog does and this must run only on the change.
  const previousWorkspaceRef = useRef(workspaceId)
  useEffect(() => {
    if (previousWorkspaceRef.current === workspaceId) return
    previousWorkspaceRef.current = workspaceId
    stop()
    setPanelA(EMPTY_PANEL)
    setPanelB(EMPTY_PANEL)
    setDraft("")
    setIsComparing(false)
    invalidateSavedState()
  }, [workspaceId, stop, invalidateSavedState])

  // Restore A from this browser; B stays empty until explicitly chosen.
  const defaultModel = models[0]?.key ?? ""
  useEffect(() => {
    if (!defaultModel || !workspaceId) return
    setPanelA((prev) =>
      prev.model
        ? prev
        : {
            ...prev,
            model: pickInitialModel(readRememberedModel(workspaceId), models),
          },
    )
  }, [defaultModel, workspaceId, models])

  useEffect(() => {
    if (panelA.model && workspaceId) rememberModel(workspaceId, panelA.model)
  }, [panelA.model, workspaceId])

  const isBusy = isAnyPanelBusy(panelA, panelB)

  const send = useCallback(
    async (question: string) => {
      const trimmed = question.trim()
      if (!trimmed || !workspaceId || !panelA.model) return
      // The send *button* becomes Stop while a reply is in flight, so the only
      // way in here is the Enter key, and a second stream into the same panel
      // interleaves two replies into one turn.
      if (isBusy || (isComparing && !panelB.model)) return
      const turn: ChatTurn = { role: "user", content: trimmed }

      setDraft("")
      invalidateSavedState()

      // Both comparison panels receive the same question and stream independently.
      await Promise.all([
        streamReply(setPanelA, panelA.model, [...panelA.turns, turn]),
        ...(isComparing && panelB.model
          ? [streamReply(setPanelB, panelB.model, [...panelB.turns, turn])]
          : []),
      ])
    },
    [
      workspaceId,
      panelA,
      panelB,
      isComparing,
      isBusy,
      streamReply,
      invalidateSavedState,
    ],
  )

  const regenerate = useCallback(
    (panel: PanelState, setPanel: typeof setPanelA) => {
      if (!panel.model || panel.isStreaming) return
      const base = turnsForRegenerate(panel.turns)
      if (base[base.length - 1]?.role !== "user") return
      invalidateSavedState()
      void streamReply(setPanel, panel.model, base)
    },
    [streamReply, invalidateSavedState],
  )

  const submit = (event: FormEvent) => {
    event.preventDefault()
    void send(draft)
  }

  // Enter sends, Shift+Enter breaks the line, which is what a chat composer is
  // expected to do and what every model of this control does elsewhere.
  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      void send(draft)
    }
  }

  const clearConversation = clearPanels

  const performSaveConversation = useCallback(() => {
    if (!workspaceId || panelA.turns.length === 0) return
    saveConversation.mutate(
      buildConversationRequest({
        workspaceId,
        model: panelA.model,
        turns: panelA.turns,
      }),
      {
        onSuccess: (saved) => {
          setIsConversationSaved(true)
          setSavedConversationId(saved.id)
        },
      },
    )
  }, [workspaceId, panelA.model, panelA.turns, saveConversation])

  /**
   * Save the single-panel transcript, asking for retention consent first if it
   * has not been granted.
   *
   * Comparing has no Save: two models means there is no single "the
   * conversation", and a comparison is recorded through an explicit rating
   * rather than a default. A Save-triggered rating would be indistinguishable
   * from a deliberate judgment in a table whose whole value is that its rows are
   * deliberate.
   */
  const requestSaveConversation = () => {
    if (panelA.turns.length === 0) return
    if (!canStoreConversations) {
      setPendingConsent({ kind: "conversation" })
      return
    }
    performSaveConversation()
  }

  const performRate = useCallback(
    (preference: PlaygroundComparisonPreference) => {
      if (!workspaceId) return
      const exchange = findRatedExchange(panelA.turns, panelB.turns)
      if (!exchange) return
      saveComparison.mutate(
        buildComparisonRequest({
          workspaceId,
          modelA: panelA.model,
          modelB: panelB.model,
          exchange,
          preference,
        }),
        {
          onSuccess: () => {
            setRatingState("acknowledged")
            ratingDismissal.current = setTimeout(
              () => setRatingState("dismissed"),
              RATING_ACKNOWLEDGEMENT_MS,
            )
          },
        },
      )
    },
    [workspaceId, panelA, panelB, saveComparison, setRatingState],
  )

  const requestRate = (preference: PlaygroundComparisonPreference) => {
    if (!canStoreComparisons) {
      setPendingConsent({ kind: "comparison", preference })
      return
    }
    performRate(preference)
  }

  /**
   * Grant the flag the pending action needs, then run the action.
   *
   * A refused grant leaves the dialog open with its own error showing, because
   * the action behind it must not run: saving without consent is the one thing
   * the prompt exists to prevent. `mutateAsync` rejects, so the rejection is
   * caught here rather than escaping a click handler.
   */
  const confirmPendingConsent = async () => {
    if (!pendingConsent) return
    try {
      if (pendingConsent.kind === "conversation") {
        await updateConsent.mutateAsync({ store_conversations: true })
        performSaveConversation()
      } else {
        await updateConsent.mutateAsync({ store_comparisons: true })
        performRate(pendingConsent.preference)
      }
    } catch {
      // `updateConsent.error` is what the dialog renders; nothing to add.
      return
    }
    setPendingConsent(undefined)
  }

  const loadConversation = async (conversationId: string) => {
    setIsHistoryOpen(false)
    const saved = conversations.data?.data.find(
      (item) => item.id === conversationId,
    )
    if (!saved) {
      setLoadError(
        new Error(
          "This conversation is no longer available. Reopen history and try again.",
        ),
      )
      return
    }
    setLoadError(undefined)
    // Before the fetch, not after: a reply still streaming would otherwise
    // patch its next fragment onto the transcript that replaced it.
    stop()
    let loaded: Awaited<ReturnType<typeof fetchPlaygroundConversation>>
    try {
      loaded = await fetchPlaygroundConversation(conversationId)
    } catch (error) {
      // Reported rather than thrown: the caller is a click handler, so a
      // rejection here would be an unhandled one and the reader would be left
      // with a dialog that did nothing.
      setLoadError(error)
      return
    }
    setIsComparing(false)
    setPanelB(EMPTY_PANEL)
    setPanelA({
      ...EMPTY_PANEL,
      // Through the same guard the seeding effect uses: a transcript can
      // outlive the model that produced it, and restoring a key the catalog no
      // longer serves leaves the picker blank while Send still dispatches it.
      model: pickInitialModel(saved.model, models),
      turns: loaded.data.map((message) => ({
        role: message.role === "assistant" ? "assistant" : "user",
        content: message.content,
        reasoning: message.reasoning ?? undefined,
      })),
    })
    // A loaded transcript is already stored, so Save is disarmed rather than
    // offering to store a second copy of it.
    setIsConversationSaved(true)
    setSavedConversationId(conversationId)
    setRatingState("none")
  }

  const removeConversation = async (conversationId: string) => {
    await deleteConversation.mutateAsync(conversationId)
    if (conversationId === savedConversationId) {
      setIsConversationSaved(false)
      setSavedConversationId(undefined)
    }
  }

  const toggleCompare = () => {
    const next = !isComparing
    // Either direction: a stream in flight belongs to the layout that started
    // it, and letting it land would repopulate a column this just cleared.
    stop()
    setIsComparing(next)
    if (next) {
      // Both columns start empty, which is where this port departs from the
      // hosted original: that one kept panel A's transcript. Each model is sent
      // its own panel's history, so a column carrying an earlier conversation
      // answers a different prompt from the one beside it, and the rating would
      // be a judgment over an unequal contest. A saved transcript is still in
      // the history.
      setPanelA((prev) => ({ ...prev, turns: [] }))
      setPanelB((prev) => ({ ...prev, model: "", turns: [] }))
      invalidateSavedState()
    }
  }

  /**
   * Point panel A at another model.
   *
   * With a transcript on screen this asks first, because switching is not a
   * free action: each request carries the panel's whole history, so the new
   * model would be answering the old one's conversation, and a save labels that
   * transcript with one model when two answered it. Confirming clears it, which
   * is what makes the stored row true. An empty panel switches straight away,
   * which is the ordinary case.
   */
  const selectPanelAModel = (modelKey: string) => {
    if (panelA.turns.length === 0) {
      setPanelA((prev) => ({ ...prev, model: modelKey }))
      return
    }
    setPendingModelChange(modelKey)
  }

  const confirmModelChange = () => {
    if (pendingModelChange === undefined) return
    stop()
    setPanelA({ ...EMPTY_PANEL, model: pendingModelChange })
    setPanelB((prev) => ({ ...prev, turns: [] }))
    invalidateSavedState()
    setPendingModelChange(undefined)
  }

  const togglePin = (modelKey: string) => {
    replacePinned.mutate(togglePinnedModel(pinnedKeys, modelKey))
  }

  const gate = derivePlaygroundGate({
    isLoading: isLoadingWorkspace || (catalog.isPending && !catalog.data),
    isCatalogError: catalog.isError,
    hasWorkspace: workspaceId !== undefined,
    modelCount: models.length,
  })

  return {
    // Where the page stands
    gate,
    // Every failure with no control of its own to report it: a save, a rating,
    // and loading a transcript back. The deletes report inside their confirm
    // dialogs and the consent grant inside its own, so neither is here.
    actionError:
      saveConversation.error ?? saveComparison.error ?? loadError ?? undefined,
    // The catalog read's own error, so the gate notice can report the failure
    // rather than a generic one: it is the only gate with something to say.
    catalogError: catalog.error,
    workspaceId,
    isBusy,
    canChat: workspaceId !== undefined && panelA.model !== "",
    isShowingWelcome: shouldShowWelcome(isComparing, panelA),

    // Composer
    draft,
    setDraft,
    submit,
    handleComposerKeyDown,
    stop,

    // Panels
    panelA,
    panelB,
    setPanelA,
    setPanelB,
    regenerate,
    isComparing,
    toggleCompare,

    // Models
    models,
    pinnedKeys,
    togglePin,
    selectPanelAModel,
    pendingModelChange,
    confirmModelChange,
    cancelModelChange: () => setPendingModelChange(undefined),

    // Tools
    tools: tools.data,
    ...toolSelection,

    // Saved transcripts
    conversations: conversations.data?.data ?? [],
    isHistoryLoading:
      (conversations.isPending && !conversations.data) ||
      (comparisons.isPending && !comparisons.data),
    historyError: conversations.error ?? comparisons.error ?? undefined,
    isHistoryOpen,
    setIsHistoryOpen,
    loadConversation,
    requestSaveConversation,
    isSavePending: saveConversation.isPending,
    isConversationSaved,
    removeConversation,

    // New chat
    isNewChatConfirmOpen,
    setIsNewChatConfirmOpen,
    clearConversation,

    // Rating
    haveBothAnswered: haveBothPanelsAnswered(isComparing, panelA, panelB),
    ratingState,
    requestRate,
    isRatePending: saveComparison.isPending,

    // Saved comparisons
    comparisons: comparisons.data?.data ?? [],
    removeComparison: (id: string) => deleteComparison.mutateAsync(id),

    // Just-in-time retention consent
    pendingConsent,
    confirmPendingConsent,
    cancelPendingConsent: () => setPendingConsent(undefined),
    isConfirmingConsent: updateConsent.isPending,
    consentError: updateConsent.error ?? undefined,
  }
}
