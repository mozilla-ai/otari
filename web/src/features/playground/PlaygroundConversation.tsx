import type { Dispatch, SetStateAction } from "react"

import { ChatPanel } from "./ChatPanel"
import type { PlaygroundModel } from "./helpers/playgroundModels"
import type { PanelState } from "./helpers/playgroundTypes"
import { ModelSelect } from "./ModelSelect"
import { CHAT_COLUMN } from "./playgroundLayout"

/**
 * The conversation area: one panel, or two side by side while comparing.
 *
 * Each column carries its own picker while comparing, and each picker excludes
 * the other column's model: comparing a model with itself produces two answers
 * nobody can tell apart, and a rating over them means nothing. Changing A to the
 * model B holds clears B's selection rather than silently swapping them, so the
 * reader chooses B's replacement instead of finding one chosen for them.
 *
 * Two columns become one stacked pair below `md`. A phone cannot show two
 * transcripts side by side, and the alternative (hiding compare on a small
 * screen) would take the feature away rather than fit it.
 *
 * Neither column scrolls on its own, and while comparing they are top-aligned
 * rather than stretched: the page is the scroll container, so the shorter
 * answer ends where it ends instead of being padded out to the taller one.
 */
export function PlaygroundConversation({
  isComparing,
  panelA,
  panelB,
  setPanelA,
  setPanelB,
  models,
  pinnedKeys,
  onTogglePin,
  onRegenerate,
}: {
  isComparing: boolean
  panelA: PanelState
  panelB: PanelState
  setPanelA: Dispatch<SetStateAction<PanelState>>
  setPanelB: Dispatch<SetStateAction<PanelState>>
  models: readonly PlaygroundModel[]
  pinnedKeys: readonly string[]
  onTogglePin: (key: string) => void
  onRegenerate: (
    panel: PanelState,
    setPanel: Dispatch<SetStateAction<PanelState>>,
  ) => void
}) {
  if (!isComparing) {
    return (
      <div className={`${CHAT_COLUMN} flex pt-8 pb-6`}>
        <ChatPanel
          panel={panelA}
          onRegenerate={() => onRegenerate(panelA, setPanelA)}
        />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-8 pt-6 pb-6 md:flex-row md:items-start">
      <div className="flex min-w-0 flex-1 flex-col gap-6">
        <h2 className="text-overline">Model A</h2>
        <ModelSelect
          label="Model A"
          value={panelA.model}
          models={models}
          pinnedKeys={pinnedKeys}
          onTogglePin={onTogglePin}
          unavailableKeys={panelB.model ? [panelB.model] : undefined}
          onChange={(key) => {
            setPanelA((prev) => ({ ...prev, model: key }))
            if (key === panelB.model) {
              setPanelB((prev) => ({ ...prev, model: "" }))
            }
          }}
          className="w-full"
        />
        <ChatPanel
          panel={panelA}
          onRegenerate={() => onRegenerate(panelA, setPanelA)}
        />
      </div>
      <div className="flex min-w-0 flex-1 flex-col gap-6">
        <h2 className="text-overline">Model B</h2>
        <ModelSelect
          label="Model B"
          value={panelB.model}
          models={models}
          pinnedKeys={pinnedKeys}
          onTogglePin={onTogglePin}
          unavailableKeys={panelA.model ? [panelA.model] : undefined}
          onChange={(key) => {
            setPanelB((prev) => ({ ...prev, model: key }))
          }}
          className="w-full"
        />
        <ChatPanel
          panel={panelB}
          onRegenerate={() => onRegenerate(panelB, setPanelB)}
        />
      </div>
    </div>
  )
}
