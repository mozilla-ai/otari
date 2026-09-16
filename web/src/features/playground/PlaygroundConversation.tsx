import type { Dispatch, SetStateAction } from "react"
import { ChatPanel } from "./ChatPanel"
import { type PlaygroundModel, splitModelKey } from "./helpers/playgroundModels"
import type { PanelState } from "./helpers/playgroundTypes"
import { ModelSelect } from "./ModelSelect"
import { CHAT_COLUMN } from "./playgroundLayout"

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
  if (!isComparing)
    return (
      <div className={`${CHAT_COLUMN} flex pt-8 pb-6`}>
        <ChatPanel
          panel={panelA}
          onRegenerate={() => onRegenerate(panelA, setPanelA)}
        />
      </div>
    )

  return (
    <div className="grid flex-1 grid-cols-1 gap-8 pt-8 pb-0 md:grid-cols-2 md:gap-0">
      {(
        [
          { name: "A", panel: panelA, setPanel: setPanelA, other: panelB },
          { name: "B", panel: panelB, setPanel: setPanelB, other: panelA },
        ] as const
      ).map(({ name, panel, setPanel, other }) => {
        const identity = splitModelKey(panel.model)
        return (
          <section
            key={name}
            aria-label={`Model ${name} response`}
            className={
              name === "B"
                ? "flex min-w-0 flex-col gap-2 border-t border-border pt-8 md:border-t-0 md:border-l md:pt-0 md:pl-4"
                : "flex min-w-0 flex-col gap-2 md:pr-4"
            }
          >
            <div className="flex items-center gap-2">
              <span className="w-6 shrink-0 text-center text-mono-caption text-subtle">
                {name}
              </span>
              <ModelSelect
                label={`Model ${name}`}
                value={panel.model}
                models={models}
                pinnedKeys={pinnedKeys}
                onTogglePin={onTogglePin}
                unavailableKeys={other.model ? [other.model] : undefined}
                // A panel's turns belong to the model that produced them, so
                // changing either model drops that column's transcript. B is
                // cleared as well when A takes its model, or its answers would
                // sit under an unchosen picker and be sent to whatever is
                // chosen next.
                onChange={(key) => {
                  setPanel((prev) =>
                    prev.model === key
                      ? prev
                      : { ...prev, model: key, turns: [] },
                  )
                  if (name === "A" && key === panelB.model)
                    setPanelB((prev) => ({ ...prev, model: "", turns: [] }))
                }}
                className="min-w-0 flex-1"
              />
            </div>
            <p className="pl-8 text-caption">
              {identity.instance ||
                (panel.model
                  ? "Model alias"
                  : `Any model other than ${name === "A" ? "B" : "A"}`)}
            </p>
            {panel.turns.length === 0 && !panel.isAwaitingFirstToken ? (
              <div className="flex min-h-40 flex-1 items-center justify-center py-12 pl-8 text-center text-body text-subtle">
                {panel.model
                  ? `${identity.label}’s answer appears here.`
                  : `Choose model ${name} to start comparing.`}
              </div>
            ) : (
              <div className="pt-4">
                <ChatPanel
                  panel={panel}
                  onRegenerate={() => onRegenerate(panel, setPanel)}
                />
              </div>
            )}
          </section>
        )
      })}
    </div>
  )
}
