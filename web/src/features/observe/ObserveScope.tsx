import { useId } from "react"

import { Tab, TabRow } from "@/design-system/navigation/TabRow"
import { useSelectedWorkspace } from "@/shared/hooks/SelectedWorkspace"
import { type ObserveScope, useObserveScope } from "./useObserveScope"

/**
 * The Mine | Everyone switch. Mine is off for a caller with no usage identity,
 * and `reasonId` names the visible text saying so.
 */
export function ObserveScopeSwitch({ reasonId = "" }: { reasonId?: string }) {
  const { scope, setScope, isMineUnavailable } = useObserveScope()
  const tab = (value: ObserveScope, label: string) => {
    const isOff = value === "mine" && isMineUnavailable
    return (
      <Tab
        isActive={scope === value}
        onPress={() => setScope(value)}
        isDisabled={isOff}
        describedBy={isOff ? reasonId : ""}
      >
        {label}
      </Tab>
    )
  }
  return (
    <TabRow>
      {tab("mine", "Mine")}
      {tab("everyone", "Everyone")}
    </TabRow>
  )
}

/**
 * The switch, then the sentence it changes: whose requests the page is showing,
 * and what counts as "mine". At the start of the line, under the title, because
 * it decides every number on the page.
 */
export function ObserveScopeCaption() {
  const { scope, isMineUnavailable, isOwnerUnknown } = useObserveScope()
  const reasonId = useId()
  const { selected: workspace } = useSelectedWorkspace()
  const where = workspace?.name ?? ""
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pb-4">
      <ObserveScopeSwitch reasonId={reasonId} />
      <div className="text-sm text-muted">
        <p>
          {scope === "mine"
            ? `Your requests in ${where || "this organization"}`
            : where
              ? `Everyone in ${where}`
              : "Everyone on this gateway"}
        </p>
        {scope === "mine" ? (
          <p className="text-caption">
            Requests made with a shared key count toward the key&apos;s owner.
          </p>
        ) : null}
        {isMineUnavailable ? (
          <p id={reasonId} className="text-caption">
            {isOwnerUnknown
              ? "Your usage identity could not be read, so Mine is off."
              : "Your account has no usage identity in this organization yet, so Mine is off."}
          </p>
        ) : null}
      </div>
    </div>
  )
}
