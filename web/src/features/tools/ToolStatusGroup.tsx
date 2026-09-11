import { useState } from "react"

import type { ManagedTool } from "@/client"
import { CopyableValue } from "@/design-system/actions/CopyField"
import { Dot } from "@/design-system/indicators/Dot"
import { SettingRow } from "@/design-system/layout/SettingRow"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { DisclosureRow } from "@/design-system/navigation/DisclosureRow"
import { DocsLink } from "@/design-system/navigation/DocsLink"
import { settingInputId } from "@/features/tools/ToolSettingRows"

/**
 * The declaration a client sends, as a value to take.
 *
 * Sized to its content rather than filling the lane, since it is a value to
 * take rather than a field to type in; it sits at the lane's leading edge with
 * the controls above it. A frame rather than bare text, because it holds
 * something to be copied verbatim; the value inside it is real selectable text,
 * since the Clipboard API does not exist on the plain-HTTP origins this
 * dashboard is served from.
 */
function DeclarationChip({ tool }: { tool: ManagedTool }) {
  return (
    <span className="inline-flex h-8 items-center border border-control-border bg-surface-alt pl-2.5">
      <CopyableValue
        value={`"type": "${tool.id}"`}
        label={`tools[].type for ${tool.id}`}
        className="font-mono text-xs text-foreground"
      />
    </span>
  )
}

/**
 * What the tool is, whether this deployment can run it, and how to ask for it.
 *
 * At the head of the page because it is the question an operator arrives with:
 * the settings below say how a search runs, and this says whether one runs at
 * all. Collapsed, because the answer is the row and the reasoning is the panel.
 *
 * The panel is the same `SettingRow` grammar as the settings below it, indented
 * to say these rows belong to the tool row that opened them. Eyebrowed
 * paragraphs read as a second system on a page that is otherwise all rows.
 */
export function ToolStatusGroup({
  tool,
  docsHref,
  urlFieldKey,
}: {
  tool: ManagedTool
  docsHref: string
  /** The setting the "no backend" case sends the operator to, when there is one. */
  urlFieldKey?: string
}) {
  const [isOpen, setIsOpen] = useState(false)
  // The provider-named keywords interception adds. Absent when it is off, which
  // is the only thing on this page that changes the list.
  const alsoAccepted = tool.accepted_types.filter((type) => type !== tool.id)

  return (
    <SettingsGroup bounded>
      <DisclosureRow
        label={<code className="text-mono-caption">{tool.id}</code>}
        help={tool.description}
        isOpen={isOpen}
        onToggle={() => setIsOpen((open) => !open)}
        trailing={
          <span className="flex items-center gap-2.5 text-mono-overline text-subtle">
            <Dot className={tool.available ? "bg-success" : "bg-text-subtle"} />
            {tool.available ? "Available" : "Unavailable · no backend"}
          </span>
        }
      >
        <div className="flex flex-col divide-y divide-border-subtle">
          {/* Absent rather than empty when the tool is available: there is no
              reason to give. */}
          {tool.available ? null : (
            <SettingRow
              nested
              label="Why unavailable"
              help="No backend URL is set, so every call is rejected with 400."
              control={
                urlFieldKey ? (
                  <button
                    type="button"
                    onClick={() => {
                      // The field's own `scroll-mt-16` keeps it clear of the
                      // top bar; focus moves separately so the smooth scroll is
                      // not cut short by the browser's own scroll-into-view.
                      const input = document.getElementById(
                        settingInputId(urlFieldKey),
                      )
                      input?.scrollIntoView({
                        behavior: "smooth",
                        block: "start",
                      })
                      input?.focus({ preventScroll: true })
                    }}
                    // The only control in this row's lane, so it carries the
                    // 44px floor itself; a bare `<button>` also keeps Tailwind's
                    // reset cursor without this.
                    className="inline-flex min-h-11 cursor-pointer items-center text-caption text-link transition-colors duration-150 ease-out hover:text-link-hover focus-visible:otari-focus-ring motion-reduce:transition-none"
                  >
                    Set backend URL ↓
                  </button>
                ) : null
              }
            />
          )}
          <SettingRow
            nested
            label="Declare in a request"
            help={
              <>
                Goes in <code className="font-mono">tools[].type</code>.{" "}
                {alsoAccepted.length > 0 ? (
                  <>
                    This deployment also routes{" "}
                    <code className="font-mono">{alsoAccepted.join(", ")}</code>{" "}
                    to it.{" "}
                  </>
                ) : null}
                <DocsLink href={docsHref}>Developer docs</DocsLink>
              </>
            }
            control={<DeclarationChip tool={tool} />}
          />
        </div>
      </DisclosureRow>
    </SettingsGroup>
  )
}
