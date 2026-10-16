import type { ReactNode } from "react"

import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { SettingsGroup } from "@/design-system/layout/SettingsGroup"
import { useRoutingSyncing } from "@/shared/api/routing"

/**
 * One section of a policy's page, with its own Save at its foot.
 *
 * Each section saves alone, writing the rest of the policy back as stored, so
 * a Save says exactly what it is about to change. Read-only, it shows the
 * section without the Save.
 *
 * Every Save waits while any routing write or list refetch is in flight: each
 * section writes the whole policy from the stored copy, so a Save sent before
 * the last one is read back would revert it.
 */
export function PolicyGroup({
  title,
  description,
  saveLabel,
  isDirty = false,
  canSave = true,
  isPending = false,
  error,
  onSave,
  isReadOnly = false,
  children,
}: {
  title: string
  description?: ReactNode
  /** Names the section, so two Saves on one page say which they write. */
  saveLabel?: string
  isDirty?: boolean
  canSave?: boolean
  isPending?: boolean
  error?: unknown
  onSave?: () => void
  isReadOnly?: boolean
  children: ReactNode
}) {
  const isSyncing = useRoutingSyncing()
  return (
    // One element around the group's two bands, so the page's gap falls
    // between groups rather than between a heading and its rows.
    <section aria-label={title}>
      <SettingsGroup title={title} description={description}>
        {/* Full width, so the rule under it spans the band; the measure is
            the inner column's. */}
        <div className="py-4">
          <div className="flex max-w-[40rem] flex-col gap-4">
            <ErrorBanner error={error} />
            {children}
          </div>
        </div>
        {isReadOnly || onSave === undefined ? null : (
          <div className="flex flex-wrap items-center gap-3 py-3">
            <Button
              variant="primary"
              isDisabled={!isDirty || !canSave || isSyncing}
              isPending={isPending}
              onPress={onSave}
            >
              {saveLabel ?? "Save"}
            </Button>
            {isDirty ? (
              <span className="text-caption">Unsaved changes</span>
            ) : null}
          </div>
        )}
      </SettingsGroup>
    </section>
  )
}
