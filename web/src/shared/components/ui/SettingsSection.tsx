import type { ReactNode } from "react"

interface SettingsSectionProps {
  title: string
  description?: string
  /** Optional right-aligned actions (e.g. a button or link) rendered beside the title. */
  actions?: ReactNode
  children: ReactNode
}

/**
 * Shared chrome for a settings page section: a header row with a semibold
 * title, optional muted description, and optional right-aligned actions,
 * followed by the section body.
 */
export const SettingsSection = ({
  title,
  description,
  actions,
  children,
}: SettingsSectionProps) => {
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-start justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h2 className="text-base font-semibold">{title}</h2>
          {description && <p className="text-sm text-muted">{description}</p>}
        </div>
        {actions && <div className="shrink-0">{actions}</div>}
      </div>
      {children}
    </section>
  )
}
