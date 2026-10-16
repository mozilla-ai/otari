import { Button, Card } from "@heroui/react"
import type { ReactNode } from "react"

// A first-run / empty-list panel: a Card with a heading, a sentence of context,
// and (optionally) a primary call to action. Pages that render a list share this
// so an empty Keys, Users, or Budgets page reads the same way instead of each
// hand-rolling the same Card. `children` slots richer content (e.g. a numbered
// getting-started list) between the copy and the action; omit the action for a
// purely informational empty state (e.g. "no usage yet").
export function EmptyState({
  title,
  description,
  actionLabel,
  onAction,
  isActionDisabled,
  secondaryActionLabel,
  onSecondaryAction,
  children,
}: {
  title: string
  // A plain sentence, rendered in a <p>. Kept to a string so a
  // block element can't land inside that paragraph; richer/blockish content goes
  // through `children`, which renders as a sibling instead.
  description?: string
  actionLabel?: string
  onAction?: () => void
  isActionDisabled?: boolean
  // A quieter alternative beside the primary, as a ghost button.
  secondaryActionLabel?: string
  onSecondaryAction?: () => void
  children?: ReactNode
}) {
  return (
    <Card>
      <Card.Content className="flex flex-col gap-4 p-6">
        <div>
          <h2 className="text-heading">{title}</h2>
          {description ? (
            <p className="mt-1 max-w-prose text-sm text-muted">{description}</p>
          ) : null}
        </div>
        {children}
        {actionLabel && onAction ? (
          <div className="flex flex-wrap items-center gap-3">
            <Button
              variant="primary"
              isDisabled={isActionDisabled}
              onPress={onAction}
            >
              {actionLabel}
            </Button>
            {secondaryActionLabel && onSecondaryAction ? (
              <Button variant="ghost" onPress={onSecondaryAction}>
                {secondaryActionLabel}
              </Button>
            ) : null}
          </div>
        ) : null}
      </Card.Content>
    </Card>
  )
}
