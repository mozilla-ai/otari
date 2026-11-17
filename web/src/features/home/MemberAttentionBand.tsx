import { Link } from "@tanstack/react-router"
import { Skeleton } from "@/design-system/feedback/Skeleton"
import type { MemberAttention } from "@/features/home/memberHomeModel"
import { Row, severityOf } from "@/features/home/TeamAttentionBand"

const ACTION =
  "shrink-0 text-sm text-link underline max-sm:inline-flex max-sm:min-h-11 max-sm:items-center"

/**
 * A member's Needs attention: the owner's band, over their own requests and
 * keys. Quiet, it is one sentence under the heading.
 */
export function MemberAttentionBand({
  attention,
}: {
  attention: MemberAttention
}) {
  return (
    <section aria-label="Needs attention" className="flex flex-col gap-2">
      <h2 className="text-overline">Needs attention</h2>
      {attention.isLoading ? (
        <Skeleton className="h-5 w-[28rem] max-w-full" />
      ) : attention.quiet ? (
        <p className="text-sm text-muted">{attention.quiet}</p>
      ) : (
        <ul className="flex flex-col border-t border-border">
          {attention.items.map((item) => (
            <Row
              key={item.id}
              severity={severityOf(item.isUrgent)}
              action={
                <Link
                  to={item.action.to}
                  search={item.action.search}
                  className={ACTION}
                >
                  {item.action.label}
                </Link>
              }
            >
              {item.text}
            </Row>
          ))}
          {attention.notChecked.map((text) => (
            <Row key={text} dot="bg-text-subtle">
              {text}
            </Row>
          ))}
        </ul>
      )}
    </section>
  )
}
