import { Link, type LinkProps } from "@tanstack/react-router"
import { FiArrowLeft } from "react-icons/fi"

/** The way back to the list a detail page was opened from. */
export function BackLink({
  to,
  search,
  label,
}: {
  to: NonNullable<LinkProps["to"]>
  /** The list's own filter, for a page that opens from one view of it. */
  search?: LinkProps["search"]
  label: string
}) {
  return (
    <Link
      to={to}
      search={search}
      className="flex min-h-11 w-fit items-center gap-1.5 text-sm text-muted hover:text-foreground"
    >
      <FiArrowLeft aria-hidden className="size-4" />
      {label}
    </Link>
  )
}
