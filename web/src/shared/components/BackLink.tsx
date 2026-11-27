import { Link, type LinkProps } from "@tanstack/react-router"
import type { MouseEventHandler } from "react"
import { FiArrowLeft } from "react-icons/fi"

const BACK_LINK_CLASS =
  "flex min-h-11 w-fit items-center gap-1.5 text-sm text-muted hover:text-foreground"

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
    <Link to={to} search={search} className={BACK_LINK_CLASS}>
      <FiArrowLeft aria-hidden className="size-4" />
      {label}
    </Link>
  )
}

/**
 * The same link for a page that opens ahead of the router, as the public
 * catalog does: a plain anchor that `followInAppLink` hands to the router once
 * one is running.
 */
export function BackAnchor({
  href,
  onClick,
  label,
}: {
  href: string
  onClick?: MouseEventHandler<HTMLAnchorElement>
  label: string
}) {
  return (
    <a href={href} onClick={onClick} className={BACK_LINK_CLASS}>
      <FiArrowLeft aria-hidden className="size-4" />
      {label}
    </a>
  )
}
