import { FiArrowUpRight } from "react-icons/fi"

/**
 * A link out to the documentation, at the end of the sentence it explains.
 *
 * Always external: `href` names a deployment's own documentation site or a
 * document rendered outside this app, never a route the router owns. The arrow
 * is the external mark, so it is `aria-hidden` and the name stays the word.
 */
export function DocsLink({
  href,
  children = "Docs",
}: {
  href: string
  children?: string
}) {
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="inline-flex items-center gap-0.5 text-link transition-colors duration-150 ease-out hover:text-link-hover focus-visible:otari-focus-ring motion-reduce:transition-none"
    >
      {children}
      <FiArrowUpRight aria-hidden="true" className="h-3 w-3" />
    </a>
  )
}
