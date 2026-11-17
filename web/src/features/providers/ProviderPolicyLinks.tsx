import { FiExternalLink } from "react-icons/fi"

import type { CatalogProviderSummary } from "@/client"

const LINK =
  "relative z-10 inline-flex min-h-11 items-center gap-1 text-sm text-link hover:text-link-hover md:min-h-0"

/**
 * A provider's Terms of service and Privacy policy, where the operator named
 * them. Nothing when it named neither, rather than a line saying so: the
 * absence is the operator's to fix, not the reader's to be told about.
 *
 * `nofollow` because the deployment publishes these links on the operator's
 * word, not as an endorsement it vouches for.
 */
export function ProviderPolicyLinks({
  provider,
}: {
  provider: Pick<
    CatalogProviderSummary,
    "display_name" | "terms_url" | "privacy_url"
  >
}) {
  const links = [
    { href: provider.terms_url, label: "Terms of service" },
    { href: provider.privacy_url, label: "Privacy policy" },
  ].filter((link): link is { href: string; label: string } => !!link.href)
  if (links.length === 0) return null
  return (
    <span className="flex flex-wrap items-center gap-x-4 gap-y-1">
      {links.map((link) => (
        <a
          key={link.label}
          href={link.href}
          target="_blank"
          rel="noopener noreferrer nofollow"
          aria-label={`${provider.display_name} ${link.label} (opens in a new tab)`}
          className={LINK}
        >
          {link.label}
          <FiExternalLink aria-hidden className="size-3.5 shrink-0" />
        </a>
      ))}
    </span>
  )
}
