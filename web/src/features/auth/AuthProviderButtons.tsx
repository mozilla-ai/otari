import type { ReactNode } from "react"
import type { IconType } from "react-icons"
import { Button } from "@/design-system/actions/Button"
import {
  OAUTH_PROVIDER_ICONS,
  OAUTH_PROVIDER_LABELS,
  type OAuthProvider,
} from "./oauthProviders"

/**
 * The pieces of an OAuth-first public auth card, shared by the pages that offer
 * one: the provider rows, the rule that separates them from the address form,
 * and the row that opens that form.
 *
 * Rows rather than a pair of side-by-side buttons while the address form is
 * folded away, because a person scans the card for the account they hold before
 * they read anything else, and a full-width row names it in words. Once the form
 * is open the same buttons shrink to a two-up row so they stay in view without
 * pushing the fields down.
 */

const ROW =
  "relative h-11 w-full min-w-0 justify-center border border-border-strong px-[15px]"

/**
 * A rule with the word on it: these are alternatives, not a second step.
 *
 * 12px of air on each side by default. The sign-in card draws it with 16 above
 * and 12 below, because the rows above it are a fixed block there and do not
 * change when the form opens.
 */
export function AuthOrRule({ className = "py-3" }: { className?: string }) {
  return (
    <div
      className={`flex items-center gap-3 text-xs leading-[18px] text-muted ${className}`}
      aria-hidden
    >
      <span className="h-px flex-1 bg-border" />
      or
      <span className="h-px flex-1 bg-border" />
    </div>
  )
}

/**
 * A full-width row with a mark pinned to its left edge and the label centered.
 *
 * The mark is absolutely positioned so the label stays centered on the row
 * whatever the width of the glyph, and the row keeps the 44px touch floor.
 */
export function AuthMethodRow({
  icon: Icon,
  children,
  ...rest
}: {
  icon: IconType
  children: ReactNode
} & Omit<React.ComponentProps<typeof Button>, "children" | "variant">) {
  return (
    <Button variant="ghost" className={ROW} {...rest}>
      <Icon
        aria-hidden
        className={`absolute inset-y-0 left-[15px] my-auto size-4`}
      />
      {children}
    </Button>
  )
}

export function AuthProviderRows({
  providers,
  layout,
  verb,
  pendingProvider,
  isDisabled,
  onSelect,
}: {
  providers: readonly OAuthProvider[]
  /** `stacked` while the address form is folded away, `two-up` once it is open. */
  layout: "stacked" | "two-up"
  verb: "Sign up" | "Sign in"
  /** The provider whose consent screen is being fetched, if any. */
  pendingProvider: string | undefined
  isDisabled: boolean
  onSelect: (provider: OAuthProvider) => void
}) {
  const isTwoUp = layout === "two-up"
  return (
    <div
      className={isTwoUp ? "flex gap-3" : "flex flex-col gap-3"}
      aria-busy={pendingProvider !== undefined}
    >
      {providers.map((provider) => {
        const Mark = OAUTH_PROVIDER_ICONS[provider]
        const label = OAUTH_PROVIDER_LABELS[provider]
        const isRedirecting = pendingProvider === provider
        return (
          <Button
            key={provider}
            type="button"
            variant="ghost"
            isDisabled={isDisabled}
            onPress={() => onSelect(provider)}
            aria-label={
              isRedirecting ? "Redirecting…" : `${verb} with ${label}`
            }
            className={`${
              isTwoUp
                ? "h-11 min-w-0 flex-1 justify-center gap-2 border border-border-strong"
                : ROW
            }${pendingProvider !== undefined ? " pointer-events-none" : ""}`}
          >
            {isRedirecting ? null : (
              <Mark
                aria-hidden
                className={
                  isTwoUp
                    ? "size-4 shrink-0"
                    : "absolute inset-y-0 left-[15px] my-auto size-4"
                }
              />
            )}
            {isRedirecting
              ? "Redirecting…"
              : isTwoUp
                ? label
                : `${verb} with ${label}`}
          </Button>
        )
      })}
    </div>
  )
}
