/**
 * The mark on the account control that ends the sidebar.
 *
 * This build draws the signed-in person's monogram, which is what the design's
 * account-menu artboard shows. A build that has something more telling to put
 * there (which of several deployments this dashboard is signed in to, say)
 * replaces this module at build time and draws that instead; the base control
 * keeps the name and the menu either way.
 *
 * `useAccountBadgeLabel` is the words for what the mark shows, folded into the
 * control's accessible name after the person's name, or `""` when the mark
 * adds nothing a screen reader should hear. The monogram is a picture of the
 * name already in that label, so this build answers `""`.
 *
 * **Reached by its `@/app/nav/overlayAccountBadge` specifier and never
 * relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why.
 */

import { Avatar } from "@/design-system/indicators/Avatar"

export function AccountBadge({ initials }: { initials: string }) {
  return <Avatar initials={initials} />
}

export function useAccountBadgeLabel(): string {
  return ""
}
