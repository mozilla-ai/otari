/**
 * The mark on the account control that ends the sidebar.
 *
 * This build draws the signed-in person's avatar: the face they picked on the
 * Account page, else the one their user id gives them, so everybody has one
 * from the first paint. Until the identity has loaded, or when there is none,
 * it draws the neutral "··" monogram, so a wrong face never flashes in. A build that has something more
 * telling to put there (which of several deployments this dashboard is signed
 * in to, say) replaces this module at build time and draws that instead; the
 * base control keeps the name and the menu either way.
 *
 * `useAccountBadgeLabel` is the words for what the mark shows, folded into the
 * control's accessible name after the person's name, or `""` when the mark
 * adds nothing a screen reader should hear. The avatar is decoration beside a
 * name that is already in that label, so this build answers `""`.
 *
 * **Reached by its `@/app/nav/overlayAccountBadge` specifier and never
 * relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why.
 */

import { Avatar } from "@/design-system/indicators/Avatar"
import { AvatarFace } from "@/features/account/avatar/AvatarFace"
import { useAvatar } from "@/features/account/avatar/avatarChoice"
import { useSessionCaller } from "@/shared/api/auth"

export function AccountBadge() {
  const avatar = useAvatar(useSessionCaller()?.user_id)
  if (avatar === null) return <Avatar initials="··" />
  // The monogram's own 26px box, so the control does not move when it swaps.
  return <AvatarFace id={avatar} className="h-[1.625rem] w-[1.625rem]" />
}

export function useAccountBadgeLabel(): string {
  return ""
}
