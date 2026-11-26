import { AvatarFace } from "./AvatarFace"
import { avatarFace } from "./avatarChoice"

/**
 * A person's avatar in a list of people: the face they picked (`face`, the
 * member row's `avatar_face`), else the one their user id gives them.
 *
 * A row with no identity yet (an invitation not accepted) gets an empty square
 * of the same size, so names stay on one lane down the list. `isCompact` is
 * the 20px face a table cell shows beside a name, and draws nothing for a row
 * with no identity: a cell has no lane of names to hold.
 */
export function MemberAvatar({
  userId,
  face,
  isCompact = false,
}: {
  userId: string | null | undefined
  face: number | null | undefined
  isCompact?: boolean
}) {
  if (isCompact) {
    return userId ? (
      <AvatarFace id={avatarFace(userId, face)} className="size-5 shrink-0" />
    ) : null
  }
  const box = "h-[1.625rem] w-[1.625rem]"
  if (!userId) {
    return (
      <span
        aria-hidden
        className={`block shrink-0 border border-border bg-surface-alt ${box}`}
      />
    )
  }
  return <AvatarFace id={avatarFace(userId, face)} className={box} />
}
