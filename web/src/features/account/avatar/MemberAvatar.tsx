import { AvatarFace } from "./AvatarFace"
import { avatarFace } from "./avatarChoice"

/**
 * A person's avatar in a list of people: the face they picked (`face`, the
 * member row's `avatar_face`), else the one their user id gives them.
 *
 * A row with no identity yet (an invitation not accepted) gets an empty square
 * of the same size, so names stay on one lane down the list.
 */
export function MemberAvatar({
  userId,
  face,
}: {
  userId: string | null | undefined
  face: number | null | undefined
}) {
  // The account control's own 26px box, so a person reads the same in the
  // sidebar and in the roster.
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
