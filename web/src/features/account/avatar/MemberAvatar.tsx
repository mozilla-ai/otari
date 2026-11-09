import { AvatarFace } from "./AvatarFace"
import { useAvatar } from "./avatarChoice"

/**
 * A person's avatar in a list of people: the face their user id gives them, or
 * the one they picked if the pick was made in this browser (picks are not on
 * the profile yet, so everyone else's row shows their assigned face).
 *
 * A row with no identity yet (an invitation not accepted) gets an empty square
 * of the same size, so names stay on one lane down the list.
 */
export function MemberAvatar({
  userId,
}: {
  userId: string | null | undefined
}) {
  const id = useAvatar(userId ?? undefined)
  // The account control's own 26px box, so a person reads the same in the
  // sidebar and in the roster.
  const box = "h-[1.625rem] w-[1.625rem]"
  if (id === null) {
    return (
      <span
        aria-hidden
        className={`block shrink-0 border border-border bg-surface-alt ${box}`}
      />
    )
  }
  return <AvatarFace id={id} className={box} />
}
