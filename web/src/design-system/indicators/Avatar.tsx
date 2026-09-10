/**
 * A person or a workspace, as a monogram.
 *
 * It takes the letters rather than deriving them, which is the one decision in
 * here worth explaining. Turning an identity into two letters is application
 * logic, not presentation: this product falls back from a full name to an email
 * address, splits a name on spaces but an address on its own punctuation
 * (`Ada Lovelace-Byron` initials as AL, `ada.lovelace@…` also as AL), and
 * counts characters rather than code units so a name outside the basic plane
 * does not get half a surrogate pair. `AccountMenu`'s `initialsFor` already
 * carries all of that. A second derivation here would be the same rules,
 * spelled worse, in a layer that has no business knowing an account has an
 * email address at all.
 *
 * No image, and no `src` prop to invite one. This dashboard has no avatar
 * upload, so the only way to fill one would be a third-party image host, which
 * would tell whoever runs it who is signed in to this gateway on every row.
 *
 * The 9px monogram is one of the type scale's two documented off-scale values
 * (typography.md): two letters in a 26px box are recognized rather than read,
 * which no step on the scale does, so `text-shell-monogram` carries it.
 */
export function Avatar({
  initials,
  size = "md",
  className = "",
}: {
  /**
   * Two letters, already derived. Longer is not truncated: a caller that
   * computed three has a bug this component should not hide.
   */
  initials: string
  size?: "sm" | "md"
  /** Position at the call site. Not for a fill or a border of its own. */
  className?: string
}) {
  return (
    <span
      // `aria-hidden`, and not an oversight: a monogram is a picture of a name
      // that is on screen beside it in every place this renders. Announcing
      // "AL" ahead of the row's own "Ada Lovelace" is noise, and labelling it
      // with the full name would read the name twice.
      aria-hidden
      // `font-semibold` resolves to 550 on the variable axis, which is the
      // weight this monogram has always rendered at. Kept as written so the
      // shell's own avatar is unchanged by being moved in here; DESIGN.md's
      // note that 600 renders 550 is why it is not a bug, and why nothing
      // should "fix" it to 600.
      className={`flex shrink-0 items-center justify-center border border-control-border bg-surface-alt text-shell-monogram font-semibold text-muted ${
        size === "sm" ? "h-5 w-5" : "h-[1.625rem] w-[1.625rem]"
      } ${className}`}
    >
      {initials}
    </span>
  )
}
