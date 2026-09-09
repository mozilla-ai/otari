/**
 * A hairline between two things.
 *
 * The flat plane's only division, which is why this is a component rather than
 * a `border-t` at a call site: `colors.md` gives the border family three rungs
 * with different jobs (3% inside a group, 6% between bands, 10% for a division
 * that has to read as structural), and spelling one inline is how a page ends
 * up with a rung chosen by whichever line somebody copied.
 *
 * A `Section`'s own rules are not this: a band draws its own edges, and
 * layout.md covers that. This is for a line inside one.
 */
export function Divider({
  weight = "default",
  orientation = "horizontal",
  className = "",
}: {
  /** `subtle` inside one group, `default` between them, `strong` for structure. */
  weight?: "subtle" | "default" | "strong"
  orientation?: "horizontal" | "vertical"
  className?: string
}) {
  const ink =
    weight === "subtle"
      ? "border-border-subtle"
      : weight === "strong"
        ? "border-border-strong"
        : "border-border"
  return (
    <span
      // Decoration, and deliberately not `role="separator"`. A rule between two
      // sections that are already separate elements adds nothing to the
      // accessibility tree, and announcing it interrupts the reading of what it
      // divides. A separator that carries meaning is a case for a landmark or a
      // heading, not for a louder line.
      aria-hidden
      className={`${
        orientation === "horizontal"
          ? `block w-full border-t ${ink}`
          : `block self-stretch border-l ${ink}`
      } ${className}`}
    />
  )
}
