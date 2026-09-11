import { Button as HeroButton } from "@heroui/react"
import type { ComponentProps } from "react"

/**
 * The product's three button variants, and no others.
 *
 * This is the whole reason the wrapper exists. HeroUI's own union still accepts
 * `secondary`, `tertiary`, `outline` and `danger-soft`, all four of which are
 * retired here, and none of which is a type error against it: they compile,
 * lint, ship, and paint an unstyled button. `src/styles/foundation.test.ts`
 * catches that by scanning the source for the retired names, which works but
 * reports it at the end of a test run rather than in the editor. Narrowing the
 * union turns the same rule into a compile error at the call site.
 *
 * `variant="ghost"` is the default because it is the answer for everything that
 * is not the one thing a band exists to do. A call site that wants the accent
 * has to ask for it, which is the direction the "one primary per band" rule
 * wants the friction to point.
 */
export type ButtonVariant = "primary" | "ghost" | "danger"

/** `sm` 32px, `md` 36px (the default), `lg` 40px. */
export type ButtonSize = "sm" | "md" | "lg"

type HeroButtonProps = ComponentProps<typeof HeroButton>

export type ButtonProps = Omit<
  HeroButtonProps,
  // `variant` and `size` are narrowed below. `color` is HeroUI v2's API and
  // does nothing in v3, so it is closed rather than left to look available.
  "variant" | "size" | "color"
> & {
  variant?: ButtonVariant
  size?: ButtonSize
}

/**
 * An action. Navigation that looks like a button is a `Link`; an action inside a
 * table row is a `RowAction`.
 *
 * `isPending` while a mutation is in flight disables the button and shows its
 * own spinner, so it must not be paired with `isDisabled` for the same
 * condition: two sources for one state is how a button ends up disabled with no
 * spinner.
 */
export function Button({ variant = "ghost", ...rest }: ButtonProps) {
  return <HeroButton variant={variant} {...rest} />
}
