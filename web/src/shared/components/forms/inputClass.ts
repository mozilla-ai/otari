/**
 * A bare `<input>` dressed as the product's field, for the places that need an
 * element rather than a HeroUI `TextField` (a grid cell that supplies its own
 * label, a row whose control sits in a column of its own).
 *
 * The same field as HeroUI's, in the same tokens at the same height. Giving it
 * a smaller, rounder treatment of its own puts a second input shape in the
 * product; the only thing this is, is an element rather than a component.
 */
export const INPUT_CLASS =
  // No padding here: `.input` carries it now, so a page cannot spell field
  // padding again and the pagination place can override it in one selector
  // rather than fighting a utility. The height is the form floor; a named place
  // lowers it.
  // No width either: it was `w-full`, which collided with the `w-12` and `w-28`
  // call sites. Two width utilities on one element are settled by Tailwind's
  // emitted order, not by the class string, so which one won was not something
  // a reader could tell from the call site. Every caller that wants full width
  // already says so.
  "input input--primary min-h-10 text-sm"
