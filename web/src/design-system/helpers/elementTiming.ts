/**
 * The Element Timing opt-in for one element, as props to spread onto it, or
 * nothing when `identifier` is absent.
 *
 * Spread rather than written as a JSX attribute because `@types/react` does not
 * declare `elementtiming` yet, and React passes it to the DOM as written either
 * way. A global augmentation of React's types would do the same job, but only
 * in a program that happens to include the file declaring it, and the superset
 * build's typecheck does not include every file this one does.
 */
export function elementTiming(identifier?: string): Record<string, string> {
  return identifier ? { elementtiming: identifier } : {}
}
