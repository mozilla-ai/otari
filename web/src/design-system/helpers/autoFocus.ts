/**
 * Whether a text field may take focus by itself when it appears.
 *
 * Not on a touch screen. Focusing a field there raises the software keyboard,
 * which covers half the screen and shoves a sheet that has just opened up the
 * page before its reader has seen what it asks. A keyboard user still lands in
 * the field. Keyed on "coarse" rather than on "fine" so an environment where no
 * media query matches, a test runner included, keeps the desktop behavior.
 */
export function allowsFieldAutoFocus(): boolean {
  return !(
    typeof window !== "undefined" &&
    window.matchMedia?.("(pointer: coarse)").matches
  )
}
