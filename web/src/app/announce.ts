/**
 * Say something to a screen reader without a live region on the page.
 *
 * React Aria's announcer keeps one polite region for the whole document, so a
 * message is read once, when it is sent, rather than every time a region's text
 * changes. The boot screen needs exactly that: its card updates after every
 * background attempt, and reading each update aloud would be chatter. Only
 * `react-aria`'s private entry exports it in this version, the same kind of
 * entry `@heroui/react` reaches into for collections.
 *
 * Its own module so a test can replace it.
 */
import { announce as ariaAnnounce } from "react-aria/private/live-announcer/LiveAnnouncer"

export function announce(message: string): void {
  ariaAnnounce(message, "polite")
}
