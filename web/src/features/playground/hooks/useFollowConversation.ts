import { useCallback, useEffect, useRef, useState } from "react"

import { shouldFollowToBottom } from "../helpers/playgroundTurns"

export interface FollowConversation {
  /**
   * Put this on an empty element after the conversation.
   *
   * Typed as React types a ref callback, which is the one place `null` reaches
   * this feature: React passes it on unmount. It is converted on the way in, so
   * nothing below stores it.
   */
  setEndMarker: (node: HTMLDivElement | null) => void
  /** Whether the reader has scrolled up away from the newest content. */
  isJumpVisible: boolean
  jumpToLatest: () => void
}

/**
 * Follow the conversation down, but only when the reader wants it.
 *
 * One scroll position, so one place that decides about it. The page scrolls
 * (the shell's `<main>` is the container), which is why this is a page-level
 * hook and not something each panel does: while comparing, two panels following
 * one scroll position would each scroll it, and the one that ran second would
 * win for reasons nobody could see.
 *
 * **Pinned-ness comes from an IntersectionObserver on the end marker, not from
 * arithmetic on `scrollTop`.** Measuring would mean knowing which ancestor
 * scrolls, which the shell owns and can change; the observer asks the browser
 * whether the bottom of the conversation is on screen, which is the actual
 * question. It also survives the reflow a streaming reply causes, where a
 * threshold on scroll offsets reads a rewrapped paragraph as the reader
 * scrolling away.
 *
 * **The marker arrives through a callback ref, not a `useRef`.** It is rendered
 * only once a conversation exists, so on the welcome screen there is no node:
 * an effect that read a ref on mount found nothing, returned, and never ran
 * again, so nothing was ever observed and the jump control never appeared. A
 * callback ref makes the node itself the trigger.
 */
export function useFollowConversation(params: {
  /** Changes whenever the conversation does. */
  turnCount: number
  /** The role of the newest turn, which decides whether following is wanted. */
  lastRole: string | undefined
}): FollowConversation {
  const { turnCount, lastRole } = params
  const [endMarker, setNode] = useState<HTMLDivElement | undefined>(undefined)
  const setEndMarker = useCallback((node: HTMLDivElement | null) => {
    setNode(node ?? undefined)
  }, [])
  const isPinnedRef = useRef(true)
  const previousCountRef = useRef(turnCount)
  const [isJumpVisible, setIsJumpVisible] = useState(false)

  const jumpToLatest = useCallback(() => {
    endMarker?.scrollIntoView({ block: "end" })
    isPinnedRef.current = true
    setIsJumpVisible(false)
  }, [endMarker])

  useEffect(() => {
    if (!endMarker) {
      // No conversation on screen, so nothing to be away from.
      isPinnedRef.current = true
      setIsJumpVisible(false)
      return
    }
    // jsdom has no IntersectionObserver, and this is a convenience rather than
    // the feature: without it the conversation still follows (the flag stays
    // pinned) and the jump control stays absent.
    if (typeof IntersectionObserver === "undefined") return
    const observer = new IntersectionObserver((entries) => {
      const isPinned = entries.some((entry) => entry.isIntersecting)
      isPinnedRef.current = isPinned
      // The control offers to jump to something, so it appears only once there
      // is somewhere to jump to.
      setIsJumpVisible(!isPinned)
    })
    observer.observe(endMarker)
    return () => observer.disconnect()
  }, [endMarker])

  useEffect(() => {
    const hasNewTurn = turnCount > previousCountRef.current
    previousCountRef.current = turnCount
    if (
      shouldFollowToBottom({
        hasNewTurn,
        lastRole,
        isPinnedToBottom: isPinnedRef.current,
      })
    ) {
      endMarker?.scrollIntoView({ block: "end" })
    }
  }, [turnCount, lastRole, endMarker])

  return { setEndMarker, isJumpVisible, jumpToLatest }
}
