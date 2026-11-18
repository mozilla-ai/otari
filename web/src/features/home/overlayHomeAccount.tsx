import type { ReactNode } from "react"

/**
 * What an edition that holds the organization's money adds to an owner's
 * Home: a tile in the figures, a row under Needs attention, and a clause in
 * the line that says nothing does.
 *
 * Inert in this build, and meant to stay so: this gateway meters spend but
 * holds no balance, so there is no tile, no row and no clause, and Home shows
 * its Failures figure where the tile would go. A build that holds a balance
 * replaces this module at build time (otari.ai shows the prepaid balance, the
 * days it lasts at this week's pace, and Add funds), and owns every read and
 * word that takes, so Home holds no vocabulary for a capability it lacks.
 *
 * **Reached by its `@/features/home/overlayHomeAccount` specifier and never
 * relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why.
 */
export interface HomeAccount {
  /**
   * Whether what the edition reads has answered. Needs attention waits for it
   * as for its own reads, and names `sourceName` when it failed.
   */
  state: "ready" | "loading" | "failed"
  /** The source as a sentence names it: "The balance". */
  sourceName: string
  /** A figure among Home's, in place of Failures. */
  tile?: {
    label: string
    value: string
    subline: string
    /** Its one action, such as a link to add funds. */
    action?: ReactNode
  }
  /** A row for Needs attention, ranked ahead of every row Home makes. */
  attention?: {
    id: string
    isUrgent: boolean
    text: string
    /** Its one action, as a rendered control. */
    action: ReactNode
  }
  /** A clause for the line that says nothing needs anyone: "the balance tops up on its own". */
  quietClause?: string
}

export const NO_HOME_ACCOUNT: HomeAccount = { state: "ready", sourceName: "" }

export function useHomeAccount(): HomeAccount {
  return NO_HOME_ACCOUNT
}
