/**
 * The shapes of `overlayRecommenderMargin`, kept off the seam module so the
 * module that replaces it can import them.
 */

export interface RecommenderMarginFieldProps {
  /** The router being edited, or "" for one being added. */
  recommenderId: string
  value: string
  onChange: (value: string) => void
  /** The message `validate` returned for the field, or "" for none. */
  error: string
}

export interface RecommenderMarginErrors {
  margin?: string
}

export interface RecommenderMargin {
  /** What the field starts from for a stored router, or "" for nothing stored. */
  initialValue: (recommenderId: string) => string
  /** Checked beside the router's own fields on submit; empty when the value may be saved. */
  validate: (value: string) => RecommenderMarginErrors
  /**
   * Saves the value for a router once the router itself is saved, and before
   * it is switched on. A rejection leaves the router switched off.
   */
  save: (recommenderId: string, value: string) => Promise<void>
  /** Appended to a router's price in the table, or "" for nothing. */
  summary: (recommenderId: string) => string
}
