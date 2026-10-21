/**
 * What an edition adds to the price of a subagent recommendation.
 *
 * Renders nothing and saves nothing in this build: a recommendation is charged
 * the decision model's tokens at the rates on the router, and nothing on top.
 * An edition that charges more replaces this module at build time, with a
 * field the router form renders after the output price and the store behind
 * it; that version owns the value and where it is kept, so the form here
 * holds it only between the field and `save`.
 *
 * The form submits in three steps so the two writes cannot leave a router
 * serving without what the edition adds: the router is saved switched off,
 * then `save` runs, then the router is switched on if it is meant to be. A
 * `save` that rejects leaves the router off and shows the error. The props and
 * the hook's shape are in `./recommenderMargin`, kept off this module so the
 * replacing one can import them.
 *
 * **Reached by its `@/features/routing/overlayRecommenderMargin` specifier and
 * never relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why.
 */

import type {
  RecommenderMargin,
  RecommenderMarginFieldProps,
} from "./recommenderMargin"

const NO_MARGIN: RecommenderMargin = {
  initialValue: () => "",
  validate: () => ({}),
  save: async () => {},
  summary: () => "",
}

export function RecommenderMarginField(_props: RecommenderMarginFieldProps) {
  return null
}

export function useRecommenderMargin(): RecommenderMargin {
  return NO_MARGIN
}
