import type { ReactNode } from "react"

/**
 * What a failed Playground turn offers under its error text.
 *
 * Renders nothing in this build: the error text already names the condition,
 * and a gateway with no wallet has no next step to put beside it. A build that
 * can act on a refusal replaces this module at build time and answers the
 * refusal it recognises (otari.ai offers to add funds on an
 * `insufficient_funds` 402), rendering nothing for every other one.
 *
 * `status` is the HTTP status and `code` the error `code` from the refusal's
 * body, each null when the turn failed some other way (a fault mid-stream, the
 * gateway unreachable). Key on `code` before `status`: a 402 is not always a
 * wallet, and a code is what says which refusal it was.
 *
 * **Reached by its `@/features/playground/overlayErrorAction` specifier and
 * never relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why. That specifier
 * is the alias key an `OVERLAY_MODULE_OVERRIDES` entry has to match.
 */
export function PlaygroundErrorAction(_props: {
  status: number | null
  code: string | null
}): ReactNode {
  return null
}
