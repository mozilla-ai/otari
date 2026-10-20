/**
 * Rows an edition adds to the account menu, under "Account settings".
 *
 * Renders nothing in this build: the account menu holds the account, the
 * appearance and the way out, and a deployment that is only itself has nothing
 * more to put in it. A build whose one dashboard reaches several deployments
 * replaces this module at build time and renders what it needs there (which
 * deployment this session is in, and the way to another); that version owns the
 * rows and what they do.
 *
 * The props are `AccountMenuRowsProps` in `./accountMenuRows`, kept off this
 * module so the replacing one can import them.
 *
 * **Reached by its `@/app/nav/overlayAccountMenuRows` specifier and never
 * relatively**, which is the seam rule and not a style call;
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why.
 */

import type { AccountMenuRowsProps } from "./accountMenuRows"

export function AccountMenuRows(_props: AccountMenuRowsProps) {
  return null
}
