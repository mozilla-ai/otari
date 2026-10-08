/**
 * What `overlayAccountMenuRows` is handed, off the seam so a build that
 * replaces it can import the props; see `shared/api/requestPolicy.ts` for why.
 */

export interface AccountMenuRowsProps {
  /** Closes the account menu, for a row that leaves the page or opens another surface. */
  closeMenu: () => void
}
