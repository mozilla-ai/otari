/**
 * The names the dashboard records events under.
 *
 * The vocabulary is base-owned and is deliberately **not** part of the seam:
 * `overlayTelemetry.ts` is what a build-time override swaps, and this file is
 * what both sides of that swap read. A replacement that had to restate these
 * names could drift from them silently, and a base call site would then record
 * under a name nothing is listening for.
 *
 * The strings themselves are `otari-ai/frontend/src/shared/helpers/mixpanel.ts`'s
 * `EVENTS`, character for character, because a superset build reports into the
 * same product-analytics workspace the platform already reports into. Renaming
 * one here would split its funnel in two.
 */
export const TELEMETRY_EVENTS = {
  // The acquisition funnel. Every one of these fires from a page in
  // `src/features/auth/`, which is why the seam exists at all: those pages are
  // Otari's, and an overlay may not edit them (ARCHITECTURE.md, "cardinal rules
  // for contributors", rule 6).
  SIGNUP_STARTED: "Signup Started",
  SIGNUP_SUCCESS: "Signup Success",
  SIGNUP_FAILED: "Signup Failed",
  LOGIN_SUCCESS: "Login Success",
  LOGIN_FAILED: "Login Failed",
  LOGOUT: "Logout",
  EMAIL_VERIFICATION_SUCCESS: "Email Verification Success",
  EMAIL_VERIFICATION_FAILED: "Email Verification Failed",
  RESEND_VERIFICATION_CLICKED: "Resend Verification Clicked",
  /**
   * A form refused its own submit before any request went out.
   *
   * Recorded from the sign-in screen, which validates on submit because its
   * button is deliberately never disabled for an empty box (see `Login.tsx`).
   * The signup and recovery forms disable their submit until they are complete,
   * so there is no such moment to record on those, and one is not manufactured.
   */
  FORM_VALIDATION_FAILED: "Form Validation Failed",

  /** A move between destinations in the sidebar. */
  TAB_CHANGED: "Tab Changed",

  /**
   * A wallet top-up abandoned at the payment provider.
   *
   * Named here and fired by nobody in this build, which is the one deliberate
   * gap in this list: billing is ARCHITECTURE.md's canonical overlay-only
   * capability, this gateway meters spend but holds no wallet, and no base page
   * ever sees the return from a checkout. It is named so the overlay that does
   * own that return records it under the platform's existing name rather than
   * inventing a second one.
   */
  CHECKOUT_CANCELLED: "Checkout Cancelled",
} as const
