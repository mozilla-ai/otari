import type { ReactNode } from "react"

/**
 * Where a build that has a post-sign-in step mounts it, and in this build there
 * is none.
 *
 * The composition seam for anything that has to sit in front of the whole app
 * once a session exists and before the dashboard behind it: a hosted signup
 * asking a few profile and use-case questions after the first sign-in of a new
 * account (otari-ai#1822) is the caller this exists for. A replacement owns the
 * questions, the answers, their storage and whatever it maps them onto; what it
 * could not do before this module was to be shown: of the six seams that predate
 * it, four are nav contributions rendered inside the chrome and two are
 * composition with no surface of their own, so a step that had to precede the
 * dashboard had nowhere to go.
 *
 * The base default renders its children unchanged, and that is the whole of it.
 * An OSS deployment sees the dashboard it saw before, with no step in front of
 * it and nothing asking its operator anything.
 *
 * One export, and its shape is the contract: a component named `PostSignInGate`
 * taking `{ children }`.
 *
 * **The position is the rest of the contract, and all three boundaries are
 * load-bearing.** `AppShell` mounts it inside `EntitlementResolver`, so a
 * replacement may gate its step on a capability; behind the auth gate `App.tsx`
 * puts the router behind, so there is a session to resolve a user from; and
 * above `AppShellChrome`, so a replacement rendering a step replaces the rails
 * with it rather than covering a sidebar the person cannot use.
 *
 * **A replacement gating on a capability reads `isLoading` alongside it.**
 * `useEntitlement` reports `entitled: false` while the axis is still resolving,
 * because it answers from the capabilities resolved so far, and nothing above
 * this seam absorbs that state the way `AppShellChrome` does for the route gate
 * (`answerIsStillComing`). A replacement that treats the first render's answer
 * as final either shows the dashboard and then pulls it out from under someone
 * already using it, or shows a step to an account that was never meant to see
 * one. Hold the *decision* until the axis settles, which is not the same as
 * holding the children back, and is the rule the paragraph below is about.
 *
 * **It must not hold its children back while it decides.** A replacement that
 * renders nothing until its query settles blanks the dashboard on every load,
 * for every user, including the ones who answered months ago. That is the
 * disappointment `overlayEntitlementResolver`'s docstring already warns about on
 * the entitlement axis, and it lands harder here because this seam sits above
 * the chrome as well as the routes. Render children while resolving, and the
 * step only once the answer calls for one.
 *
 * **A replacement owns the telemetry identity and the connectivity feedback
 * while its step is up.** The three app-level components that have no place in
 * the layout (`TelemetryIdentity`, `UpdatePrompt`, `ConnectionStatus`) live
 * inside `AppShellChrome`, so a replacement rendering a step *instead of* its
 * children unmounts all three. Only `TelemetryIdentity` renders nothing
 * unconditionally; the other two render on a condition, a reload pill and a
 * `role="alert"` toast, and both of those conditions can hold while a step is up.
 *
 * Two of the three cost something. `TelemetryIdentity`'s own docstring stops
 * being true for as long as it is unmounted: it is no longer mounted exactly
 * while a session is, and the events a signup step records are the acquisition
 * funnel the telemetry seam exists to carry, so they would land with no actor
 * named. And `ConnectionStatus` is the app-level "the gateway is not answering",
 * which is precisely what someone posting a step's answers needs to be told, so
 * a replacement owns its own submit-failure feedback for as long as the toast is
 * unmounted. `UpdatePrompt` is the one that can wait: a step is short and the
 * stale tab keeps working, which is what that pill already says.
 *
 * Hoisting `TelemetryIdentity` above this seam is deliberately not the fix: it
 * reads `GET /v1/organizations/me`, and the account answering a signup's first
 * questions may hold no membership yet, so the fetch that would name the actor
 * is the one most likely to have nothing to name. A replacement that records
 * during its step identifies the actor itself.
 *
 * **This gateway stores nothing for it.** `models/tenancy.py` says the hosted
 * CRM and onboarding columns "are simply not part of the reconciled schema", and
 * a replacement's query, its step and its record of the answer all live on the
 * side that has them. Nor is `GrowthSignalPort.record_onboarding_completed` the
 * thing being wired here: the answers reach the overlay's own contributed route,
 * so the port stays uncalled for the reason its docstring gives.
 *
 * **Not a change to the setup guide.** `SetupGuideCard`'s docstring argues why
 * this repository shows a panel where the platform shows a blocking sheet, and
 * that argument stands for a self-hosted operator. This seam leaves it alone: a
 * sheet exists only in a build that has a signup to serve.
 *
 * **Reached by its `@/app/overlayPostSignInGate` specifier and never
 * relatively**, which is the seam rule rather than a style call:
 * `overlaySeams.test.ts` enforces it and web/AGENTS.md says why. That specifier
 * is the alias key an `OVERLAY_MODULE_OVERRIDES` entry has to match. The module
 * sits in `src/app/` beside `overlayEntitlementResolver.tsx` because it is
 * composition and not a hook or a piece of chrome.
 */
export function PostSignInGate({ children }: { children: ReactNode }) {
  return <>{children}</>
}
