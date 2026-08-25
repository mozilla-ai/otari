import type { Telemetry } from "@/shared/telemetry/types"

/**
 * The tracker the base build ships, and it records nothing.
 *
 * This dashboard carries no telemetry, and that is the correct state rather
 * than an omission: `web/package.json` declares no analytics vendor, and an OSS
 * gateway that reported to one by default would be a defect. What this module
 * adds is a **seam**, not telemetry. The base default stays a genuine no-op and
 * pulls in no dependency, so an OSS build is free of telemetry code as well as
 * of telemetry traffic.
 *
 * An overlay replaces this module at build time to record the events its own
 * deployment has consent for. The events worth recording are fired from the
 * sign-in screen, the signup and verification pages, and the sidebar, all of
 * which are Otari's files; without this seam an overlay would have to edit them,
 * which ARCHITECTURE.md's "cardinal rules for contributors" rule 6 forbids, or
 * re-implement those pages, which is the duplicate shell this track exists to
 * retire.
 *
 * **Reach it by its `@/…` specifier.** The override is an exact-match alias on
 * `@/shared/telemetry/overlayTelemetry` in the superset build's Vite config
 * (the `ee_else_ce` mechanism; `otari-ai/frontend/vite.config.ts` is the working
 * one). A relative import resolves to a file path that alias never sees, so it
 * would quietly keep this no-op in every edition and the events would go
 * missing with no error anywhere. Same seam pattern as
 * `src/app/nav/overlaySections.ts`, one grain wider: a cross-cutting module
 * rather than a nav contribution.
 *
 * The shape a replacement implements is `types.ts` beside this file, and the
 * names it receives are `events.ts`. Neither is swapped: both sides of the
 * override read one declaration of each, so a replacement cannot drift from the
 * vocabulary its call sites use.
 *
 * Which is why the import below is `@/shared/telemetry/types` and not the
 * `./types` a same-directory sibling would ordinarily be written as. A
 * replacement is a file in the overlay's own tree, not in this directory, so a
 * relative sibling path is the one thing in this module that cannot survive
 * being copied: over there it would resolve against the overlay directory and
 * either fail to resolve or, worse, find something else. Written this way, the
 * file is copyable verbatim.
 */
const NO_TELEMETRY: Telemetry = {
  // Not "denied": nothing has been asked and nothing decided. A build that adds
  // a consent UI is what turns this into an answer.
  consent: "unknown",
  recordEvent: () => undefined,
  identify: () => undefined,
}

/**
 * A hook rather than bare functions, for the two reasons the platform's
 * `useTracking` is one: `consent` can change while the app is running, so a
 * caller reading it has to re-render, and a replacement needs somewhere to hold
 * the client it initializes. The returned object is referentially stable here,
 * and a replacement owes the same: `TelemetryIdentity.tsx` has it in an effect's
 * dependencies, and a fresh object each render would re-identify forever.
 */
export function useTelemetry(): Telemetry {
  return NO_TELEMETRY
}
