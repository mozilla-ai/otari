import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/organization/pricing")({
  // The retired path for organization pricing. Providers is where an
  // organization's models are offered, priced and switched, so a bookmark or a
  // link still pointing here lands there.
  //
  // Only `override` travels. It names the organization's own rate for a model,
  // which is what the destination's editor opens on. `model` names the
  // *deployment's* rate, a different price list with no editor on this page, so
  // forwarding it would leave a parameter nothing reads and translating it would
  // point a deployment rate at a tenant's. Dropped, so the URL says what the
  // page will do.
  //
  // In `beforeLoad`, so the page it replaces is never mounted and the dead path
  // leaves no history entry.
  beforeLoad: ({ search }) => {
    // Read rather than cast: this route declares no schema, so `search` is
    // whatever the URL carried, and a non-string `override` would otherwise be
    // handed to the destination as one.
    const override = (search as Record<string, unknown>).override
    throw redirect({
      to: "/organization/provider-keys",
      search: typeof override === "string" && override ? { override } : {},
      replace: true,
    })
  },
})
