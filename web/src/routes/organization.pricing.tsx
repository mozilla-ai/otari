import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/organization/pricing")({
  // Model pricing folded into Providers, where an organization's models are
  // offered, priced and switched in one place. The old path redirects so
  // bookmarks and any link still pointing at it keep working.
  //
  // Only `override` travels. It named the organization's own rate for a model
  // and still does, so a link carrying one lands on the editor it always meant.
  // `model` named the *deployment's* rate, whose editor this change removed
  // along with the price table it sat in, and the two are different price lists:
  // forwarding it would leave a parameter nothing reads, and translating it to
  // `override` would silently point a deployment rate at a tenant's. Dropped, so
  // the URL says what the page will do.
  //
  // In `beforeLoad`, so the page it replaces is never mounted and the dead path
  // leaves no history entry.
  beforeLoad: ({ search }) => {
    const { override } = search as { override?: string }
    throw redirect({
      to: "/organization/provider-keys",
      search: override ? { override } : {},
      replace: true,
    })
  },
})
