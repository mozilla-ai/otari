import { API_ROOT } from "@/shared/api/client"
import { bootstrap, organizationContext } from "@/tests/fixtures"

// Throwaway: lets `pnpm demo` run the real shell with no gateway behind it.
const paged = () => ({ data: [], total: 0, count: 0 })
const handlers: Record<string, () => unknown> = {
  "/organizations/me/memberships": paged,
  "/organizations/me/pending-memberships": paged,
  "/budgets": paged,
  "/providers": () => ({ providers: [] }),
  "/settings": () => ({
    config: [],
    default_pricing: false,
    master_key_source: "configured",
    mode: "standalone",
  }),
  "/bootstrap": () => bootstrap(),
  "/organizations/me": () =>
    organizationContext({ deployment_operator: true } as never),
}

export function installMockGateway(): void {
  const realFetch = window.fetch.bind(window)
  window.fetch = async (input, init) => {
    const url = new URL(
      typeof input === "string" || input instanceof URL
        ? String(input)
        : input.url,
      window.location.origin,
    )
    if (!url.pathname.startsWith(API_ROOT)) return realFetch(input, init)
    const path = url.pathname.slice(API_ROOT.length)
    const handler = handlers[path]
    if (!handler) {
      console.warn("[mockGateway] unhandled", path)
      return new Response(JSON.stringify({ detail: "mock: not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      })
    }
    return new Response(JSON.stringify(handler()), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  }
}
