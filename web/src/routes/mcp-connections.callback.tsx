import { createFileRoute } from "@tanstack/react-router"

import {
  McpConnectionCallbackPage,
  type McpConnectionCallbackSearch,
} from "@/features/tools/McpConnectionCallbackPage"

function text(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined
}

function Callback() {
  return <McpConnectionCallbackPage search={Route.useSearch()} />
}

// The gateway's callback redirects here (`DASHBOARD_CALLBACK_ROUTE` in
// `services/tools/_mcp_connections.py`), so the path is a contract with it.
export const Route = createFileRoute("/mcp-connections/callback")({
  validateSearch: (
    search: Record<string, unknown>,
  ): McpConnectionCallbackSearch => ({
    state: text(search.state),
    code: text(search.code),
    iss: text(search.iss),
    error: text(search.error),
    connected: text(search.connected),
    return_to: text(search.return_to),
  }),
  component: Callback,
})
