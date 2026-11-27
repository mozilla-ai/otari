import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/tools/mcp-servers/")({
  // The old path of the MCP servers list, from when it was a tab of Tools. The
  // redirect keeps the query string and leaves no history entry. A sign-in
  // that started before the move may still name this path as where to return,
  // so it has to keep landing.
  beforeLoad: ({ search }) => {
    throw redirect({ to: "/mcp-servers", search, replace: true })
  },
})
