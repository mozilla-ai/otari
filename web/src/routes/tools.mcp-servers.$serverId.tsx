import { createFileRoute, redirect } from "@tanstack/react-router"

export const Route = createFileRoute("/tools/mcp-servers/$serverId")({
  // The old path of one server's page. The query carries the workspace a trace
  // opened it for, so it travels with the redirect.
  validateSearch: (
    search: Record<string, unknown>,
  ): { workspace?: string } => ({
    workspace:
      typeof search.workspace === "string" && search.workspace !== ""
        ? search.workspace
        : undefined,
  }),
  beforeLoad: ({ params, search }) => {
    throw redirect({
      to: "/mcp-servers/$serverId",
      params: { serverId: params.serverId },
      search,
      replace: true,
    })
  },
})
