import { buttonVariants } from "@heroui/react"
import { Link } from "@tanstack/react-router"
import { FiGitBranch } from "react-icons/fi"

import type { UsageEntry } from "@/client"
import { useRequestTrace } from "@/shared/api/traces"
import { useSurfaces } from "@/shared/hooks/useDeployment"

// Opens the session a request was traced into, on this row's LLM call: the
// gateway names an LLM span after the usage row it records. Nothing shows
// until the session is found, and nothing at all for a request without one.
export function RequestTraceLink({ entry }: { entry: UsageEntry }) {
  const hostsTraces = useSurfaces()("traces")
  const trace = useRequestTrace(entry.request_group_id ?? "", hostsTraces)
  if (!trace.data) return null
  return (
    <Link
      to="/sessions"
      search={{
        peek: trace.data.trace_id,
        peek_workspace: trace.data.workspace_id,
        span: entry.id,
      }}
      className={buttonVariants({ size: "sm", variant: "ghost" })}
    >
      <FiGitBranch aria-hidden="true" className="size-4" />
      View trace
    </Link>
  )
}
