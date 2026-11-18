import { Link } from "@tanstack/react-router"

import { EmptyState } from "@/design-system/feedback/EmptyState"
import { PageError } from "@/design-system/feedback/PageError"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { useDeploymentOperator } from "@/shared/api/organizations"

import type { PlaygroundGate } from "./helpers/playgroundView"

/**
 * What stands in for the Playground when it cannot run yet.
 *
 * Three distinct answers rather than one "not available", because each sends
 * somebody somewhere different. A failed catalog read is not fixed by adding a
 * provider, so it says so and offers no link; no workspace is an operator's to
 * fix and not the reader's; no models is the one case where there is a page to
 * go to and something to do on it: the deployment's own keys for an operator,
 * the organization's for anyone else.
 */
export function PlaygroundGateNotice({
  gate,
  error,
}: {
  gate: Exclude<PlaygroundGate, "ready">
  /** The catalog read's error, for the one gate that has one. */
  error?: unknown
}) {
  const { isOperator } = useDeploymentOperator()

  if (gate === "loading") {
    return <PageLoading label="Loading the Playground" />
  }

  if (gate === "catalogError") {
    return <PageError error={error} />
  }

  if (gate === "readOnly") {
    return (
      <EmptyState
        title="The Playground is not open to viewers"
        description="A Playground request spends like an API key does, and a viewer has read-only access. Ask an owner or admin for a member role to use it."
      />
    )
  }

  if (gate === "noWorkspace") {
    return (
      <EmptyState
        title="No workspace to chat in"
        description="The Playground runs in a workspace, and this account is not a member of one yet. Ask an operator to add you."
      />
    )
  }

  return (
    <EmptyState
      title="No models to chat with"
      description="This deployment serves no chat models yet. Add a provider credential, and the models it offers appear here."
    >
      <p className="text-body text-muted">
        Start on the{" "}
        <Link
          to={
            isOperator
              ? "/admin/hosted-providers"
              : "/organization/provider-keys"
          }
          className="font-medium text-link hover:text-link-hover"
        >
          {isOperator ? "Hosted providers" : "Providers"}
        </Link>{" "}
        page.
      </p>
    </EmptyState>
  )
}
