import { useState } from "react"
import { FiArrowRight, FiCheck, FiCpu, FiGrid, FiShield } from "react-icons/fi"
import type { Workspace } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { ComboBoxField } from "@/design-system/forms/ComboBoxField"
import { ProductMark } from "@/design-system/ProductMark"
import { SCOPE_INFO } from "@/features/keys/scopes"
import {
  useAuthorizeOAuthApp,
  useOAuthApp,
  useOAuthAppWorkspaces,
} from "@/shared/api/oauthApps"
import { LoginPageShell } from "./LoginPageShell"

/** Consent stays behind the dashboard's normal session gate. */
export function AppAuthorizationPage() {
  const params = new URLSearchParams(window.location.hash.split("?")[1] ?? "")
  const [workspace, setWorkspace] = useState<Workspace>()
  const [search, setSearch] = useState("")
  const app = useOAuthApp(
    params.get("client_id") ?? "",
    params.get("redirect_uri") ?? "",
  )
  const workspaces = useOAuthAppWorkspaces(
    search,
    Boolean(app.data) && !app.error,
  )
  const approval = useAuthorizeOAuthApp(params)
  const choices = workspaces.data?.data ?? []
  const soleWorkspace =
    !search && !workspaces.isPlaceholderData && workspaces.data?.count === 1
      ? choices[0]
      : undefined
  const hasCompleteChoices =
    !search &&
    !workspaces.isPlaceholderData &&
    workspaces.data?.count === choices.length
  const selectedWorkspace =
    soleWorkspace ??
    (hasCompleteChoices
      ? choices.find((choice) => choice.id === workspace?.id)
      : workspace)
  // Keep a chosen label after an empty search returns to the first page.
  const options =
    selectedWorkspace &&
    !search &&
    !choices.some((choice) => choice.id === selectedWorkspace.id)
      ? [...choices, selectedWorkspace]
      : choices
  const failure = app.error ?? workspaces.error ?? approval.error
  const loading =
    (app.isPending && !app.data) ||
    (Boolean(app.data) && workspaces.isPending && !workspaces.data)
  const deny = () => {
    if (!app.data || app.error) return
    const url = new URL(app.data.redirect_uri)
    url.searchParams.set("error", "access_denied")
    url.searchParams.set("state", params.get("state") ?? "")
    url.searchParams.set("iss", app.data.issuer)
    window.location.assign(url.toString())
  }

  return (
    <LoginPageShell isBusy={approval.isPending}>
      <div
        aria-hidden="true"
        className="flex items-center justify-center gap-4"
      >
        <div className="flex size-12 items-center justify-center border border-border-strong bg-background">
          <ProductMark className="h-auto w-7 text-accent" />
        </div>
        <div className="flex items-center gap-2 text-subtle">
          <span className="h-px w-4 bg-border-strong" />
          <FiArrowRight className="size-4" />
          <span className="h-px w-4 bg-border-strong" />
        </div>
        <div className="flex size-12 items-center justify-center border border-border-strong bg-background">
          <FiGrid className="size-6 text-muted" />
        </div>
      </div>
      <div className="flex flex-col gap-2 text-center">
        <h1 className="text-display text-balance">
          {app.data ? `Connect to ${app.data.name}` : "Connect an app"}
        </h1>
        <p className="text-body text-muted text-pretty">
          {app.data
            ? `${app.data.name} would like to use your Otari workspace.`
            : "Review this app's access to your workspace."}
        </p>
      </div>
      <ErrorBanner error={failure} />
      {loading && <PageLoading label="Loading connection details…" />}
      {app.data && !app.error && !loading && (
        <>
          <div className="flex flex-col gap-4 border-y border-border py-5">
            <h2 className="text-overline">What you’re allowing</h2>
            <div className="flex items-start gap-3">
              <FiCpu
                aria-hidden="true"
                className="mt-0.5 size-4 shrink-0 text-muted"
              />
              <div className="flex flex-col gap-1">
                <p className="text-emphasis">
                  This app will be able to, with the access you have in this
                  workspace:
                </p>
                <ul className="flex flex-col gap-1">
                  {app.data.scopes.map((scope) => {
                    // A gateway newer than this bundle can name a scope it has
                    // no description for, which is still worth showing raw.
                    const info = SCOPE_INFO[scope]
                    return (
                      <li key={scope} className="text-caption">
                        {info?.label ?? scope}
                        {info?.risk ? (
                          <span className="text-warning">
                            {" "}
                            &middot; {info.risk}
                          </span>
                        ) : null}
                      </li>
                    )
                  })}
                </ul>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <FiShield
                aria-hidden="true"
                className="mt-0.5 size-4 shrink-0 text-muted"
              />
              <div className="flex flex-col gap-1">
                <p className="text-emphasis">Your limits stay in place</p>
                <p className="text-caption">
                  Requests count toward your existing budgets. Your provider
                  keys stay in Otari.
                </p>
              </div>
            </div>
          </div>
          {soleWorkspace && selectedWorkspace ? (
            <div className="flex flex-col gap-2">
              <h2 className="text-overline">Workspace</h2>
              <div className="flex items-center gap-3 border border-border-strong bg-background p-3">
                <FiGrid
                  className="size-4 shrink-0 text-muted"
                  aria-hidden="true"
                />
                <span className="min-w-0 flex-1 break-words text-emphasis">
                  {selectedWorkspace.name}
                </span>
                <FiCheck
                  className="size-4 shrink-0 text-success"
                  aria-hidden="true"
                />
              </div>
            </div>
          ) : choices.length > 0 || search ? (
            <ComboBoxField
              label="Workspace"
              value={selectedWorkspace?.id ?? ""}
              onChange={(id) =>
                setWorkspace(options.find((choice) => choice.id === id))
              }
              onQueryChange={setSearch}
              placeholder="Choose a workspace"
              options={options.map((workspace) => ({
                value: workspace.id,
                label: workspace.name,
              }))}
              isSourceEmpty={!search && choices.length === 0}
              noMatchesMessage="No workspace matches your search."
              emptyMessage="No workspaces available."
              menuTrigger="input"
              isDisabled={approval.isPending}
            />
          ) : !workspaces.error ? (
            <p role="status" className="text-body text-muted">
              You don’t have a workspace to connect. Ask your organization
              administrator for workspace access.
            </p>
          ) : null}
          <div className="flex flex-col gap-2 pt-1">
            <Button
              variant="primary"
              fullWidth
              isPending={approval.isPending}
              isDisabled={
                !selectedWorkspace ||
                Boolean(workspaces.error) ||
                workspaces.isFetching ||
                approval.isPending
              }
              onPress={() =>
                selectedWorkspace && approval.mutate(selectedWorkspace.id)
              }
            >
              Allow access{" "}
              <FiArrowRight className="size-4" aria-hidden="true" />
            </Button>
            <Button
              variant="ghost"
              fullWidth
              isDisabled={approval.isPending}
              onPress={deny}
            >
              Cancel
            </Button>
          </div>
          <div className="flex flex-col gap-2 border-t border-border pt-4 text-center">
            <p className="text-caption">
              You can revoke access anytime from API keys in Otari.
            </p>
            <p className="text-caption text-subtle break-words">
              Return to{" "}
              <span className="text-mono-caption">
                {new URL(app.data.redirect_uri).origin}
              </span>
            </p>
          </div>
        </>
      )}
    </LoginPageShell>
  )
}
