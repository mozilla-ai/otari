import { useState } from "react"
import { FiArrowRight, FiCheck, FiCpu, FiGrid, FiShield } from "react-icons/fi"
import type { Workspace } from "@/client"
import { Button } from "@/design-system/actions/Button"
import { ErrorBanner } from "@/design-system/feedback/ErrorBanner"
import { InfoBanner } from "@/design-system/feedback/InfoBanner"
import { PageLoading } from "@/design-system/feedback/PageLoading"
import { ComboBoxField } from "@/design-system/forms/ComboBoxField"
import { Field } from "@/design-system/forms/Field"
import { Select } from "@/design-system/forms/Select"
import { ProductMark } from "@/design-system/ProductMark"
import { SCOPE_INFO } from "@/features/keys/scopes"
import {
  type AppApproval,
  useAuthorizeOAuthApp,
  useOAuthApp,
  useOAuthAppWorkspaces,
} from "@/shared/api/oauthApps"
import {
  useDeploymentOperator,
  useOrganizationContext,
} from "@/shared/api/organizations"
import { LoginPageShell } from "./LoginPageShell"

const EXPIRATIONS: { value: AppApproval["expiration"]; label: string }[] = [
  { value: "never", label: "Never" },
  { value: "1-hour", label: "In 1 hour" },
  { value: "1-day", label: "In 1 day" },
  { value: "7-days", label: "In 7 days" },
  { value: "30-days", label: "In 30 days" },
  { value: "90-days", label: "In 90 days" },
  { value: "180-days", label: "In 180 days" },
  { value: "1-year", label: "In 1 year" },
]

// An MCP client registered itself, so nobody vouches for it: its sign-in always
// ends, within 90 days, and an expiry its link suggests is not taken.
const SIGN_IN_EXPIRATIONS = EXPIRATIONS.filter((choice) =>
  ["1-day", "7-days", "30-days", "90-days"].includes(choice.value),
)

/** A positive amount under a million dollars, in whole cents, as the server accepts. */
function isLimit(value: string) {
  return /^\d{1,6}(\.\d{1,2})?$/.test(value) && Number(value) > 0
}

function expirationFrom(
  value: string | null,
): AppApproval["expiration"] | undefined {
  return EXPIRATIONS.find((choice) => choice.value === value)?.value
}

/** Consent stays behind the dashboard's normal session gate. */
export function AppAuthorizationPage() {
  const params = new URLSearchParams(window.location.search)
  const [workspace, setWorkspace] = useState<Workspace>()
  const [search, setSearch] = useState("")
  // The app may suggest both; one the screen cannot use starts blank or at never.
  const [limit, setLimit] = useState(() => {
    const suggested = params.get("limit") ?? ""
    return isLimit(suggested) ? suggested : ""
  })
  const [chosenExpiration, setExpiration] = useState(() =>
    expirationFrom(params.get("expiration")),
  )
  const [signInExpiration, setSignInExpiration] =
    useState<AppApproval["expiration"]>("30-days")
  const limitInvalid = limit !== "" && !isLimit(limit)
  const app = useOAuthApp(
    params.get("client_id") ?? "",
    params.get("redirect_uri") ?? "",
  )
  // A self-registered client, such as an MCP client, is given the person's own
  // read-only access token: no workspace to bill and no limit to set.
  const isAccessToken = app.data?.credential === "access_token"
  const expiration = isAccessToken
    ? signInExpiration
    : (chosenExpiration ?? "never")
  const context = useOrganizationContext(isAccessToken)
  const organization = context.data?.organization
  const { isOperator } = useDeploymentOperator(isAccessToken)
  const workspaces = useOAuthAppWorkspaces(
    search,
    Boolean(app.data) && !app.error && !isAccessToken,
  )
  const approval = useAuthorizeOAuthApp(params, app.data?.scopes ?? [])
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
  const failure =
    app.error ??
    workspaces.error ??
    (isAccessToken ? context.error : null) ??
    approval.error
  const loading =
    (app.isPending && !app.data) ||
    (Boolean(app.data) &&
      !isAccessToken &&
      workspaces.isPending &&
      !workspaces.data) ||
    (isAccessToken && context.isPending)
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
          {!app.data.verified && (
            <InfoBanner tone="warning">
              Otari has not verified this site. Allow access only if you trust{" "}
              <span className="text-mono-caption">{app.data.name}</span>.
            </InfoBanner>
          )}
          <div className="flex flex-col gap-4 border-y border-border py-5">
            <h2 className="text-overline">What you’re allowing</h2>
            {isAccessToken ? (
              <div className="flex items-start gap-3">
                <FiShield
                  aria-hidden="true"
                  className="mt-0.5 size-4 shrink-0 text-muted"
                />
                <p className="text-caption">
                  This app will read what you can see in Otari, as you: usage,
                  requests, budgets and keys. It cannot change anything or call
                  models.
                  {isOperator
                    ? " Because you operate this deployment, that includes its settings and every key, member and trace."
                    : null}
                </p>
              </div>
            ) : (
              <>
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
                            {info
                              ? info.label
                              : `${scope} (not described here)`}
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
              </>
            )}
          </div>
          {isAccessToken ? (
            organization ? (
              <div className="flex flex-col gap-2">
                <h2 className="text-overline">Organization</h2>
                <div className="flex items-center gap-3 border border-border-strong bg-background p-3">
                  <FiGrid
                    className="size-4 shrink-0 text-muted"
                    aria-hidden="true"
                  />
                  <span className="min-w-0 flex-1 break-words text-emphasis">
                    {organization.name}
                  </span>
                  <FiCheck
                    className="size-4 shrink-0 text-success"
                    aria-hidden="true"
                  />
                </div>
              </div>
            ) : (
              <p role="status" className="text-body text-muted">
                You aren’t in an organization to connect.
              </p>
            )
          ) : soleWorkspace && selectedWorkspace ? (
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
          <div className="grid gap-4 sm:grid-cols-2">
            {app.data.spend_limits && !isAccessToken && (
              <Field
                label="Spending limit (USD)"
                value={limit}
                onChange={setLimit}
                placeholder="No limit"
                inputMode="decimal"
                isDisabled={approval.isPending}
                isInvalid={limitInvalid}
                errorMessage="Enter an amount above 0 and under 1,000,000, in dollars and cents."
              />
            )}
            {isAccessToken ? (
              <Select
                label="Access expires"
                value={signInExpiration}
                onChange={(value) =>
                  setSignInExpiration(
                    SIGN_IN_EXPIRATIONS.find((choice) => choice.value === value)
                      ?.value ?? "30-days",
                  )
                }
                options={SIGN_IN_EXPIRATIONS}
                isDisabled={approval.isPending}
              />
            ) : (
              <Select
                label="Key expires"
                value={expiration}
                onChange={(value) => setExpiration(expirationFrom(value))}
                options={EXPIRATIONS}
                isDisabled={approval.isPending}
              />
            )}
          </div>
          <div className="flex flex-col gap-2 pt-1">
            <Button
              variant="primary"
              fullWidth
              isPending={approval.isPending}
              isDisabled={
                (!isAccessToken && !selectedWorkspace) ||
                (isAccessToken && !organization) ||
                limitInvalid ||
                Boolean(workspaces.error) ||
                workspaces.isFetching ||
                approval.isPending
              }
              onPress={() => {
                if (isAccessToken && organization) {
                  approval.mutate({
                    organizationId: organization.id,
                    limit: "",
                    expiration,
                  })
                } else if (selectedWorkspace) {
                  approval.mutate({
                    workspaceId: selectedWorkspace.id,
                    limit: app.data.spend_limits ? limit : "",
                    expiration,
                  })
                }
              }}
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
              {isAccessToken
                ? "You can revoke access anytime from Account settings in Otari."
                : "You can revoke access anytime from API keys in Otari."}
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
