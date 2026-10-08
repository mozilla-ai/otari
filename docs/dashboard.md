# Admin dashboard

Otari serves its dashboard at the gateway root. The dashboard uses the same
management API available to scripts, so access and behavior are determined by the
server rather than duplicated in the frontend.

## Deployment modes

- **Standalone** shows the local management UI and serves inference from the same
  process.
- **Hosted** shows the multi-tenant control plane. Inference snippets point to
  `data_plane_url`; the hosted process does not serve inference.
- **Hybrid** shows gateway health and a link to the otari.ai control plane. It has
  no local management UI.

The dashboard reads `GET /api/v1/bootstrap` before rendering. That response controls
the available sign-in methods and navigation surfaces, so a page that is not
available in the current mode is not offered.

## Sign-in and secrets

The master key and `OTARI_SECRET_KEY` are different credentials:

| Credential | Purpose |
| --- | --- |
| Master key | Authorizes deployment management and signs in the first operator |
| `OTARI_SECRET_KEY` | Encrypts provider and search-tool credentials stored in the database |

If `OTARI_MASTER_KEY` is unset on a new standalone database, Otari generates a
master key and prints it once. The dashboard exchanges it for an HttpOnly session
cookie; it does not store the master key in browser storage.

Set `OTARI_SECRET_KEY` before saving provider or search-tool credentials through
the dashboard:

```bash
otari gen-secret-key
```

Keep this key separate from the database. Losing it makes stored credentials
undecryptable. Providers configured in `config.yml` do not require it.

The operator can claim a new deployment by setting an email address and password
from Account settings. After that, the sign-in page uses the password rather than
the master key. The master key remains valid for management API calls and can
reset the operator password. It resets the operator's password only: to help a
member who forgot theirs, send them a password link from Accounts, which emails
it where mail is configured and otherwise gives you the link to share (see
[Access control](access-control.md#deployment-wide-account-administration)).

The sign-in page offers the email and password form as soon as any identity holds
a password, which can be before the operator claims the deployment: a member
added to the roster and signed up signs in there. While both credentials still
work, the page offers the master-key box beside the form.

Account settings also carries the name you are known by, which is not a
credential: it is what the account menu, the organization roster and the user
column on Usage and Activity show. Someone an admin added to the roster by
address has no name until they set one here. Clearing it puts those surfaces
back to naming you by your sign-in address. The dashboard reads who is signed in
from `GET /api/v1/auth/profile`, which answers whether or not you belong to an
organization, so the account menu names you even before you join one.

## First-run walkthrough

1. Start Otari in standalone mode.
2. Open the gateway root, usually `http://localhost:8000/`.
3. Sign in with the configured or generated master key.
4. Add a provider. If storing its credential in the dashboard, configure
   `OTARI_SECRET_KEY` first.
5. Test the provider connection or declare model IDs for a backend that has no
   model-listing endpoint.
6. Create an API key from the setup guide or the Keys page.
7. Send a request and confirm it appears under Activity.

The [Quickstart](quickstart.md) includes a complete request example.

## The setup guide

Overview offers a short setup flow until the selected workspace serves its first
successful gateway request from outside the product. A message sent in the
Playground does not close it: the guide marks the moment somebody's own code
first reached the gateway, which is the point of the flow. The key it creates is
an ordinary workspace API key.
Skipping the guide hides it for that workspace; it does not revoke a key already
created. Set `activation_guide: false` to disable the flow for the deployment.

## Navigation

The workspace view contains day-to-day gateway operations:

- Overview, Activity, and Usage
- Playground, Models, and Routing. Models is the catalog grouped by model: a list of cards with a rail of filters beside it, and a page per model where every offering of it is compared, each with its own limits and the price your organization is charged. Each row shows the provider's own model ID under the provider, which tells apart two offerings on one provider, such as Bedrock's regional inference profiles. It is read-only; a rate is set on Providers. Routing lists each policy with its main model, its backups in order, and who it applies to. Create policy first asks what the policy is for (a fallback, a traffic split, a smart router, spill-over at rate limits, or a budget tier-down) and then shows only that kind's fields; the empty list offers a fallback directly. Models are chosen by model and then by provider. Each policy has its own page, opened from Edit in the list, where its main selection, fallback chain, budget tier-down and guardrails are combined, each section with its own Save. Policies from `config.yml` open there read-only. A hybrid or hosted deployment says that its policies are not where a request's attempt plan comes from.
- Tools
- API keys and workspace members

The organization view contains tenant-wide administration:

- Organization-wide usage, in hosted mode
- Workspaces and organization members
- Email domains, for joining colleagues automatically
- Spend and budgets
- General: Providers, followed by Deployment providers and Org settings.
  Providers is the organization's own upstream credentials, the models each one
  reaches, and what this organization pays for them: adding a key offers every
  model the provider lists on it, each carrying a switch that decides whether
  the runtime serves it. Each is priced by the first rung that answers, which is
  the order a request is metered by: this organization's own rate, then the
  deployment price list, then the community defaults. A model nothing prices is
  offered and left unserved. Each key's row counts how many of its offered
  models are serving, and a key serving none of them is marked, because every
  request through it is refused until a switch is on. Deployment providers is
  the process-wide credential list, which a deployment operator manages and
  which is served to every organization; it appears in standalone mode only.

Settings shows the effective non-secret configuration. Some values can be changed
at runtime and others require a restart. The server marks that distinction in the
settings response.
Its Rate limit rules card adds, edits and removes limits on requests per minute,
tokens per minute and requests in flight; rules from config.yml are listed there
read-only.

What a page shows can also depend on who is signed in, not only on the
deployment. Spend and budgets is the clearest case: an organization owner or
admin manages their own organization's budgets and the spend ceilings holding
them, while a deployment operator gets the deployment's own budgets, their
active organization's, and the gateway users assigned to them. Usage, activity,
traces, routing and the overview's key count likewise show an operator their
active organization, not every organization on the deployment.

Providers answers to the organization role rather than to deployment authority.
Its owners and admins manage the keys, the models each key offers, and the rate
each model is billed at. An organization may set its own rate for a model it
supplies the provider key for. A model reached through one of the deployment's
own provider instances is not one of those: the deployment holds that credential
and settles its upstream bill, so its rate is the deployment price list's, which
is `/api/v1/pricing` and has no page of its own. The models.dev defaults are
likewise kept current through `/api/v1/pricing/refresh` rather than from the
dashboard.

A rate is per million tokens, per request, or per image, and the rate form asks
which. A request or image rate is entered per 1,000, the way providers publish
one: $2 per 1,000 searches for a rerank model is entered as 2 and stored as 2000
per million, which is how every rate travels on the wire. The model lists name
such a rate by its unit, for example "$0.002 per request". The same choice is on
Activity's "Price this model", offered on a served request that carried no cost
(a failed attempt billed nothing, so its error is the guide instead), which
opens per request for a rerank, audio or moderation request and per image for
an image generation. The unit decides what
a rate bills. A token rate on an audio or moderation request is still read per
request for now and logs a warning; a later release reads it strictly, as
tokens, which those requests do not report, so price them per request.

Exact page names and availability can change with deployment mode and installed
extensions. The running dashboard is the source of truth.

What a page shows can also depend on who is signed in. On API keys, an operator
manages every key in the organization and chooses each key's owner, while a
member sees the same page scoped to their own keys, always billed to themselves
and never budget-exempt.

Overview splits the same way. Everyone lands on their own spend, traffic and
recent requests. An organization owner or admin also gets a budget-health
figure, read from the spend ceilings holding their organization, while a
deployment operator gets provider health and the deployment's own budgets
instead. A member who belongs to no workspace yet is told so beside the
activity table, rather than shown workspace counts.

## Playground

The Playground is the in-product chat page: pick a model the gateway serves and
talk to it, or put two side by side and record which answered better. Requests
run through the same path as any other completion, so routing policies,
guardrails, budgets and the tool loop all apply, and every request appears in
Activity and Usage.

A question can carry files. Attach them with the paperclip in the composer or
drop them anywhere on the page; each uploads at once and shows as a chip with
its name and size until the question is sent. The model reads them the way an
API request's [uploaded files](files.md) are read, so a text-only local model is
given their extracted text. When comparing, both models get the same files. A
saved conversation remembers which files each question carried. Removing a
chip takes the file off that question and keeps the upload: the Files dialog
lists everything you uploaded in the workspace, to attach again or to delete.
These controls appear only where the deployment stores files: not when file
uploads are off, and not on a hosted control plane.

The Tools menu attaches web search, code execution and any number of MCP
servers to the next message, in any mix: the model is offered all of them in one
tool loop. A tool the gateway runs shows in the transcript above the reply as a
collapsed row with its status, opening to the arguments the model sent and what
the tool returned (a long result is shortened for display). This activity is
shown in the Playground only (not on a hosted control plane, whose requests run
on the data plane) and is not saved with a conversation; a request
made through the API still returns just the final message, as
[What clients receive](tools.md#what-clients-receive) describes.

Two things are worth knowing about how it is billed and what it stores.

A Playground request carries no API key. It is authorized by the dashboard
session and billed to whoever sent it, in the workspace the switcher has
selected, against that person's own budget and rate limit; its usage row
therefore has no key attached. Picking a different model changes which provider
credential the gateway resolves, never who pays. A request made here also does
not count as the workspace's first request, so the setup guide keeps waiting for
one from outside the product, which is the milestone it exists to mark.

Nothing is stored until asked for. Saving a conversation or recording a
comparison prompts once for content retention, and the flags are per person and
revocable; withdrawing one blocks new saves and deletes nothing. Saved
transcripts, recorded comparisons and pinned models are visible only to the
person who created them, even to colleagues in the same workspace, and each can
be deleted from the page. Hosted control planes serve no inference, so they do
not offer the page at all.

## Observability

Activity is the per-request log. Sessions lists [agent sessions](traces.md)
where the deployment records traces (`trace_capture_enabled`, on by default; a
hybrid gateway records none), and has no entry where it does not. A session
opens in a side panel as a tree of its turns, LLM calls and tool calls, a
timeline of the same spans, or a flat log; before a span is picked, the panel
shows what the session cost against its key's own spend ceiling and where the
gateway stepped in. Usage provides aggregates and time series. Routed requests share a request-group ID, so the
request log can show the attempted models and the model that served the
response. Imported usage is labeled by
source and does not consume a budget.

Use Prometheus at `/metrics` for process-level monitoring when
`enable_metrics` is enabled. The scrape needs the `metrics` extra
(`pip install gateway[metrics]`); the Docker image installs it, and a
source install that sets `enable_metrics` without it refuses to start rather
than serving an empty scrape.

## Organization

Organizations own workspaces. The workspace switcher determines which keys,
members, usage, tools, aliases, and stored routing policies are in view. The
organization section manages resources shared across those workspaces.

What a signed-in person can see or change is enforced by their organization and
workspace roles. Deployment-wide operations require an operator. See
[Access control](access-control.md).

Invitations waiting on you are listed under Invitations, reached from the
organization switcher. After you accept one, the page offers to take you into
that organization. Someone signed in who is not yet a member of any
organization is taken to Invitations when one is waiting; otherwise Overview
says how to get in, and the sidebar holds only Overview. Creating an
organization is offered to them only where `open_signup` is on, since the
gateway refuses it otherwise.

Opening an invitation link in a browser signed in as a different address, or as
an identity with no address, says who the invitation is for and offers to sign
out and continue, because accepting always adds the invited address, whoever is
signed in. After a
password is set from an invitation link, the sign-in screen opens with the
invited address filled in. When the link cannot set a password and the address
has none yet, the page says how to get in instead: a Google or GitHub sign-in,
or "Forgot your password?", whichever the deployment offers. The other pages in
front of sign-in carry the address the same way: from the sign-in screen to
"Forgot your password?", from signup or a resend to the page saying which
address was mailed and on to another resend, and back to the sign-in screen
after a verification or a recovery request. A password never travels with it.

On Members, an invited row offers Reissue and Revoke, and says when its
invitation has expired. Reissue sends a new accept link (or shows it to share,
when the email did not go out) and the old one stops working, which is how a
lost link is recovered.

## Authentication options

Password sign-in works for an existing account. Optional passkeys, Google OAuth and GitHub OAuth add more ways to sign in. Set `passkeys_enabled: false` (`OTARI_PASSKEYS_ENABLED`) to switch passkeys off entirely. When the deployment sets `open_signup: true`, a new address can also create an account with its own organization, through the signup form or a Google or GitHub sign-in. Where the deployment publishes terms (`terms_url`), the signup page asks for them before any of those, and the invitation page asks before "Accept and sign in with a provider instead". A provider sign-in from the sign-in screen that would set up an account without them stops on the page the provider returns to, which asks for the terms and starts the sign-in again. A passkey never creates an account. OAuth requires `public_base_url` plus the provider's client ID and secret. Passkeys can instead use `public_base_url`, or an explicit `webauthn_rp_id` and `webauthn_allowed_origins` pair.

A link to a dashboard page opened while signed out shows the sign-in screen first, then that page once you have signed in, whichever of these methods you use.

Signing in *can* add a membership in one case. If an organization has claimed and
proven the email domain that the identity's verified address belongs to, the
sign-in joins them to that organization at the role the claim names, whichever of
the three credentials they used. It is the only path that grants membership
without somebody deciding about the person, and it is fenced accordingly: see
[Email-domain auto-join](access-control.md#email-domain-auto-join).

Mail is optional. When an invitation email does not go out (no mail, a failed send, or the
daily invitation mail limit), you get the accept link to share,
and an invitee new to the deployment chooses a password on the page it opens,
so a deployment without SMTP can still bring people in. See [Configuration](configuration.md#mail).

## Bundled guide and custom documentation

This guide is bundled into the dashboard at `/#/docs`. Set `docs_url` to make
the dashboard's Documentation link open a different site. The bundled route
remains available.

## Legal pages

`terms_url` and `privacy_url` name where this deployment's terms of service and
privacy notice live. The account menu links each row it has an address for; see
[Configuration](configuration.md#legal-pages).

## Development

The dashboard source is under `web/`. The published Docker image includes the
bundle. A source checkout needs `make dashboard` for the gateway to serve it, or
`pnpm --dir web dev` for frontend development. See [web/README.md](../web/README.md).
