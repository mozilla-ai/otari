import { Button } from "@heroui/react"
import { useState } from "react"
import { useMailSettings, useSendTestMail } from "@/shared/api/hooks"
import { Field } from "@/shared/components/Field"
import { SettingsGroup } from "@/shared/components/surface"
import { ErrorBanner, InfoBanner, PageLoading } from "@/shared/components/ui"

// What each transport means to an operator reading this page. Keyed by the
// server's value rather than derived from it, so an unknown transport (a build
// serving a newer gateway) renders its own name instead of a wrong description.
const TRANSPORT_LABEL: Record<string, string> = {
  smtp: "SMTP",
  console: "Console (logged, not delivered)",
  none: "None",
}

// The settings the server can report as missing, in the wording an operator
// would search the configuration docs for.
const SETTING_LABEL: Record<string, string> = {
  smtp_host: "OTARI_SMTP_HOST",
  mail_from_email: "OTARI_MAIL_FROM_EMAIL",
  public_base_url: "OTARI_PUBLIC_BASE_URL",
  mail_transport: "OTARI_MAIL_TRANSPORT",
}

function MissingSettings({ missing }: { missing: string[] }) {
  return (
    <InfoBanner>
      Otari sends no email on this deployment, so anything that would be emailed
      offers a link to share by hand instead. To turn mail on, set{" "}
      {missing.map((key, index) => (
        <span key={key}>
          {index > 0 ? (index === missing.length - 1 ? " and " : ", ") : ""}
          <code>{SETTING_LABEL[key] ?? key}</code>
        </span>
      ))}
      , then restart the gateway.
    </InfoBanner>
  )
}

/**
 * Outgoing mail: the transport in effect, and a test send to prove it works.
 *
 * Reports the settings that would turn mail on when it is off, and disables the
 * test send in that state rather than offering one that would fail. Why mail is
 * optional at all is in docs/configuration.md#mail.
 */
export function MailDeliveryCard() {
  const mail = useMailSettings()
  const sendTest = useSendTestMail()
  const [to, setTo] = useState("")

  const data = mail.data
  const ready = data?.ready ?? false
  const result = sendTest.data
  // Nothing is claimed about mail until the server has answered. Falling back
  // to "unavailable" while the request is in flight would state the very thing
  // this card exists to report honestly, and would state it wrongly half the
  // time.
  const loading = mail.isPending && !data

  return (
    <SettingsGroup title="Email delivery">
      <div className="flex flex-col gap-4 py-4">
        <ErrorBanner error={mail.error} />
        {loading ? <PageLoading label="Loading mail settings…" /> : null}
        {data ? (
          <>
            <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[10rem_1fr]">
              <dt className="text-muted">Transport</dt>
              <dd className="text-foreground">
                {TRANSPORT_LABEL[data.transport] ?? data.transport}
              </dd>
              <dt className="text-muted">From</dt>
              <dd className="text-foreground">
                {data.from_email
                  ? `${data.from_name} <${data.from_email}>`
                  : "Not set"}
              </dd>
              <dt className="text-muted">Public base URL</dt>
              <dd className="break-all text-foreground">
                {data.public_base_url ?? "Not set"}
              </dd>
            </dl>
            {ready ? null : <MissingSettings missing={data.missing} />}
          </>
        ) : null}
      </div>

      <div className="flex flex-col gap-4 py-4">
        <div className="min-w-0">
          <p className="text-sm font-medium text-foreground">
            Send a test email
          </p>
          <p className="mt-1 max-w-3xl text-caption">
            {loading
              ? "Checking whether this deployment can send mail…"
              : ready
                ? "Sends a short message through the configured transport, so you can confirm delivery before anyone is invited."
                : "Unavailable until a transport and a public base URL are configured."}
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <Field
            label="Recipient"
            value={to}
            onChange={setTo}
            placeholder="you@example.com"
          />
          <Button
            size="sm"
            variant="outline"
            isDisabled={!ready || to.trim() === "" || sendTest.isPending}
            onPress={() => sendTest.mutate({ to: to.trim() })}
          >
            {sendTest.isPending ? "Sending…" : "Send test email"}
          </Button>
        </div>
        <ErrorBanner error={sendTest.error} />
        {result ? (
          <p
            className={`text-sm ${result.ok ? "text-success" : "text-danger"}`}
            role="status"
            aria-live="polite"
          >
            {result.ok
              ? result.transport === "console"
                ? // The console transport delivers to nobody, so telling an
                  // operator to check an inbox would send them looking for a
                  // message that was only ever written to the gateway log.
                  "Written to the gateway log. The console transport delivers to nobody."
                : `Sent over ${result.transport}. Check the recipient's inbox.`
              : `Not sent: ${result.reason ?? "the transport gave no reason."}`}
          </p>
        ) : null}
      </div>
    </SettingsGroup>
  )
}
