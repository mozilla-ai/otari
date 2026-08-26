import { Card, Link } from "@heroui/react"
import { useGatewayHealth } from "@/shared/api/hooks"
import { CopyableValue } from "@/shared/components/ui"
import { useDeployment } from "@/shared/hooks/useDeployment"

/**
 * The root a hybrid gateway serves: what this process is, whether it is working,
 * and where its control plane actually lives.
 *
 * A gateway attached to otari.ai is a data-plane runtime. Its organizations,
 * credentials, routing, budgets and usage are owned there, so this page links to
 * them rather than reproducing them, and it holds no management state of its
 * own: no hosted session, no credential, no budget data, and no proxy of the
 * hosted dashboard. The only thing it reads is `/health`, which is public and
 * carries no secret, and the only thing it renders from the bootstrap is a link
 * target the operator configured.
 *
 * What is left is what an operator standing at this URL actually needs: is the
 * gateway up, can it reach otari.ai, what do I point a client at, and where do I
 * go to change anything.
 */
export function HybridLanding() {
  const { management_url } = useDeployment()
  const health = useGatewayHealth()

  // The gateway serving this page and the control plane behind it fail
  // separately, and an operator's next move differs by which one did: a dead
  // gateway is this host's problem, an unreachable platform is a network or
  // status-page question. So they are two rows, not one verdict.
  const gateway: Status = health.isError
    ? { tone: "alert", state: "Not responding" }
    : health.isPending
      ? { tone: "pending", state: "Checking…" }
      : health.data?.status === "healthy"
        ? { tone: "ok", state: "Healthy" }
        : { tone: "warn", state: health.data?.status ?? "Unknown" }

  // Only the gateway can answer this, so its own failure leaves the answer
  // unknown rather than bad: saying "unreachable" here would blame otari.ai for
  // a local process that is not talking to anyone.
  const platform: Status = health.isError
    ? { tone: "warn", state: "Unknown" }
    : health.isPending
      ? { tone: "pending", state: "Checking…" }
      : health.data?.platform_reachable === "yes"
        ? { tone: "ok", state: "Connected" }
        : health.data?.platform_reachable === "no"
          ? { tone: "alert", state: "Unreachable" }
          : { tone: "warn", state: "Unknown" }

  // What a client is configured with, taken from the URL this page was served
  // from rather than from anything the gateway reports: a deployment behind a
  // proxy or a custom domain is reached at the address the operator is looking
  // at, which is this one.
  const baseUrl = `${window.location.origin}/v1`

  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <Card className="w-full max-w-lg">
        <Card.Content className="flex flex-col gap-6 p-7">
          <div className="flex flex-col items-center gap-3 text-center">
            {/* Decorative: the heading beside it already names the product. */}
            <img src="/favicon.svg" alt="" className="h-12 w-12" />
            <div>
              <h1 className="text-lg font-semibold text-foreground">
                Otari gateway
              </h1>
              <p className="mt-1 text-sm text-muted">
                This gateway serves requests. Its providers, routing, budgets
                and usage are managed on otari.ai.
              </p>
            </div>
          </div>

          <div>
            <StatusRow label="Gateway" status={gateway} />
            <StatusRow label="otari.ai connection" status={platform} />
          </div>

          <div className="flex flex-col gap-2">
            <h2 className="text-title">Point a client here</h2>
            <CopyableValue
              value={baseUrl}
              label="gateway base URL"
              className="font-mono text-sm text-foreground"
            />
            <p className="text-sm text-muted">
              Use it as the base URL of any OpenAI-compatible client, with an
              otari.ai API key as the bearer token. Keys are issued on otari.ai;
              this gateway stores none.
            </p>
          </div>

          {management_url ? (
            <Link
              href={management_url}
              target="_blank"
              rel="noreferrer"
              className="text-sm font-medium text-link hover:text-link-hover"
            >
              Manage this gateway on otari.ai
            </Link>
          ) : null}
        </Card.Content>
      </Card>
    </div>
  )
}

// "pending" is deliberately not one of the foundation's three status roles: a
// check still in flight is not a state of the thing being checked, so it reads
// as ordinary text rather than claiming a verdict for the half-second before one
// arrives.
type Tone = "ok" | "warn" | "alert" | "pending"

interface Status {
  tone: Tone
  state: string
}

const TONES: Record<Tone, { pill: string; dot: string }> = {
  ok: {
    pill: "border-success bg-success-subtle text-success",
    dot: "bg-success",
  },
  warn: {
    pill: "border-warning bg-warning-subtle text-warning",
    dot: "bg-warning",
  },
  alert: {
    pill: "border-danger bg-danger-subtle text-danger",
    dot: "bg-danger",
  },
  pending: { pill: "border-border text-muted", dot: "bg-muted" },
}

/**
 * One live condition, as a labeled pill. Color is never the only signal: the
 * pill carries the state as a word, as the provider health pills do.
 */
function StatusRow({ label, status }: { label: string; status: Status }) {
  const { pill, dot } = TONES[status.tone]
  return (
    <div className="flex items-center justify-between gap-3 border-t border-border py-2.5 first:border-t-0">
      <span className="text-sm text-muted">{label}</span>
      <span
        className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${pill}`}
      >
        <span aria-hidden className={`h-1.5 w-1.5 rounded-full ${dot}`} />
        {status.state}
      </span>
    </div>
  )
}
