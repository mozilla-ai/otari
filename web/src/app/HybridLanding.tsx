import { Card, Link } from "@heroui/react"
import { useDeployment } from "@/shared/hooks/useDeployment"

/**
 * The root a hybrid gateway serves: what this process is, and where its control
 * plane actually lives.
 *
 * A gateway attached to otari.ai is a data-plane runtime. Its organizations,
 * credentials, routing, budgets and usage are owned there, so this page links to
 * them rather than reproducing them, and it holds no management state of its
 * own: no hosted session, no credential, no budget data, and no proxy of the
 * hosted dashboard.
 *
 * Deliberately the minimum that renders from the bootstrap. Health, connection
 * state and setup guidance are #587's, and this is the seam they land in.
 */
export function HybridLanding() {
  const { management_url } = useDeployment()

  return (
    <div className="flex min-h-full items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <Card.Content className="flex flex-col items-center gap-4 p-7 text-center">
          <img src="/favicon.svg" alt="Otari" className="h-12 w-12" />
          <div>
            <h1 className="text-lg font-semibold text-[var(--otari-ink)]">
              Connected to otari.ai
            </h1>
            <p className="mt-1 text-sm text-[var(--otari-muted)]">
              This gateway serves requests. Its providers, routing, budgets and
              usage are managed on otari.ai.
            </p>
          </div>
          {management_url ? (
            <Link
              href={management_url}
              target="_blank"
              rel="noreferrer"
              className="text-sm font-medium text-[var(--otari-brand-dark)]"
            >
              Open otari.ai
            </Link>
          ) : null}
        </Card.Content>
      </Card>
    </div>
  )
}
