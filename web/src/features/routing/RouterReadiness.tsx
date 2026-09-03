import { Button } from "@heroui/react"
import { useState } from "react"
import { UserComboBox } from "@/features/users/UserComboBox"
import { useRouterStatus, useUsers } from "@/shared/api/hooks"
import { Dot, Meter } from "@/shared/components/surface"
import { ErrorBanner } from "@/shared/components/ui"

/** Records against the seed count, as a bar plus the plain numbers.
 *
 *  The number is the part an operator acts on ("six more examples"), so the bar
 *  never replaces it.
 */
function Warmth({
  records,
  seed,
  warm,
}: {
  records: number
  seed: number
  warm: boolean
}) {
  const pct =
    seed === 0 ? 100 : Math.min(100, Math.round((records / seed) * 100))
  return (
    <div className="flex items-center gap-4">
      {/* The share-bar shape, not a progress meter: 3px on the active-control
          rung with an accent fill. The quantity is in the length and the state
          is in the word beside it, which is why the fill is the accent whether
          or not the router is ready. A bar that turned amber while filling said
          "something is wrong" about a policy that is simply new. */}
      <Meter
        fraction={pct / 100}
        ariaLabel={`${records} of ${seed} examples`}
      />
      <span className="text-sm text-foreground">
        {records} / {seed} examples
      </span>
      <span
        className={`flex items-center gap-2 font-mono text-[13px] ${
          warm ? "text-foreground" : "text-subtle"
        }`}
      >
        <Dot className={warm ? "bg-accent" : "bg-text-subtle"} />
        {warm ? "ROUTING" : "WARMING UP"}
      </span>
    </div>
  )
}

/** How many scored examples a learned policy has, and so whether it can act yet.
 *
 *  Read-only on purpose. A policy that defers to a router serves its default
 *  target on every request until examples exist, which is safe but
 *  indistinguishable from a broken router: the operator sees the strong model in
 *  Activity every time and concludes the feature does not work. This is the
 *  surface that answers it, and it is the whole answer the dashboard owes for now,
 *  because recording examples is an API job (see `docs/routing.md`).
 *
 *  Warmth is per *user*, not per policy: the examples are one user's own prompts,
 *  so a policy every caller resolves warms once per caller. That is why there is a
 *  picker here rather than a number in the table, and why the copy says so.
 */
export function RouterReadiness({
  policyName,
  candidates,
  defaultTarget,
  backend,
  scopedUserId,
  onClose,
}: {
  policyName: string
  candidates: string[]
  defaultTarget: string
  backend: string
  /** The user a user-scoped policy belongs to: then it is the only memory in play. */
  scopedUserId: string | null
  onClose: () => void
}) {
  const users = useUsers()
  const [userId, setUserId] = useState<string | null>(scopedUserId)
  const status = useRouterStatus(userId)
  const chosen = userId !== null && userId !== ""

  return (
    <div>
      <div className="flex items-center justify-between border-b border-border px-4 py-2">
        <span className="text-body">
          Examples for <code>{policyName}</code>
        </span>
        <Button size="sm" variant="ghost" onPress={onClose}>
          Close
        </Button>
      </div>

      <div className="flex flex-col gap-5 px-4 py-4">
        <div className="flex flex-col gap-1">
          <span className="text-caption">Candidates</span>
          <span className="text-body">
            <code>{backend}</code> ranks {candidates.join(", ")} for each
            request.
          </span>
          <span className="text-caption">
            <code>{defaultTarget}</code> serves whenever it declines: too few
            examples, a weakly supported pick, a request carrying tools, or{" "}
            <code>Otari-Router: off</code>.
          </span>
        </div>

        <div className="flex flex-col gap-2">
          <span className="text-caption">Examples</span>
          {scopedUserId === null ? (
            <UserComboBox
              label="Whose memory"
              value={userId ?? ""}
              onChange={setUserId}
              users={users.data ?? []}
              placeholder="Pick a user…"
              description="Examples are one user's own prompts, so this policy warms once per caller rather than once overall."
              unknownHint={
                <span className="text-danger">
                  No such user. Pick an existing one.
                </span>
              }
            />
          ) : (
            <span className="text-caption">
              Scoped to user <code>{scopedUserId}</code>, so that is the only
              memory it can use.
            </span>
          )}

          <ErrorBanner error={status.error} />

          {!chosen ? (
            <span className="text-sm text-muted">
              Pick a user to see how warm this policy&apos;s memory is.
            </span>
          ) : status.isLoading ? (
            <span className="text-sm text-muted">Loading…</span>
          ) : status.data ? (
            <div className="flex flex-col gap-2">
              <div className="flex flex-wrap items-center gap-3">
                <span className="text-body">Default pool</span>
                <Warmth
                  records={status.data.default_pool.records}
                  seed={status.data.seed_count}
                  warm={status.data.default_pool.warm}
                />
              </div>
              {status.data.tasks.map((pool) => (
                <div
                  key={pool.task_id}
                  className="flex flex-wrap items-center gap-3"
                >
                  <code className="font-mono text-body">{pool.task_id}</code>
                  <Warmth
                    records={pool.records}
                    seed={status.data.seed_count}
                    warm={pool.warm}
                  />
                </div>
              ))}
              {status.data.tasks.length > 0 ? (
                <span className="text-caption">
                  The default pool counts every example this user has, including
                  the ones filed under a task, so it can be warm while a task
                  partition is not.
                </span>
              ) : null}
              <span className="text-caption">
                Scoring with <code>{status.data.embedding_model}</code>,{" "}
                {status.data.k} nearest examples per decision, cost dial{" "}
                {status.data.alpha}, deciding once per{" "}
                {status.data.granularity === "trace_sticky"
                  ? "conversation"
                  : "call"}
                . Change these with the <code>OTARI_ROUTER_*</code> environment
                variables.
              </span>
            </div>
          ) : null}
        </div>

        <div className="flex flex-col gap-1">
          <span className="text-caption">Adding examples</span>
          <span className="text-caption">
            Examples are recorded over the API, with{" "}
            <code>POST /v1/routing/preferences/rank</code>. Score a batch of
            prompts from 0 (bad) to 1 (great) per candidate; two good answers is
            the case that lets the cheaper model win. See{" "}
            <a
              className="text-link hover:underline"
              href="https://mozilla-ai.github.io/otari/routing/#teach-it"
              target="_blank"
              rel="noreferrer"
            >
              Teach it
            </a>{" "}
            in the routing guide.
          </span>
        </div>
      </div>
    </div>
  )
}
