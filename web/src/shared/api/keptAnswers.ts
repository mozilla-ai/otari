/**
 * Answers kept in this browser from one load for the next, where the request
 * policy turns them on (`RequestPolicy.keptAnswers`).
 *
 * Each entry names the API origin it was read from, so a region chosen later
 * never reads another's, and the time it was kept. An entry older than a
 * dashboard session lives is treated as absent: the session that read it has
 * expired by then, so whoever opens the page next is not shown what it read.
 * Storage that refuses a read or a write keeps nothing, and the load waits for
 * the current answer as it would with the switch off.
 */

/**
 * The gateway's default `dashboard_session_ttl_hours`. The dashboard cannot
 * read the configured value before it has a session, so it takes the default;
 * `tests/unit/test_kept_answers_lifetime.py` fails when the two disagree.
 */
export const KEPT_ANSWER_HOURS = 168

const KEPT_ANSWER_MS = KEPT_ANSWER_HOURS * 60 * 60 * 1000

interface Entry {
  origin?: unknown
  keptAt?: unknown
  value?: unknown
}

export interface KeptAnswer<T> {
  read(): T | undefined
  /** Keeps `value`, or forgets the entry when it is `undefined`. */
  write(value: T | undefined): void
}

// Parsed once per stored value, so every reader of an entry holds the same
// object, as every reader of a cached query does.
const parsed = new Map<string, { raw: string; entry: Entry | null }>()

function readEntry(key: string): Entry | null {
  const raw = window.localStorage.getItem(key)
  if (raw === null) return null
  const seen = parsed.get(key)
  if (seen?.raw === raw) return seen.entry
  let entry: Entry | null = null
  try {
    const value: unknown = JSON.parse(raw)
    entry = typeof value === "object" && value !== null ? value : null
  } catch {
    entry = null
  }
  parsed.set(key, { raw, entry })
  return entry
}

/**
 * The answer kept under `key` for the API at `origin`.
 *
 * Trusted unchecked beyond its envelope, because only this dashboard writes
 * the key: an entry from an older bundle stands in for one round trip at most,
 * until the current answer replaces it.
 */
export function keptAnswer<T>(key: string, origin: string): KeptAnswer<T> {
  return {
    read() {
      try {
        const entry = readEntry(key)
        if (!entry || entry.origin !== origin) return undefined
        const age =
          typeof entry.keptAt === "number"
            ? Date.now() - entry.keptAt
            : Number.NaN
        if (!(age >= 0 && age < KEPT_ANSWER_MS)) return undefined
        return (entry.value ?? undefined) as T | undefined
      } catch {
        return undefined
      }
    },
    write(value) {
      if (value === undefined) {
        forgetKeptAnswer(key)
        return
      }
      try {
        window.localStorage.setItem(
          key,
          JSON.stringify({ origin, keptAt: Date.now(), value }),
        )
      } catch {
        // Private mode or a full store: the next load waits for the answer.
      }
    },
  }
}

export function forgetKeptAnswer(key: string): void {
  try {
    window.localStorage.removeItem(key)
  } catch {
    // Storage refused: nothing was kept either.
  }
}

/**
 * Records what a kept answer did on this load as a performance mark, for the
 * boot timing to read back (`boot:kept-bootstrap:hit`, `ctx:kept:off`, ...).
 */
export function markKeptAnswer(name: string): void {
  try {
    performance.mark(name)
  } catch {
    // An environment without the User Timing API records nothing.
  }
}
