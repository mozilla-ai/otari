import { expect, type Page } from "@playwright/test";

import { authHeaders, expectOk } from "./helpers";

// The fixture the behavioural-parity specs read.
//
// Every value is namespaced (`parity-…`) so an assertion can name exactly the
// rows this fixture owns. The suite shares one gateway database with
// dashboard.spec.ts, which seeds its own usage under a different source, and the
// pages under test are gateway-wide: an assertion on a global count would be a
// bet on what ran before it. Filtering to this namespace is what makes the
// behaviour, not the surrounding state, the thing under test.
//
// Rows are imported through POST /v1/usage/external-events rather than by making
// provider calls: the endpoint writes real usage_logs rows (priced through the
// same pricing path) with no upstream to reach, which is what lets the suite run
// with no network egress. Imported rows carry counts_toward_budget=false, which
// is also what makes them eligible for Activity's bulk actions.
export const PARITY = {
  source: "parity-e2e",
  users: {
    // Two users so the User breakdown and the user filter both have something to
    // separate.
    heavy: "parity-heavy@example.com",
    light: "parity-light@example.com",
  },
  sessions: {
    heavy: "parity-session-heavy",
    light: "parity-session-light",
  },
  models: {
    // Priced, so the row carries a cost and the Priced filter has a positive case.
    priced: { provider: "openai", model: "gpt-parity-priced" },
    // Deliberately left without a pricing row: the negative case for the Priced
    // filter, and the row whose detail panel offers "Price this model".
    unpriced: { provider: "groq", model: "llama-parity-unpriced" },
    // Owned by the bulk-delete test, which consumes it. Kept off the two models
    // above so deleting it cannot make an earlier assertion unreproducible.
    scratch: { provider: "groq", model: "scratch-parity-model" },
  },
} as const;

export const PRICED_MODEL_KEY = `${PARITY.models.priced.provider}:${PARITY.models.priced.model}`;
export const UNPRICED_MODEL_KEY = `${PARITY.models.unpriced.provider}:${PARITY.models.unpriced.model}`;

// Row counts, exported so the specs assert against the fixture rather than
// against a number repeated in three files.
export const COUNTS = {
  priced: 30,
  unpriced: 6,
  errors: 3,
  scratch: 4,
} as const;

// Every seeded row lands inside this many hours of now, so all of them fall in
// Activity's default 24h window (and therefore in Usage's default 30d one).
const WINDOW_HOURS = 20;

interface SeedEvent {
  source_event_id: string;
  timestamp: string;
  provider: string;
  model: string;
  status?: "success" | "error";
  input_tokens?: number;
  output_tokens?: number;
  cache_read_tokens?: number;
  cache_write_tokens?: number;
  duration_ms?: number;
  session_label?: string;
  user_id?: string;
}

// Spread `count` events across the window, newest first. A fixed step (rather
// than a random or a clock-derived offset) keeps the histogram, the bucket
// counts and the paging order identical on every run.
function spread(count: number, index: number): string {
  const step = (WINDOW_HOURS * 3_600_000) / (count + 1);
  return new Date(Date.now() - (index + 1) * step).toISOString();
}

function pricedEvents(): SeedEvent[] {
  return Array.from({ length: COUNTS.priced }, (_, i) => ({
    source_event_id: `parity-priced-${i}`,
    timestamp: spread(COUNTS.priced, i),
    ...PARITY.models.priced,
    // Cache buckets on every priced row so the Tokens column renders its
    // composition bar rather than a bare total.
    input_tokens: 10_000,
    output_tokens: 2_000,
    cache_read_tokens: 4_000,
    cache_write_tokens: 1_000,
    duration_ms: 420 + i,
    session_label: PARITY.sessions.heavy,
    user_id: PARITY.users.heavy,
  }));
}

function unpricedEvents(): SeedEvent[] {
  return Array.from({ length: COUNTS.unpriced }, (_, i) => ({
    source_event_id: `parity-unpriced-${i}`,
    timestamp: spread(COUNTS.unpriced, i),
    ...PARITY.models.unpriced,
    input_tokens: 800,
    output_tokens: 150,
    duration_ms: 1_800 + i,
    session_label: PARITY.sessions.light,
    user_id: PARITY.users.light,
  }));
}

// Failures carry no tokens: an error before the provider replied is the shape
// the Activity page's null-usage branch exists for.
function errorEvents(): SeedEvent[] {
  return Array.from({ length: COUNTS.errors }, (_, i) => ({
    source_event_id: `parity-error-${i}`,
    timestamp: spread(COUNTS.errors, i),
    ...PARITY.models.unpriced,
    status: "error" as const,
    duration_ms: 90 + i,
    session_label: PARITY.sessions.light,
    user_id: PARITY.users.light,
  }));
}

function scratchEvents(): SeedEvent[] {
  return Array.from({ length: COUNTS.scratch }, (_, i) => ({
    source_event_id: `parity-scratch-${i}`,
    timestamp: spread(COUNTS.scratch, i),
    ...PARITY.models.scratch,
    input_tokens: 100,
    output_tokens: 20,
    duration_ms: 200 + i,
    user_id: PARITY.users.light,
  }));
}

// Create the users the imported rows attribute to. Ingestion rejects usage for a
// user that does not exist, and a re-run against a warm database must not fail on
// its own leftovers, so an existing user is accepted.
//
// 409 only, never 400: the route answers 409 for "already exists" and keeps 400
// for a rejected user_id, so accepting both would swallow a malformed id here and
// re-report it as a row count that is short by half, three files later.
async function ensureUsers(page: Page): Promise<void> {
  for (const userId of Object.values(PARITY.users)) {
    const created = await page.request.post("/v1/users", {
      headers: authHeaders,
      data: { user_id: userId },
    });
    if (created.status() !== 409) {
      await expectOk(created, `create user ${userId}`);
    }
  }
}

// A fixed instant, well before any seeded row. Backdated because an imported
// event is priced at the rate in force at *its* timestamp: a price effective from
// now would leave every backdated row uncosted, and the Priced filter would then
// have no positive case at all. Fixed rather than derived from the clock because
// a pricing row is keyed by `effective_at`, so a clock-derived value writes a new
// revision on every run instead of reusing the one already there.
const PRICE_EFFECTIVE_AT = "2020-01-01T00:00:00.000Z";

// Price one of the two models.
async function ensurePricing(page: Page): Promise<void> {
  const priced = await page.request.post("/v1/pricing", {
    headers: authHeaders,
    data: {
      model_key: PRICED_MODEL_KEY,
      input_price_per_million: 3,
      output_price_per_million: 15,
      cache_read_price_per_million: 0.3,
      cache_write_price_per_million: 3.75,
      effective_at: PRICE_EFFECTIVE_AT,
    },
  });
  await expectOk(priced, `price ${PRICED_MODEL_KEY}`);
}

/**
 * Seed the parity fixture. Idempotent: ingestion is keyed on
 * `(source, source_event_id)`, so a re-run against a warm database reports
 * duplicates instead of doubling the row counts the specs assert on.
 */
export async function seedParityUsage(page: Page): Promise<void> {
  await ensureUsers(page);
  await ensurePricing(page);

  const events = [...pricedEvents(), ...unpricedEvents(), ...errorEvents(), ...scratchEvents()];
  const seeded = await page.request.post("/v1/usage/external-events", {
    headers: authHeaders,
    data: { source: PARITY.source, events },
  });
  await expectOk(seeded, "seed parity usage");

  // A per-event rejection is reported inside a 200, so the status alone does not
  // say the fixture landed: an event the ingest could not attribute leaves the
  // counts short and surfaces three files later as an unexplained off-by-N.
  const result = (await seeded.json()) as {
    accepted: number;
    duplicate: number;
    rejected: number;
    errors: unknown[];
  };
  expect(result.rejected, `rejected events: ${JSON.stringify(result.errors)}`).toBe(0);
  expect(result.accepted + result.duplicate, "every seeded event has to be stored or already present").toBe(
    events.length,
  );
}
