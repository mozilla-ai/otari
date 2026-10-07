// Realistic input and output text per trace name, picked by trace index.
export const MOCK_TEXT: Record<string, { input: string; output: string }[]> = {
  "sql-assistant": [
    {
      input: "Which customers spent more than $5,000 last quarter?",
      output:
        "SELECT c.name, SUM(o.total) AS spend FROM customers c JOIN orders o ON o.customer_id = c.id WHERE o.created_at >= date_trunc('quarter', now() - interval '3 months') GROUP BY c.name HAVING SUM(o.total) > 5000;",
    },
    {
      input: "Show weekly signups for the last 8 weeks, split by plan.",
      output:
        "SELECT date_trunc('week', created_at) AS week, plan, COUNT(*) FROM signups WHERE created_at > now() - interval '8 weeks' GROUP BY 1, 2 ORDER BY 1;",
    },
    {
      input:
        "Why is the orders query slow? EXPLAIN shows a seq scan on orders.",
      output:
        "The filter on customer_id has no index. Add CREATE INDEX CONCURRENTLY orders_customer_id_idx ON orders (customer_id); the scan should become an index scan.",
    },
  ],
  "extract-invoice": [
    {
      input:
        "INVOICE #4821  Acme Supplies Ltd  Date: 2026-09-14  2x Standing desk 349.00  1x Monitor arm 89.50  Total EUR 787.50",
      output:
        '{"invoice_number":"4821","vendor":"Acme Supplies Ltd","date":"2026-09-14","currency":"EUR","total":787.5,"line_items":2}',
    },
    {
      input:
        "Rechnung 2026-0097  Nordwind GmbH  Cloud hosting Aug 2026  EUR 1,240.00  VAT 19% 235.60  Gesamt 1,475.60",
      output:
        '{"invoice_number":"2026-0097","vendor":"Nordwind GmbH","currency":"EUR","subtotal":1240,"vat":235.6,"total":1475.6}',
    },
    {
      input:
        "Receipt from Blue Bottle Coffee  2026-10-02  Latte 5.50  Croissant 4.25  Tip 2.00  Total $11.75",
      output:
        '{"vendor":"Blue Bottle Coffee","date":"2026-10-02","currency":"USD","total":11.75,"category":"meals"}',
    },
  ],
  "code-review-bot": [
    {
      input:
        "@@ -41,7 +41,9 @@ def retry(fn, attempts=3):\n-    for _ in range(attempts):\n+    for i in range(attempts):\n         try:\n             return fn()\n-        except Exception:\n-            pass\n+        except Exception as exc:\n+            last = exc",
      output:
        "The loop now keeps the last exception but never re-raises it after the final attempt, so callers get None. Raise `last` once the loop ends.",
    },
    {
      input:
        "diff --git a/src/cache.ts b/src/cache.ts\n+  const key = `${user.id}:${query}`\n+  if (cache.has(key)) return cache.get(key)",
      output:
        "Cache keys include the raw query string, which is unbounded. Hash it or cap the length to avoid memory growth.",
    },
  ],
  "support-agent": [
    {
      input:
        "Hi, I was charged twice for my October subscription. Can you refund one of them?",
      output:
        "I found two charges of $29 on Oct 1. I have refunded the duplicate; it will show on your statement in 3 to 5 days.",
    },
    {
      input: "My API key stopped working after I rotated it yesterday.",
      output:
        "Rotating a key invalidates the old one immediately. Update the key in your deployment settings and redeploy; the new key is active.",
    },
    {
      input: "How do I add a teammate to my workspace?",
      output:
        "Open Members, choose Invite, and enter their email. They will get a link that expires in 7 days.",
    },
  ],
  "summarize-document": [
    {
      input:
        "Summarize the Q3 planning memo (4 pages) for the leadership update.",
      output:
        "Q3 priorities: ship the routing policies beta, cut p95 latency by 30%, and close the EU data residency gap. Risks: two open hires and a vendor contract renewal in November.",
    },
    {
      input: "Give me the key decisions from the incident review doc.",
      output:
        "Decisions: add a circuit breaker to the fallback chain, page the on-call after 3 consecutive 429s, and publish a status note within 15 minutes.",
    },
  ],
  "triage-ticket": [
    {
      input:
        "Subject: Dashboard shows no data since this morning. Body: Our usage page is empty for the last 6 hours.",
      output:
        '{"priority":"high","category":"outage","assignee_team":"platform","reason":"Data gap affecting reporting for 6 hours"}',
    },
    {
      input:
        "Subject: Feature request: dark mode for the playground. Body: Would love this, my eyes hurt at night!",
      output:
        '{"priority":"low","category":"feature-request","assignee_team":"product","reason":"UI preference request"}',
    },
  ],
  "translate-batch": [
    {
      input:
        "Translate to French: 'Your export is ready. Download it from the Reports page.'",
      output: "Votre export est prêt. Téléchargez-le depuis la page Rapports.",
    },
    {
      input:
        "Translate to German: 'We could not process your payment. Please update your card.'",
      output:
        "Wir konnten Ihre Zahlung nicht verarbeiten. Bitte aktualisieren Sie Ihre Karte.",
    },
  ],
  "chat-completion": [
    {
      input:
        "Write a one-line commit message for fixing a null check in the auth middleware.",
      output:
        "fix(auth): guard against a missing session before reading the user",
    },
    {
      input: "What's the difference between a 502 and a 504?",
      output:
        "A 502 means the gateway got an invalid response from upstream; a 504 means upstream did not answer before the timeout.",
    },
    {
      input: "Suggest three names for an internal metrics dashboard.",
      output: "Lighthouse, Vitals, Pulse.",
    },
  ],
}

export function pickText(name: string, index: number) {
  const options = MOCK_TEXT[name] ?? MOCK_TEXT["chat-completion"]
  return options[index % options.length]
}
