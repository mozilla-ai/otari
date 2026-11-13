"""A fictional team's memory, and questions whose right answers are known, to measure automatic recall on.

Entries are alike on purpose: staging and production deploys, several meetings, people's work and their
hobbies, so that what shares words with a question is not always what answers it. A few are in Spanish.

ENTRIES: id -> (kind, scope, text). Member-scope entries are Alice's own.
CASES: each a question, the user's turns before it, who asks (ALICE or BOB), the entries that answer it
(``want``) and those that are fair to add beside them (``ok``). An empty ``want`` means nothing should be added.
"""

ENTRIES: dict[str, tuple[str, str, str]] = {
    # Staging deploys
    "st1": ("procedure", "workspace", "To deploy the gateway to staging, tag the release, push the tag with git push --tags, then run make deploy-staging from the repo root."),
    "st2": ("fact", "workspace", "Staging runs database migrations at startup (auto_migrate), so a failed migration shows up as the gateway never turning healthy, not as an error in the deploy."),
    "st3": ("procedure", "workspace", "To roll back a staging deploy, run make rollback-staging with the previous release tag; it redeploys that image without running migrations again."),
    "st4": ("fact", "workspace", "The staging gateway health check is /api/v1/health on port 8010; the load balancer marks it unhealthy after three failed checks 10 seconds apart."),
    "st5": ("fact", "workspace", "Staging shares one Postgres instance with the preview environments, so a long migration on staging slows every open preview."),
    "st6": ("decision", "workspace", "Staging is reset from a production snapshot every Monday at 06:00 UTC, with customer content scrubbed."),
    # Production deploys
    "pr1": ("procedure", "workspace", "A production deploy needs an approved release ticket and a second engineer on call; run make deploy-prod REGION=eu-west-1 first, then us-east-1 an hour later if the EU error rate stays flat."),
    "pr2": ("decision", "workspace", "There is a production deploy freeze from December 20 to January 6; only security fixes go out, with approval from the head of engineering."),
    "pr3": ("fact", "workspace", "Production runs on ArgoCD: merging to the release branch syncs the eu-west-1 cluster automatically, while us-east-1 needs a manual sync."),
    "pr4": ("procedure", "workspace", "To roll back production, revert the release branch to the previous tag and let ArgoCD sync; never run migrations backwards in production."),
    "pr5": ("fact", "workspace", "Production database migrations run as a separate Kubernetes job before the new pods start, unlike staging, where they run at startup."),
    # Incidents
    "in1": ("fact", "workspace", "Incident on Oct 3: the Redis cache filled up and evicted rate-limit counters, so some customers briefly got unlimited requests. Fix: a separate Redis for rate limits with noeviction."),
    "in2": ("fact", "workspace", "Incident on Oct 21: the TLS certificate for the public API expired because the renewal job had lost its DNS permissions. Certificates now renew 30 days early and alert at 14 days."),
    "in3": ("procedure", "workspace", "When the gateway returns 502s in a region, first check the provider status pages, then the egress NAT gateway; most 502s in September were an exhausted NAT port pool."),
    "in4": ("decision", "workspace", "After the Oct 3 incident, every postmortem must be written within five working days and reviewed in the Thursday ops meeting."),
    "in5": ("fact", "workspace", "The Nov 4 latency spike in us-east-1 came from a noisy neighbour on the shared node pool; inference pods now run on a dedicated pool."),
    # Architecture decisions
    "ar1": ("decision", "workspace", "Meeting with Alejandro on Nov 10: we agreed shared agent memory goes in the gateway as a core feature, with per-workspace keys and a consolidator that decides add, update or conflict."),
    "ar2": ("decision", "workspace", "Agent memory uses Postgres row-level security keyed on the workspace, so a bug in a query filter cannot return another workspace's rows."),
    "ar3": ("decision", "workspace", "Customer-held keys for agent memory go through the customer's XKS proxy, called directly by the gateway (route B), and are optional per organization."),
    "ar4": ("decision", "workspace", "Memory entries and their embeddings are sealed with AES-256-GCM under a per-workspace data key that rotates every 30 days."),
    "ar5": ("decision", "workspace", "Automatic recall adds what it finds at the end of the newest user message, so the prompt prefix a provider caches stays the same."),
    "ar6": ("decision", "workspace", "We chose Postgres over a dedicated vector database for memory: a workspace holds at most 5,000 entries, so ranking them in the gateway is fast enough."),
    # Meetings
    "mt1": ("decision", "workspace", "Meeting with Priya on Oct 28 about pricing: we keep per-token pricing for chat models and move rerank to per-search-unit billing in Q1."),
    "mt2": ("decision", "workspace", "Weekly sync on Nov 6: Marco owns the dashboard redesign, Lucía the EU data residency work, and the memory demo is due Nov 14."),
    "mt3": ("decision", "workspace", "Roadmap review on Nov 3: Q1 priorities are data residency for EU customers, the memory feature, and SSO for enterprise plans."),
    "mt4": ("decision", "workspace", "Meeting with Tomás on Nov 7: the Stripe migration to usage-based invoices waits until the billing ledger is reconciled."),
    "mt5": ("fact", "workspace", "In the Nov 12 design review, Ana asked for the memory review queue to show who proposed an entry and from which session."),
    # People: their work, and their lives outside it
    "pp1": ("fact", "workspace", "Alejandro likes painting with watercolors at the weekend."),
    "pp2": ("fact", "workspace", "Alejandro leads the platform team and reviews infrastructure changes."),
    "pp3": ("fact", "workspace", "Priya goes bouldering twice a week and organises the team's climbing evenings."),
    "pp4": ("fact", "workspace", "Priya owns pricing and billing, and approves every change to the price list."),
    "pp5": ("fact", "workspace", "Marco plays chess online and is rated around 1800."),
    "pp6": ("fact", "workspace", "Marco works from Lisbon, one hour behind Madrid, and prefers code reviews asynchronously in the pull request."),
    "pp7": ("fact", "workspace", "Lucía is on parental leave from December 1 to March 1; Tomás covers her data residency work."),
    "pp8": ("fact", "workspace", "Tomás bakes sourdough bread and brings it to the office on Fridays."),
    "pp9": ("fact", "workspace", "Ana is the security lead and must approve any change to IAM or key management."),
    # Billing
    "bi1": ("procedure", "workspace", "When Stripe webhooks fail, check the signing secret first: it changes when the endpoint is recreated in the Stripe dashboard."),
    "bi2": ("fact", "workspace", "Invoices are generated on the first day of each month from the usage ledger; a usage row recorded late lands on the next invoice."),
    "bi3": ("decision", "workspace", "Free-trial credits expire after 30 days and are spent before any paid balance."),
    "bi4": ("fact", "workspace", "Models without a price are refused for budgeted traffic unless require_pricing is turned off."),
    # Local development
    "ld1": ("procedure", "workspace", "When an Ollama model fails with MissingApiKeyError, register Ollama as a hosted provider: providers in config.yml are no longer read."),
    "ld2": ("fact", "workspace", "The local Postgres for agent memory runs in the otari-memory-pg container on port 5433, not 5432, to avoid clashing with a system Postgres."),
    "ld3": ("procedure", "workspace", "Run the gateway locally with uv run otari serve -c config.yml; it applies migrations at startup when auto_migrate is true."),
    "ld4": ("fact", "workspace", "Integration tests that use Bedrock fail locally when ~/.aws uses the login credential provider, because botocore then needs the crt extra."),
    "ld5": ("procedure", "workspace", "Regenerate the dashboard's API types with npm run client:generate after scripts/generate_openapi.py writes docs/public/openapi.json."),
    # Dashboard conventions
    "fe1": ("decision", "workspace", "Every Intl formatter in the dashboard is pinned to en-US; a guard test fails the build otherwise."),
    "fe2": ("decision", "workspace", "The dashboard uses react-aria components from the design system; never import a raw HTML input in a feature."),
    "fe3": ("fact", "workspace", "Biome formats and lints the dashboard; run npx biome check --write before committing."),
    # Observability
    "ob1": ("fact", "workspace", "Gateway logs and traces go to SigNoz; development logs are kept 7 days and production logs 30 days."),
    "ob2": ("procedure", "workspace", "To find a slow request, search SigNoz traces by service.name otari-gateway and sort by duration; the trace id is in the Otari-Trace-Id response header."),
    "ob3": ("decision", "workspace", "Alerts page the on-call engineer only for customer-facing errors; everything else goes to the ops alerts channel."),
    # Security
    "se1": ("procedure", "workspace", "If an AWS access key leaks, deactivate it at once, then rotate every secret the key could read, then review CloudTrail for the time it was exposed."),
    "se2": ("decision", "workspace", "Identity Center and SSO configuration live in the infrastructure-identity-management repository, not in the main infrastructure repo."),
    "se3": ("fact", "workspace", "Provider API keys are stored encrypted with the master key; rotating the master key needs the reencrypt endpoint, or stored keys become unreadable."),
    # Customers
    "cu1": ("fact", "workspace", "Acme's workspaces must stay in the EU: route their traffic to eu-west-1 providers only."),
    "cu2": ("fact", "workspace", "Globex requires zero data retention: only providers marked zero_data_retention may serve their requests."),
    "cu3": ("fact", "workspace", "Initech is on the enterprise plan and gets a monthly usage report on the third working day."),
    # Written in Spanish
    "es1": ("procedure", "workspace", "Para rotar la clave de Fireworks, crea una nueva en su consola, actualízala en el proveedor alojado desde el dashboard y borra la antigua cuando el tráfico ya no la use."),
    "es2": ("fact", "workspace", "La demo de memoria compartida se prepara con ./prepare.sh, que borra la memoria, siembra entradas y ejecuta un experimento de 50 sesiones."),
    "es3": ("decision", "workspace", "Decidimos que el juez de relevancia con un modelo de chat sea la opción para las versiones self-hosted sin reranker."),
    "es4": ("fact", "workspace", "Together solo ofrece sus rerankers como endpoint dedicado, que se paga por hora; Fireworks ofrece qwen3-reranker-8b sin endpoint dedicado."),
    # Alice's own
    "ma1": ("fact", "member", "My own staging dashboard is pinned to the eu-west-1 region."),
    "ma2": ("fact", "member", "I keep my notes on the memory feature in a markdown file in my home directory."),
}

MEMORY_TALK = "I've been talking with Alejandro about how agents could share memory across models and sessions."


def case(name: str, question: str, want: list[str], *, ok: list[str] | None = None, earlier: list[str] | None = None,
         who: str = "ALICE", kind: str = "direct") -> dict:
    return {"name": name, "question": question, "want": want, "ok": ok or [], "earlier": earlier or [], "who": who,
            "kind": kind}


CASES = [
    # Asked straight out
    case("staging deploy", "How do I deploy the gateway to staging?", ["st1"]),
    case("staging rollback", "How do I roll back a staging deploy?", ["st3"]),
    case("production rollback", "How do I roll back production?", ["pr4"], kind="near miss"),
    case("prod migrations", "Do database migrations run at startup in production?", ["pr5"], ok=["st2"], kind="near miss"),
    case("rate limit incident", "Why did some customers get unlimited requests in October?", ["in1"]),
    case("502s", "We're getting 502s in eu-west-1, where should I look?", ["in3"]),
    case("stripe webhooks", "Stripe webhooks are failing with signature errors.", ["bi1"]),
    case("late usage", "A customer says usage from the 31st is missing from their invoice.", ["bi2"]),
    case("ollama key", "My Ollama model says MissingApiKeyError.", ["ld1"]),
    case("postgres port", "I can't connect to the memory Postgres on 5432.", ["ld2"]),
    case("bedrock tests", "The Bedrock streaming tests fail on my laptop with a botocore error.", ["ld4"]),
    case("locale", "Can I format numbers with the browser's locale in the dashboard?", ["fe1"]),
    case("leaked key", "I think an AWS key leaked in a commit, what do I do?", ["se1"]),
    case("sso config", "Where do I change the SSO configuration?", ["se2"]),
    case("acme region", "Can Acme's requests go to a US provider?", ["cu1"]),
    case("globex", "Which providers may serve Globex?", ["cu2"]),
    case("vector db", "Why didn't we use a vector database for memory?", ["ar6"]),
    case("encryption", "How are memory entries encrypted?", ["ar4"], ok=["ar3"]),
    case("customer keys", "Can a customer hold the memory encryption key themselves?", ["ar3"], ok=["ar4"]),
    case("q1 priorities", "What are the Q1 priorities?", ["mt3"]),
    case("dashboard owner", "Who owns the dashboard redesign?", ["mt2"]),
    case("rerank pricing", "What did Priya and I decide about rerank pricing?", ["mt1"], ok=["pp4"], kind="meeting"),
    # Follow-ups: the subject is in an earlier turn
    case("holiday freeze", "Is there anything that would block us around the holidays?", ["pr2"],
         earlier=["We're planning the production release for next week."], kind="follow-up"),
    case("certificate", "What happened with the certificate?", ["in2"],
         earlier=["I'm looking into the October incidents."], kind="follow-up"),
    case("tag not found", "what should I run?", ["st1"],
         earlier=["Our staging deploy failed with tag not found."], kind="follow-up"),
    case("last meeting", "What did we decide in our last meeting?", ["ar1"], earlier=[MEMORY_TALK], kind="meeting"),
    case("last meeting, named", "What did Alejandro and I agree in our last meeting?", ["ar1"], earlier=[MEMORY_TALK],
         kind="meeting"),
    case("gift for Tomás", "Any idea what he likes?", ["pp8"], earlier=["I'm preparing a gift for Tomás's birthday."],
         kind="follow-up"),
    case("Marco's reviews", "How does he prefer to get review comments?", ["pp6"],
         earlier=["Marco is reviewing my pull request."], kind="follow-up"),
    # About a person: their work or their life, not both
    case("Alejandro for fun", "What does Alejandro do for fun?", ["pp1"], kind="person"),
    case("infra review", "Who should review my infrastructure change?", ["pp2"], ok=["pp9"], kind="person"),
    case("price list", "I need to change the price list, who approves it?", ["pp4"], ok=["mt1"], kind="person"),
    case("Lucía back", "When is Lucía back from leave?", ["pp7"], kind="person"),
    case("residency cover", "Who is covering the data residency work while Lucía is away?", ["pp7"], ok=["mt2"],
         kind="person"),
    # The subject changes
    case("memory then stripe", "Unrelated: Stripe webhooks are failing, what do I check?", ["bi1"],
         earlier=[MEMORY_TALK], kind="topic change"),
    case("incidents then ollama", "Different thing: my Ollama model says MissingApiKeyError.", ["ld1"],
         earlier=["I'm looking into the October incidents."], kind="topic change"),
    # Across languages
    case("es→en staging", "¿Cómo despliego el gateway en staging?", ["st1"], kind="language"),
    case("es→en certificate", "¿Por qué expiró el certificado en octubre?", ["in2"], kind="language"),
    case("en→es fireworks", "How do I rotate the Fireworks key?", ["es1"], kind="language"),
    case("es→es together", "¿Qué ofrece Together para rerank?", ["es4"], kind="language"),
    # Nothing in memory answers these
    case("haiku", "Write a haiku about autumn.", [], kind="nothing"),
    case("capital", "What's the capital of Australia?", [], kind="nothing"),
    case("thanks", "Thanks, that's all for today!", [], earlier=["Let's talk about the memory feature."], kind="nothing"),
    # Member memory stays its member's
    case("my dashboard, Alice", "Which region is my staging dashboard pinned to?", ["ma1"], kind="privacy"),
    case("my dashboard, Bob", "Which region is my staging dashboard pinned to?", [], who="BOB", kind="privacy"),
]
