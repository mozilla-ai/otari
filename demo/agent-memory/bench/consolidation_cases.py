"""Proposals and the notes memory already holds, with the decision the consolidator should take on each.

Each case: ``existing`` notes (1-based, as the consolidator numbers them), a ``proposal``, and the ``want``ed
action with its 1-based ``target`` (None for add). ``ok`` lists other answers that are fair, such as
``update`` where ``no_change`` is the best one. Many existing notes are alike to the proposal on purpose: a
note that only shares a name or a place with it is not one it replaces.
"""

FREEZE = "Staging is frozen on Fridays from 15:00 because of the weekly release; if a deploy can't wait, ask Lucía in #release first."
DEPLOY = "To deploy the gateway to staging, tag the release, then run make deploy-staging from the repo root."
MIGRATIONS = "Staging runs migrations at startup (auto_migrate), so a failed migration shows up as the gateway never turning healthy."
OLLAMA = "When an Ollama model fails with MissingApiKeyError, register Ollama as a hosted provider."
ALEJANDRO_ROLE = "Alejandro leads the platform team and reviews infrastructure changes."
ALEJANDRO_HOBBY = "Alejandro likes painting with watercolors at the weekend."
MEETING = "Meeting with Alejandro on Nov 10: shared agent memory goes in the gateway as a core feature, with per-workspace keys."
PRICING = "Rerank is billed per search unit from Q1; chat models stay priced per token."
PORT = "The local Postgres for agent memory listens on port 5433."
RETENTION = "Production logs are kept 30 days in SigNoz."


def case(name: str, existing: list[str], proposal: str, want: str, target: int | None = None,
         ok: list[tuple[str, int | None]] | None = None, kind: str = "") -> dict:
    return {"name": name, "existing": existing, "proposal": proposal, "want": (want, target), "ok": ok or [],
            "kind": kind or want}


CASES = [
    # Something none of the notes says, though some share words with it
    case("out of hours vs freeze", [FREEZE, DEPLOY],
         "If staging breaks outside working hours, notify Lucía in #release before acting.", "add"),
    case("rollback vs deploy", [DEPLOY, MIGRATIONS],
         "To roll back a staging deploy, run make rollback-staging with the previous release tag.", "add"),
    case("hobby vs role", [ALEJANDRO_ROLE],
         "Alejandro plays the guitar in a band on Thursdays.", "add"),
    case("role vs hobby", [ALEJANDRO_HOBBY],
         "Alejandro approves every change to the Terraform modules.", "add"),
    case("second meeting", [MEETING],
         "Meeting with Priya on Oct 28: rerank moves to per-search-unit billing in Q1.", "add",
         ok=[("no_change", None)]),
    case("ollama other error", [OLLAMA],
         "When an Ollama model times out on first use, pull it first with ollama pull so the gateway does not wait for the download.", "add"),
    case("prod logs vs dev logs", [RETENTION],
         "Development logs are kept 7 days in SigNoz.", "add"),
    case("other port", [PORT],
         "The local gateway for the memory demo listens on port 8010.", "add"),
    case("unrelated", [FREEZE, PRICING],
         "Biome formats and lints the dashboard; run npx biome check --write before committing.", "add"),
    case("customer region", [PORT, RETENTION],
         "Acme's workspaces must stay in the EU: route their traffic to eu-west-1 providers only.", "add"),
    case("same person other fact", [ALEJANDRO_ROLE, ALEJANDRO_HOBBY],
         "Alejandro is on holiday from December 22 to January 7.", "add"),
    case("deploy prod vs staging", [DEPLOY],
         "A production deploy needs an approved release ticket and a second engineer on call.", "add"),
    # Newer or more precise than one note, which it replaces
    case("tag step added", [DEPLOY, MIGRATIONS],
         "To deploy the gateway to staging, tag the release, push the tag with git push --tags, then run make deploy-staging.", "update", 1),
    case("freeze time moved", [DEPLOY, FREEZE],
         "Staging is now frozen on Fridays from 14:00, not 15:00, because of the weekly release; ask Lucía in #release if it can't wait.", "update", 2,
         ok=[("conflict", 2)]),
    case("port changed", [PORT],
         "The local Postgres for agent memory moved from port 5433 to 5434.", "update", 1, ok=[("conflict", 1)]),
    case("retention extended", [RETENTION, PORT],
         "Production logs are now kept 90 days in SigNoz, up from 30.", "update", 1, ok=[("conflict", 1)]),
    case("ollama more precise", [OLLAMA],
         "When an Ollama model fails with MissingApiKeyError, register Ollama as a hosted provider in the dashboard: providers in config.yml are no longer read.", "update", 1,
         ok=[("no_change", 1)]),
    case("meeting detail", [MEETING, PRICING],
         "Meeting with Alejandro on Nov 10: shared agent memory goes in the gateway as a core feature, with per-workspace keys and a consolidator that decides add, update or conflict.", "update", 1,
         ok=[("no_change", 1)]),
    case("pricing date", [PRICING],
         "From January 15, rerank is billed per search unit; chat models stay priced per token.", "update", 1,
         ok=[("no_change", 1)]),
    case("role broadened", [ALEJANDRO_ROLE, ALEJANDRO_HOBBY],
         "Alejandro leads the platform team and reviews every infrastructure and security change.", "update", 1,
         ok=[("no_change", 1)]),
    # Already said
    case("freeze reworded", [DEPLOY, FREEZE],
         "No staging deploys on Fridays after 3pm (weekly release); urgent ones need Lucía's OK in #release.", "no_change", 2,
         ok=[("update", 2)]),
    case("deploy reworded", [DEPLOY],
         "Staging deploy of the gateway: tag the release and run make deploy-staging in the repository root.", "no_change", 1,
         ok=[("update", 1)]),
    case("hobby reworded", [ALEJANDRO_ROLE, ALEJANDRO_HOBBY],
         "On weekends Alejandro paints with watercolours.", "no_change", 2, ok=[("update", 2)]),
    case("port reworded", [PORT, RETENTION],
         "Agent memory's local Postgres is on 5433.", "no_change", 1, ok=[("update", 1)]),
    case("ollama reworded", [MIGRATIONS, OLLAMA],
         "MissingApiKeyError from an Ollama model means Ollama has to be registered as a hosted provider.", "no_change", 2,
         ok=[("update", 2)]),
    case("migrations reworded", [MIGRATIONS],
         "Because staging migrates on startup, a broken migration looks like a gateway that never gets healthy.", "no_change", 1,
         ok=[("update", 1)]),
    case("retention reworded", [RETENTION],
         "SigNoz keeps production logs for thirty days.", "no_change", 1, ok=[("update", 1)]),
    case("pricing reworded", [MEETING, PRICING],
         "Starting Q1 rerank is charged per search unit while chat stays per token.", "no_change", 2, ok=[("update", 2)]),
    # Contradicts a note, and nothing says which is right
    case("freeze denied", [FREEZE],
         "There is no deploy freeze on staging; deploy whenever you need.", "conflict", 1),
    case("port contradicts", [PORT, RETENTION],
         "The local Postgres for agent memory listens on the default port 5432.", "conflict", 1, ok=[("update", 1)]),
    case("retention contradicts", [RETENTION],
         "Production logs are kept 7 days in SigNoz.", "conflict", 1, ok=[("update", 1)]),
    case("deploy contradicts", [DEPLOY],
         "Never run make deploy-staging by hand: staging deploys only through the CI pipeline.", "conflict", 1,
         ok=[("update", 1), ("add", None)]),
    case("ollama contradicts", [OLLAMA],
         "Do not register Ollama as a hosted provider; declare it in config.yml under providers.", "conflict", 1,
         ok=[("update", 1)]),
    case("lead contradicts", [ALEJANDRO_ROLE, ALEJANDRO_HOBBY],
         "Priya leads the platform team; Alejandro moved to the data team.", "conflict", 1, ok=[("update", 1)]),
]
