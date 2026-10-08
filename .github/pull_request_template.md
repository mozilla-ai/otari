## Description
<!-- In plain English, for a reader with no context on this area: what changes
     for someone using Otari, and why. A few sentences. Skip the file paths and
     the class names; the diff and the commits carry those. -->


## How to test it locally
<!-- The steps a reviewer follows to see the change work: what to run, where to
     look, what they should see. Name the automated checks that cover it too, so
     a reviewer knows what is already proven and what is left to eyeball. -->


## PR Type
<!-- Check all that apply -->

- [ ] New Feature
- [ ] Bug Fix
- [ ] Refactor
- [ ] Documentation
- [ ] Infrastructure / CI

## Relevant issues
<!-- e.g. "Fixes #123" -->

## Checklist
<!-- If you delete this checklist, your PR will be automatically closed -->

- [ ] I understand the code I am submitting.
- [ ] I have added or updated tests that cover my change (`tests/unit`, `tests/integration`).
- [ ] I ran the Definition of Done checks locally (`make lint`, `make typecheck`, `make test`).
- [ ] Documentation was updated where necessary.
- [ ] If the API contract changed, I regenerated the OpenAPI spec (`uv run python scripts/generate_openapi.py`).
- [ ] If this changes a rule in `ARCHITECTURE.md` or `scripts/check_architecture.py`, the description names the rule and says why.

## Schema changes
<!-- Delete this section if the PR adds no migration and changes nothing under
     `src/gateway/models/`. Otherwise fill it in: a migration that has run
     against production cannot be reverted the way code can, and the budget and
     usage tables decide refusals and carry money. -->

- [ ] **A named human has reviewed the schema by hand.** An automated review is
      not sign-off for this section. Who: <!-- @handle -->
- [ ] Every data conversion is named below, with the direction of each arm and
      whether any of them makes an existing limit, window or permission *more
      permissive* than it was.
- [ ] `downgrade()` restores what `upgrade()` changed, and a chain test runs
      upgrade, downgrade, upgrade on SQLite (pattern:
      `tests/unit/test_tenancy_schema_chain.py`), and I ran the same on
      PostgreSQL (`uv run alembic upgrade head`, `downgrade -1`, `upgrade head`
      with `OTARI_DATABASE_URL` set).
- [ ] Rows that predate a new non-nullable column get a `server_default` that is
      correct for them, not merely valid.

<!-- What the reviewer should check, in your words. Name the columns, the
     conversions and anything you are unsure about. -->

## AI Usage
<!-- Check one -->

- [ ] No AI was used.
- [ ] AI was used for drafting/refactoring.
- [ ] This is fully AI-generated.

**AI Model/Tool used:**


**Any additional AI details you'd like to share:**


**NOTE:**
When responding to reviewer questions, please respond yourself rather than copy/pasting reviewer comments into an AI and pasting back its answer. We want to discuss with you, not your AI :)

- [ ] I am an AI Agent filling out this form (check box if true)
