---
description: Create guardrails that monitor or block unsafe requests across workspaces.
icon: shield-halved
---

# Manage guardrails

A guardrail checks the content of a request before it reaches a provider, to
catch what breaks your safety or compliance rules: personal data, toxicity,
prompt injection and the like. A guardrail can block the requests it flags, or
only record them, and runs on the whole organization or on chosen workspaces.

Only organization owners and admins can add, change or remove guardrails.

## Configure a guardrail

1. Open **Organization**, then **Guardrails**.
2. Click **Configure guardrail**.
3. Choose the check you want, such as content safety, PII or hallucination,
   then a guardrail from the catalog for it.
4. Give it a name, enter the provider's API key if it needs one, and fill in
   the remaining fields, which depend on the guardrail you chose.
5. Click **Configure guardrail** to save it.

## Mandate a guardrail

A configured guardrail runs on nothing until you mandate it.

1. In **Guardrails**, click **Mandate a guardrail**.
2. Choose the guardrail you configured, or your own guardrail service.
3. Choose the **Mode**: block the requests it flags, or monitor them without
   blocking.
4. Choose what happens **If it can't run**: refuse the request or serve it
   unchecked. Only a blocking guardrail can refuse; a monitoring one always
   serves.
5. Choose where it runs: every workspace, including new ones, or chosen
   workspaces.
6. Click **Mandate a guardrail** to save it.

From then on, the guardrail runs on every request from those workspaces.

## Test a guardrail

Click **Test** on a guardrail's row, enter the text to check, and run the test.
The guardrail evaluates it as it would a real request, and nothing you enter is
saved.

## Change or remove a guardrail

Click **Edit** on a guardrail's row to change its credentials or settings, or
**Remove** to take it away. Remove its mandates first: a guardrail a mandate
still runs cannot be removed.

What each guardrail checks, how mandates combine with the guardrails a request
asks for, and the API for all of this are in
[Inference Guardrails](guardrails.md). Every mandate and definition field is in
the [Inference Guardrails reference](guardrails-reference.md).
