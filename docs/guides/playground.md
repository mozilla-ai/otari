# Playground

The Playground lets you test different AI models directly in Otari, experiment with prompts, and compare responses. You can evaluate model performance, including response quality, token usage, latency, and cost, to help you choose the best model for your needs.

## Test a single model

1. From your workspace, click **Playground** in the top navigation menu.
2. Make sure **Single** is selected.
3. Choose a model from the dropdown menu below the prompt box.
4. Enter your prompt or select one of the suggested Starting Points.
5. Optionally, click **+ Tools** to add tools to your request.
6. Press Enter or click the Send button to submit your prompt.
7. Review the model's response and available usage and performance information.

## Compare two models

Use Compare mode to evaluate how two different models respond to the same prompt.

1. Click **Compare** at the top of the Playground.
2. Select Model A and Model B using the dropdown menus.
3. Enter your prompt in the text box at the bottom.
4. Optionally, click **+ Tools** to include tools in your request.
5. Submit your prompt to see both models' responses side by side.
6. Compare their answers and available performance metrics to determine which model best suits your needs.

## View your history

You can revisit previous conversations and model comparisons through the Playground's History feature.

1. Click **History** in the top-right corner.
2. Filter your history by All, Chats, or Comparisons.
3. Use the **Search history** field to find a specific conversation or comparison.
4. Select a saved entry to revisit it, or click View all to access your complete history.

Your saved Playground history is private and visible only to you.

## How Playground requests work

The Playground is the in-product chat page: pick a model the gateway serves and
talk to it, or put two side by side and record which answered better. Requests
run through the same path as any other completion, so routing policies,
guardrails, budgets and the tool loop all apply, and every request appears in
Activity and Usage.

A question can carry files. Attach them with the paperclip in the composer or
drop them anywhere on the page; each uploads at once and shows as a chip with
its name and size until the question is sent. The model reads them the way an
API request's [uploaded files](files.md) are read, so a text-only local model is
given their extracted text. When comparing, both models get the same files. A
saved conversation remembers which files each question carried. Removing a
chip takes the file off that question and keeps the upload: the Files dialog
lists everything you uploaded in the workspace, to attach again or to delete.
These controls appear only where the deployment stores files: not when file
uploads are off, and not on a hosted control plane.

The Tools menu attaches web search, code execution and any number of MCP
servers to the next message, in any mix: the model is offered all of them in one
tool loop. A tool the gateway runs shows in the transcript above the reply as a
collapsed row with its status, opening to the arguments the model sent and what
the tool returned (a long result is shortened for display). This activity is
shown in the Playground only (not on a hosted control plane, whose requests run
on the data plane) and is not saved with a conversation; a request
made through the API still returns just the final message, as
[What clients receive](tools.md#what-clients-receive) describes.

Two things are worth knowing about how it is billed and what it stores.

A Playground request carries no API key. It is authorized by the dashboard
session and billed to whoever sent it, in the workspace the switcher has
selected, against that person's own budget and rate limit; its usage row
therefore has no key attached. Picking a different model changes which provider
credential the gateway resolves, never who pays. A request made here also does
not count as the workspace's first request, so the setup guide keeps waiting for
one from outside the product, which is the milestone it exists to mark.

Nothing is stored until asked for. Saving a conversation or recording a
comparison prompts once for content retention, and the flags are per person and
revocable; withdrawing one blocks new saves and deletes nothing. Saved
transcripts, recorded comparisons and pinned models are visible only to the
person who created them, even to colleagues in the same workspace, and each can
be deleted from the page. Hosted control planes serve no inference, so they do
not offer the page at all.
