/**
 * Runnable "make a request" snippets, for the two places that hand out a key.
 *
 * The Keys page shows them on its one-time reveal and the setup guide shows them
 * beside the key it issues, so they live here rather than in either feature: two
 * copies would drift into two dialects of the same call, and an operator who
 * followed one and then the other would be told to authenticate two ways.
 *
 * The base URL is the browser's own origin, because the gateway serves this
 * dashboard: whatever address reached this page is an address that reaches the
 * API, which is more reliable than anything the server could report about itself
 * from behind a proxy.
 */

/** Stands in for a model when the deployment has none to name yet. */
export const SNIPPET_MODEL_PLACEHOLDER = "your-model"

const DEFAULT_MESSAGE = "Hello"

export interface RequestSnippetInput {
  /** Origin the dashboard was served from, with no trailing slash. */
  origin: string
  apiKey: string
  /** Defaults to the placeholder, which the caller's copy then has to explain. */
  model?: string
  message?: string
}

/**
 * cURL, with the key in the canonical `Otari-Key` header.
 *
 * Deliberately not `Authorization: Bearer`, which the gateway also accepts: the
 * header it names itself is the one worth teaching, and it cannot be confused
 * with the master key an operator holds for the management API.
 */
export function buildCurlSnippet({
  origin,
  apiKey,
  model = SNIPPET_MODEL_PLACEHOLDER,
  message = DEFAULT_MESSAGE,
}: RequestSnippetInput): string {
  return [
    `curl ${origin}/v1/chat/completions \\`,
    `  -H "Otari-Key: ${apiKey}" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d '{"model": "${model}", "messages": [{"role": "user", "content": "${message}"}]}'`,
  ].join("\n")
}

/** Python through the OpenAI SDK, which the gateway is wire-compatible with. */
export function buildPythonSnippet({
  origin,
  apiKey,
  model = SNIPPET_MODEL_PLACEHOLDER,
  message = DEFAULT_MESSAGE,
}: RequestSnippetInput): string {
  return [
    "from openai import OpenAI",
    "",
    `client = OpenAI(base_url="${origin}/v1", api_key="${apiKey}")`,
    "resp = client.chat.completions.create(",
    `    model="${model}",`,
    `    messages=[{"role": "user", "content": "${message}"}],`,
    ")",
    "print(resp.choices[0].message.content)",
  ].join("\n")
}
