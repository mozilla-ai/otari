/**
 * Runnable "make a request" snippets, for the two places that hand out a key.
 *
 * The Keys page shows them on its one-time reveal and the setup guide shows them
 * beside the key it issues, so they live here rather than in either feature: two
 * copies would drift into two dialects of the same call, and an operator who
 * followed one and then the other would be told to authenticate two ways.
 *
 * The base URL is the browser's own origin, because the gateway serving this
 * dashboard is also the gateway serving the API; see `resolveSnippetBaseUrl`.
 */

import { API_ROOT } from "@/shared/api/client"

/** Stands in for a model when the deployment has none to name yet. */
export const SNIPPET_MODEL_PLACEHOLDER = "your-model"

const DEFAULT_MESSAGE = "Hello"

/**
 * A string as a quoted literal, escaped.
 *
 * The values here are interpolated into a JSON body and into Python source, and
 * neither is ours to constrain: a model id is whatever the provider's catalog
 * says it is, and the base URL is whatever an operator configured. A quote or a
 * backslash in either would otherwise produce a snippet that does not run. A
 * JSON string literal is also a valid Python one (same quoting, same escapes),
 * so one helper covers both.
 *
 * The shell is a second layer and is handled by ``shellSingleQuoted`` below.
 */
const literal = (value: string): string => JSON.stringify(value)

/**
 * A word as one single-quoted shell argument.
 *
 * cURL's arguments are shell words, so JSON escaping alone is not enough: an
 * apostrophe in a model id would close the quote and leave the rest of the
 * payload as shell words, and a space in a configured base URL would split the
 * URL into two arguments. `'\''` is the portable way through that (close the
 * quote, emit an escaped one, reopen), and it leaves a word without apostrophes
 * byte-identical, which is every word anyone will actually see.
 */
const shellSingleQuoted = (word: string): string =>
  `'${word.replaceAll("'", `'\\''`)}'`

export interface RequestSnippetInput {
  /** Where this deployment's API lives, with no trailing slash. */
  baseUrl: string
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
  baseUrl,
  apiKey,
  model = SNIPPET_MODEL_PLACEHOLDER,
  message = DEFAULT_MESSAGE,
}: RequestSnippetInput): string {
  const body = `{"model": ${literal(model)}, "messages": [{"role": "user", "content": ${literal(message)}}]}`
  return [
    `curl ${shellSingleQuoted(`${baseUrl}${API_ROOT}/chat/completions`)} \\`,
    `  -H "Otari-Key: ${apiKey}" \\`,
    `  -H "Content-Type: application/json" \\`,
    `  -d ${shellSingleQuoted(body)}`,
  ].join("\n")
}

/**
 * Python through the Otari SDK (`pip install otari`), which sends the key in
 * the same `Otari-Key` header the cURL snippet teaches.
 *
 * `api_base` is the gateway origin with no path: the SDK appends `/api/v1`
 * itself and rejects a value that already carries it.
 */
export function buildPythonSnippet({
  baseUrl,
  apiKey,
  model = SNIPPET_MODEL_PLACEHOLDER,
  message = DEFAULT_MESSAGE,
}: RequestSnippetInput): string {
  return [
    "from otari import OtariClient",
    "",
    `client = OtariClient(api_base=${literal(baseUrl)}, api_key="${apiKey}")`,
    "resp = client.completion(",
    `    model=${literal(model)},`,
    `    messages=[{"role": "user", "content": ${literal(message)}}],`,
    ")",
    "print(resp.choices[0].message.content)",
  ].join("\n")
}

/** TypeScript through the Otari SDK (`@mozilla-ai/otari`), the same call the Python snippet makes. */
export function buildTypescriptSnippet({
  baseUrl,
  apiKey,
  model = SNIPPET_MODEL_PLACEHOLDER,
  message = DEFAULT_MESSAGE,
}: RequestSnippetInput): string {
  return [
    'import { OtariClient } from "@mozilla-ai/otari"',
    "",
    `const client = new OtariClient({ apiBase: ${literal(baseUrl)}, apiKey: "${apiKey}" })`,
    "const resp = await client.completion({",
    `  model: ${literal(model)},`,
    `  messages: [{ role: "user", content: ${literal(message)} }],`,
    "})",
    "console.log(resp.choices[0].message.content)",
  ].join("\n")
}

/**
 * The environment variable the agent prompt points at instead of the key.
 *
 * The prompt tells the agent to read the key from the environment and never
 * print or hardcode it, so pasting the live secret into the same text would
 * contradict the instruction and put the key in the agent's transcript. The
 * runnable snippets still inline it, because there it is the thing being run.
 */
export const SNIPPET_KEY_ENV_VAR = "OTARI_API_KEY"

/**
 * A prompt to paste into a coding agent, for the operator whose first request
 * is going to be made by one.
 *
 * Deliberately not a snippet with the key in it. It names the variable, the
 * base URL and the model, tells the agent where to read the secret from, and
 * tells it to stop rather than guess if the variable is missing. The last line
 * asks for the reply, which is what makes a failed call visible in the agent's
 * own output rather than only here.
 */
export function buildAgentPrompt({
  baseUrl,
  model = SNIPPET_MODEL_PLACEHOLDER,
  message = DEFAULT_MESSAGE,
}: Omit<RequestSnippetInput, "apiKey">): string {
  const endpoint = `${baseUrl}${API_ROOT}/chat/completions`
  return [
    "Call Otari once with an OpenAI-compatible chat completion.",
    "Do not change any of my editor or IDE settings.",
    "",
    `base URL    ${baseUrl}${API_ROOT}`,
    `env var     ${SNIPPET_KEY_ENV_VAR}`,
    `model       ${model}`,
    "",
    `Read ${SNIPPET_KEY_ENV_VAR} from my environment. Never request, print, or hardcode it.`,
    "If it is not set, stop and tell me how to export it.",
    "",
    `POST ${endpoint}`,
    `Otari-Key: $${SNIPPET_KEY_ENV_VAR}`,
    `Content-Type: application/json`,
    `Body: {"model": ${literal(model)}, "messages": [{"role": "user", "content": ${literal(message)}}]}`,
    "",
    "Show me the assistant's reply.",
  ].join("\n")
}

/**
 * Where a snippet should send its request: the address that served this page.
 *
 * The gateway serves the API on the same origin as the dashboard, and the
 * browser's own origin is more trustworthy than anything the server could say
 * about itself from behind a proxy.
 */
export function resolveSnippetBaseUrl(
  origin: string = typeof window === "undefined" ? "" : window.location.origin,
): string {
  return origin.trim().replace(/\/+$/, "")
}
