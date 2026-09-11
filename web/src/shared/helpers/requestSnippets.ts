/**
 * Runnable "make a request" snippets, for the two places that hand out a key.
 *
 * The Keys page shows them on its one-time reveal and the setup guide shows them
 * beside the key it issues, so they live here rather than in either feature: two
 * copies would drift into two dialects of the same call, and an operator who
 * followed one and then the other would be told to authenticate two ways.
 *
 * The base URL is usually the browser's own origin, because the gateway serving
 * this dashboard is also the gateway serving the API: whatever address reached
 * this page is an address that reaches `/chat/completions`, which is more
 * reliable than anything a server behind a proxy could report about itself.
 *
 * A hosted control plane breaks that, which is what `resolveSnippetBaseUrl`
 * below exists for: it serves the dashboard and is deliberately not where
 * inference belongs (otari#823, otari#822), so it publishes the data-plane
 * gateway's address on `/bootstrap` and the snippets are built from that.
 */

import type { DeploymentBootstrap } from "@/client"
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

/** Python through the OpenAI SDK, which the gateway is wire-compatible with. */
export function buildPythonSnippet({
  baseUrl,
  apiKey,
  model = SNIPPET_MODEL_PLACEHOLDER,
  message = DEFAULT_MESSAGE,
}: RequestSnippetInput): string {
  return [
    "from openai import OpenAI",
    "",
    `client = OpenAI(base_url=${literal(`${baseUrl}/v1`)}, api_key="${apiKey}")`,
    "resp = client.chat.completions.create(",
    `    model=${literal(model)},`,
    `    messages=[{"role": "user", "content": ${literal(message)}}],`,
    ")",
    "print(resp.choices[0].message.content)",
  ].join("\n")
}

/** TypeScript through the OpenAI SDK, the same call the Python snippet makes. */
export function buildTypescriptSnippet({
  baseUrl,
  apiKey,
  model = SNIPPET_MODEL_PLACEHOLDER,
  message = DEFAULT_MESSAGE,
}: RequestSnippetInput): string {
  return [
    'import OpenAI from "openai"',
    "",
    `const client = new OpenAI({ baseURL: ${literal(`${baseUrl}/v1`)}, apiKey: "${apiKey}" })`,
    "const resp = await client.chat.completions.create({",
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
 * Where a snippet should send its request, or undefined when nothing here knows.
 *
 * Three deployments, two answers. A standalone gateway and a hybrid one both
 * serve the API at the address that served this page, so the browser's own
 * origin is the answer and is more trustworthy than anything the server could
 * say about itself from behind a proxy. A hosted control plane is the exception
 * the whole function exists for: it serves this dashboard, and customer
 * inference belongs on the data-plane gateway rather than on it, so it has to
 * name that address itself (`data_plane_url` on `/bootstrap`).
 *
 * Undefined when a hosted deployment names none. Falling back to the origin
 * there is the bug this replaces: it hands somebody a runnable command aimed at
 * the one host their traffic should not reach. A placeholder host would be no
 * better, since nobody reading it can know what to put in its place, so the
 * caller shows no snippet and says why. Undefined and never `""`, so that a
 * caller cannot build `curl /api/v1/chat/completions` out of an origin that is not
 * there; this is the boundary where the bootstrap's `null` becomes the absent
 * value the rest of the tree branches on.
 *
 * `data_plane_url` is read whichever mode published it, but only a hosted
 * deployment does: `bootstrap.py` answers null for standalone and hybrid even
 * when the setting is present, because each of those serves its own API. The
 * trailing slash is trimmed again here because this is the value the gateway
 * published, not the value it validated.
 */
export function resolveSnippetBaseUrl(
  deployment: Pick<DeploymentBootstrap, "deployment_type" | "data_plane_url">,
  origin: string = typeof window === "undefined" ? "" : window.location.origin,
): string | undefined {
  const configured = deployment.data_plane_url?.trim().replace(/\/+$/, "")
  if (configured) return configured
  if (deployment.deployment_type === "hosted") return undefined
  return origin.trim().replace(/\/+$/, "") || undefined
}
