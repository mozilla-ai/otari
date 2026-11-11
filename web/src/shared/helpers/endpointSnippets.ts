/**
 * A request for each route a catalog offering is served on, in cURL and Python.
 *
 * The chat route keeps `requestSnippets.ts`'s own request, so the Keys page, the
 * setup guide and the model drawer teach one call. Every other route sends the
 * smallest body it takes, with the same escaping.
 */

import type { CatalogOffering } from "@/client"
import { API_ROOT } from "@/shared/api/client"
import {
  buildCurlSnippet,
  buildPythonSnippet,
  DEFAULT_MESSAGE,
  literal,
  pythonKey,
  pythonProgram,
  type RequestSnippetInput,
  SNIPPET_MODEL_PLACEHOLDER,
  shellSingleQuoted,
} from "@/shared/helpers/requestSnippets"

export type CatalogEndpoint = NonNullable<CatalogOffering["endpoints"]>[number]

export interface EndpointRequest {
  curl: string
  python: string
  /** What the Python request imports, to `pip install`. */
  pythonPackage: "openai" | "anthropic" | "httpx"
  /** The local file the request uploads, which the reader replaces with their own. */
  upload: string
  /** The voice a speech request asks for, which differs by provider. */
  voice: string
}

type RouteInput = {
  url: string
  apiKey: string
  model: string
  /** The gateway's API root, where the OpenAI client points. */
  apiRoot: string
}
type RouteRequest = Omit<EndpointRequest, "curl" | "python" | "voice"> & {
  curl: (input: RouteInput) => string
  python: (input: RouteInput) => string
}

const IMAGE_PROMPT = "A lighthouse on a cliff at dawn"
const EDIT_PROMPT = "Make the sky purple"
const VIDEO_PROMPT = "Waves breaking on a beach at sunset"
const SPEECH_INPUT = "Hello from Otari"
/**
 * Voices for the speech models whose providers name their own. Every other
 * model gets OpenAI's `alloy`, which most speech APIs copy; the drawer says to
 * swap it where the provider lists others.
 */
const SPEECH_VOICES: readonly [RegExp, string][] = [
  [/gemini/i, "Kore"],
  [/orpheus/i, "troy"],
]
const speechVoice = (model: string): string =>
  SPEECH_VOICES.find(([family]) => family.test(model))?.[1] ?? "alloy"
const RERANK_QUERY = "What is the capital of France?"
const RERANK_DOCUMENTS = [
  "Paris is the capital of France.",
  "Berlin is the capital of Germany.",
]

function curl(url: string, apiKey: string, args: string[] = []): string {
  return [
    `curl ${shellSingleQuoted(url)}`,
    `-H "Otari-Key: ${apiKey}"`,
    ...args,
  ].join(" \\\n  ")
}

function curlJson(
  { url, apiKey }: RouteInput,
  body: string,
  args: string[] = [],
): string {
  return curl(url, apiKey, [
    `-H "Content-Type: application/json"`,
    `-d ${shellSingleQuoted(body)}`,
    ...args,
  ])
}

/** A text field of a form. `--form-string` keeps a leading `@` or `<` from reading a file. */
const formField = (name: string, value: string): string =>
  `--form-string ${shellSingleQuoted(`${name}=${value}`)}`

function openAiClient({ apiRoot, apiKey }: RouteInput): string[] {
  return [
    "from openai import OpenAI",
    "",
    `client = OpenAI(base_url=${literal(apiRoot)}, api_key=${pythonKey(apiKey)})`,
  ]
}

/** Image models answer with base64 data or with a URL, so the request takes either. */
function saveImage(path: string): string[] {
  return [
    "image = resp.data[0]",
    "if image.b64_json:",
    `    with open(${literal(path)}, "wb") as f:`,
    "        f.write(base64.b64decode(image.b64_json))",
    "else:",
    "    print(image.url)",
  ]
}

/** Every route but chat, which keeps the shared request, and the decision routes, which take none. */
const ROUTES: Record<
  Exclude<CatalogEndpoint, "/chat/completions" | "/decisions" | "/systemone">,
  RouteRequest
> = {
  "/responses": {
    pythonPackage: "openai",
    upload: "",
    curl: (input) =>
      curlJson(
        input,
        `{"model": ${literal(input.model)}, "input": ${literal(DEFAULT_MESSAGE)}}`,
      ),
    python: (input) =>
      pythonProgram(
        input.apiKey,
        [],
        [
          ...openAiClient(input),
          "resp = client.responses.create(",
          `    model=${literal(input.model)},`,
          `    input=${literal(DEFAULT_MESSAGE)},`,
          ")",
          "print(resp.output_text)",
        ],
      ),
  },
  "/messages": {
    pythonPackage: "anthropic",
    upload: "",
    curl: (input) =>
      curlJson(
        input,
        `{"model": ${literal(input.model)}, "max_tokens": 1024, "messages": [{"role": "user", "content": ${literal(DEFAULT_MESSAGE)}}]}`,
      ),
    python: (input) =>
      pythonProgram(
        input.apiKey,
        [],
        [
          "from anthropic import Anthropic",
          "",
          // The Anthropic SDK appends `/v1/messages` to its base URL itself.
          `client = Anthropic(base_url=${literal(input.apiRoot.replace(/\/v1$/, ""))}, api_key=${pythonKey(input.apiKey)})`,
          "resp = client.messages.create(",
          `    model=${literal(input.model)},`,
          "    max_tokens=1024,",
          `    messages=[{"role": "user", "content": ${literal(DEFAULT_MESSAGE)}}],`,
          ")",
          "print(resp.content[0].text)",
        ],
      ),
  },
  "/embeddings": {
    pythonPackage: "openai",
    upload: "",
    curl: (input) =>
      curlJson(
        input,
        `{"model": ${literal(input.model)}, "input": ${literal(DEFAULT_MESSAGE)}}`,
      ),
    python: (input) =>
      pythonProgram(
        input.apiKey,
        [],
        [
          ...openAiClient(input),
          "resp = client.embeddings.create(",
          `    model=${literal(input.model)},`,
          `    input=${literal(DEFAULT_MESSAGE)},`,
          ")",
          "print(len(resp.data[0].embedding))",
        ],
      ),
  },
  "/moderations": {
    pythonPackage: "openai",
    upload: "",
    curl: (input) =>
      curlJson(
        input,
        `{"model": ${literal(input.model)}, "input": ${literal(DEFAULT_MESSAGE)}}`,
      ),
    python: (input) =>
      pythonProgram(
        input.apiKey,
        [],
        [
          ...openAiClient(input),
          "resp = client.moderations.create(",
          `    model=${literal(input.model)},`,
          `    input=${literal(DEFAULT_MESSAGE)},`,
          ")",
          "print(resp.results[0].flagged)",
        ],
      ),
  },
  "/images/generations": {
    pythonPackage: "openai",
    upload: "",
    curl: (input) =>
      curlJson(
        input,
        `{"model": ${literal(input.model)}, "prompt": ${literal(IMAGE_PROMPT)}}`,
      ),
    python: (input) =>
      pythonProgram(
        input.apiKey,
        ["base64"],
        [
          ...openAiClient(input),
          "resp = client.images.generate(",
          `    model=${literal(input.model)},`,
          `    prompt=${literal(IMAGE_PROMPT)},`,
          ")",
          ...saveImage("image.png"),
        ],
      ),
  },
  "/images/edits": {
    pythonPackage: "openai",
    upload: "image.png",
    curl: ({ url, apiKey, model }) =>
      curl(url, apiKey, [
        formField("model", model),
        formField("prompt", EDIT_PROMPT),
        "-F image=@image.png",
      ]),
    python: (input) =>
      pythonProgram(
        input.apiKey,
        ["base64"],
        [
          ...openAiClient(input),
          'with open("image.png", "rb") as source:',
          "    resp = client.images.edit(",
          `        model=${literal(input.model)},`,
          "        image=source,",
          `        prompt=${literal(EDIT_PROMPT)},`,
          "    )",
          ...saveImage("edited.png"),
        ],
      ),
  },
  "/audio/speech": {
    pythonPackage: "openai",
    upload: "",
    curl: (input) =>
      curlJson(
        input,
        `{"model": ${literal(input.model)}, "input": ${literal(SPEECH_INPUT)}, "voice": ${literal(speechVoice(input.model))}}`,
        ["--fail", "--output speech.mp3"],
      ),
    python: (input) =>
      pythonProgram(
        input.apiKey,
        [],
        [
          ...openAiClient(input),
          "with client.audio.speech.with_streaming_response.create(",
          `    model=${literal(input.model)},`,
          `    voice=${literal(speechVoice(input.model))},`,
          `    input=${literal(SPEECH_INPUT)},`,
          ") as resp:",
          '    resp.stream_to_file("speech.mp3")',
        ],
      ),
  },
  "/audio/transcriptions": {
    pythonPackage: "openai",
    upload: "audio.mp3",
    curl: ({ url, apiKey, model }) =>
      curl(url, apiKey, [formField("model", model), "-F file=@audio.mp3"]),
    python: (input) =>
      pythonProgram(
        input.apiKey,
        [],
        [
          ...openAiClient(input),
          'with open("audio.mp3", "rb") as audio:',
          "    resp = client.audio.transcriptions.create(",
          `        model=${literal(input.model)},`,
          "        file=audio,",
          "    )",
          "print(resp.text)",
        ],
      ),
  },
  "/rerank": {
    pythonPackage: "httpx",
    upload: "",
    curl: (input) =>
      curlJson(
        input,
        `{"model": ${literal(input.model)}, "query": ${literal(RERANK_QUERY)}, "documents": [${RERANK_DOCUMENTS.map(literal).join(", ")}]}`,
      ),
    python: ({ url, apiKey, model }) =>
      pythonProgram(
        apiKey,
        [],
        [
          "import httpx",
          "",
          "resp = httpx.post(",
          `    ${literal(url)},`,
          `    headers={"Otari-Key": ${pythonKey(apiKey)}},`,
          "    json={",
          `        "model": ${literal(model)},`,
          `        "query": ${literal(RERANK_QUERY)},`,
          `        "documents": [${RERANK_DOCUMENTS.map(literal).join(", ")}],`,
          "    },",
          "    timeout=60,",
          ")",
          "resp.raise_for_status()",
          "print(resp.json())",
        ],
      ),
  },
  "/videos": {
    // Not the OpenAI SDK: its `videos.create` posts a form, and the route takes JSON.
    pythonPackage: "httpx",
    upload: "",
    curl: (input) =>
      [
        curlJson(
          input,
          `{"model": ${literal(input.model)}, "prompt": ${literal(VIDEO_PROMPT)}}`,
        ),
        "",
        '# Poll with the job\'s "id" until its "status" is "completed", then download it:',
        curl(`${input.url}/VIDEO_ID`, input.apiKey),
        curl(`${input.url}/VIDEO_ID/content`, input.apiKey, [
          "--fail",
          "--output video.mp4",
        ]),
      ].join("\n"),
    python: ({ apiRoot, apiKey, model }) =>
      pythonProgram(
        apiKey,
        ["time"],
        [
          "import httpx",
          "",
          "client = httpx.Client(",
          `    base_url=${literal(apiRoot)},`,
          `    headers={"Otari-Key": ${pythonKey(apiKey)}},`,
          "    timeout=60,",
          ")",
          "video = client.post(",
          '    "/videos",',
          `    json={"model": ${literal(model)}, "prompt": ${literal(VIDEO_PROMPT)}},`,
          ").raise_for_status().json()",
          "deadline = time.monotonic() + 600",
          'while video["status"] in ("queued", "in_progress") and time.monotonic() < deadline:',
          "    time.sleep(5)",
          "    video = client.get(f\"/videos/{video['id']}\").raise_for_status().json()",
          'if video["status"] == "completed":',
          "    content = client.get(f\"/videos/{video['id']}/content\").raise_for_status()",
          '    with open("video.mp4", "wb") as f:',
          "        f.write(content.content)",
          'elif video["status"] == "failed":',
          '    print(video["error"])',
          "else:",
          "    print(f\"Still {video['status']} after 10 minutes; check GET /videos/{video['id']} later.\")",
        ],
      ),
  },
}

/**
 * The request for `endpoint`, or `undefined` for a route that takes a
 * `decision_providers` selector rather than the offering's.
 */
export function buildEndpointRequest(
  endpoint: CatalogEndpoint,
  input: RequestSnippetInput,
): EndpointRequest | undefined {
  if (endpoint === "/decisions" || endpoint === "/systemone") return undefined
  if (endpoint === "/chat/completions")
    return {
      curl: buildCurlSnippet(input),
      python: buildPythonSnippet(input),
      pythonPackage: "openai",
      upload: "",
      voice: "",
    }
  const route = ROUTES[endpoint]
  const apiRoot = `${input.baseUrl}${API_ROOT}`
  const routeInput: RouteInput = {
    url: `${apiRoot}${endpoint}`,
    apiKey: input.apiKey,
    model: input.model ?? SNIPPET_MODEL_PLACEHOLDER,
    apiRoot,
  }
  return {
    curl: route.curl(routeInput),
    python: route.python(routeInput),
    pythonPackage: route.pythonPackage,
    upload: route.upload,
    voice: endpoint === "/audio/speech" ? speechVoice(routeInput.model) : "",
  }
}
