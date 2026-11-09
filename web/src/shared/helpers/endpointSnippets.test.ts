import { describe, expect, it, vi } from "vitest"
import { API_ROOT } from "@/shared/api/client"
import {
  buildEndpointRequest,
  type CatalogEndpoint,
} from "@/shared/helpers/endpointSnippets"
import {
  buildCurlSnippet,
  buildPythonSnippet,
} from "@/shared/helpers/requestSnippets"

vi.mock("@/shared/api/overlayRequestPolicy", () => ({
  prepareRequests: async () => {},
  requestPolicy: () => ({ origin: "", credentials: "same-origin" }),
}))

const INPUT = {
  baseUrl: "https://otari.example.com",
  apiKey: "gw-abc123",
  model: "openai:some-model",
}
const ROOT = `https://otari.example.com${API_ROOT}`

/** The request for a route the builder covers. */
function requestFor(endpoint: CatalogEndpoint, input = INPUT) {
  const request = buildEndpointRequest(endpoint, input)
  if (!request) throw new Error(`no request for ${endpoint}`)
  return request
}

/**
 * The `-d` argument as the shell would hand it to cURL: outer single quotes
 * dropped, and each `'\''` sequence back to one apostrophe.
 */
function jsonBody(curl: string): unknown {
  const quoted = curl.slice(curl.indexOf("-d '") + 3)
  const end =
    quoted.indexOf("'\n") === -1
      ? quoted.lastIndexOf("'")
      : quoted.indexOf("'\n")
  return JSON.parse(quoted.slice(1, end).replaceAll(`'\\''`, "'"))
}

describe("buildEndpointRequest", () => {
  it("gives the chat route the request the Keys page and the setup guide show", () => {
    const request = requestFor("/chat/completions")
    expect(request.curl).toBe(buildCurlSnippet(INPUT))
    expect(request.python).toBe(buildPythonSnippet(INPUT))
    expect(request.pythonPackage).toBe("openai")
  })

  it.each([
    ["/responses", { model: INPUT.model, input: "Hello" }],
    [
      "/messages",
      {
        model: INPUT.model,
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello" }],
      },
    ],
    ["/embeddings", { model: INPUT.model, input: "Hello" }],
    ["/moderations", { model: INPUT.model, input: "Hello" }],
    [
      "/images/generations",
      { model: INPUT.model, prompt: "A lighthouse on a cliff at dawn" },
    ],
    [
      "/audio/speech",
      { model: INPUT.model, input: "Hello from Otari", voice: "alloy" },
    ],
    [
      "/rerank",
      {
        model: INPUT.model,
        query: "What is the capital of France?",
        documents: [
          "Paris is the capital of France.",
          "Berlin is the capital of Germany.",
        ],
      },
    ],
    [
      "/videos",
      { model: INPUT.model, prompt: "Waves breaking on a beach at sunset" },
    ],
  ] as const)("posts the smallest body %s takes", (endpoint, body) => {
    const { curl } = requestFor(endpoint)
    expect(curl.split("\n")[0]).toBe(`curl '${ROOT}${endpoint}' \\`)
    expect(curl).toContain('-H "Otari-Key: gw-abc123"')
    expect(curl).toContain('-H "Content-Type: application/json"')
    expect(jsonBody(curl)).toEqual(body)
  })

  it("escapes a model id in every JSON body", () => {
    const hostile = `someone's "odd"\\model`
    for (const endpoint of [
      "/responses",
      "/messages",
      "/embeddings",
      "/moderations",
      "/images/generations",
      "/audio/speech",
      "/rerank",
      "/videos",
    ] as const) {
      const { curl, python } = requestFor(endpoint, {
        ...INPUT,
        model: hostile,
      })
      expect(jsonBody(curl)).toMatchObject({ model: hostile })
      expect(python).toContain(JSON.stringify(hostile))
    }
  })

  it("uploads the file a form route reads, with the text fields taken as text", () => {
    const edit = requestFor("/images/edits", { ...INPUT, model: "@odd'model" })
    expect(edit.curl.split("\n")[0]).toBe(`curl '${ROOT}/images/edits' \\`)
    // `--form-string` keeps a leading `@` or `<` in a value from reading a file.
    expect(edit.curl).toContain(`--form-string 'model=@odd'\\''model'`)
    expect(edit.curl).toContain(`--form-string 'prompt=Make the sky purple'`)
    expect(edit.curl).toContain("-F image=@image.png")
    expect(edit.upload).toBe("image.png")
    expect(edit.python).toContain('open("image.png", "rb")')

    const transcription = requestFor("/audio/transcriptions")
    expect(transcription.curl).toContain(
      `--form-string 'model=openai:some-model'`,
    )
    expect(transcription.curl).toContain("-F file=@audio.mp3")
    expect(transcription.upload).toBe("audio.mp3")
    expect(transcription.python).toContain(
      "client.audio.transcriptions.create(",
    )
  })

  it("names no upload for a route that sends none", () => {
    expect(requestFor("/embeddings").upload).toBe("")
  })

  it("writes a binary answer to a file", () => {
    const speech = requestFor("/audio/speech")
    // An error answer is not saved as the audio.
    expect(speech.curl).toContain("--fail \\\n  --output speech.mp3")
    expect(speech.python).toContain('resp.stream_to_file("speech.mp3")')
  })

  it("polls a video job and downloads the file", () => {
    const video = requestFor("/videos")
    expect(video.curl).toContain(`curl '${ROOT}/videos/VIDEO_ID' \\`)
    expect(video.curl).toContain(`curl '${ROOT}/videos/VIDEO_ID/content' \\`)
    // An error answer is not saved as the video.
    expect(video.curl).toContain(
      `curl '${ROOT}/videos/VIDEO_ID/content' \\\n  -H "Otari-Key: gw-abc123" \\\n  --fail \\\n  --output video.mp4`,
    )
    // The OpenAI SDK posts a video as a form, which the route refuses.
    expect(video.pythonPackage).toBe("httpx")
    expect(video.python).toContain(`base_url="${ROOT}"`)
    expect(video.python).toContain('json={"model": "openai:some-model"')
    // Each pass refreshes the job, and the wait has a deadline.
    expect(video.python).toContain("deadline = time.monotonic() + 600\n")
    expect(video.python).toContain(
      'while video["status"] in ("queued", "in_progress") and time.monotonic() < deadline:\n' +
        "    time.sleep(5)\n" +
        `    video = client.get(f"/videos/{video['id']}").raise_for_status().json()`,
    )
    expect(video.python).toContain('with open("video.mp4", "wb") as f:')
    // A job still running at the deadline names itself, to check later.
    expect(video.python).toContain(
      `print(f"Still {video['status']} after 10 minutes; check GET /videos/{video['id']} later.")`,
    )
  })

  it("asks a Gemini or Orpheus speech model for one of its own voices", () => {
    const gemini = requestFor("/audio/speech", {
      ...INPUT,
      model: "google:gemini-2.5-flash-preview-tts",
    })
    expect(gemini.voice).toBe("Kore")
    expect(jsonBody(gemini.curl)).toMatchObject({ voice: "Kore" })
    expect(gemini.python).toContain('voice="Kore",')

    const orpheus = requestFor("/audio/speech", {
      ...INPUT,
      model: "groq:canopylabs/orpheus-v1-english",
    })
    expect(jsonBody(orpheus.curl)).toMatchObject({ voice: "troy" })
    expect(orpheus.python).toContain('voice="troy",')

    const openai = requestFor("/audio/speech")
    expect(openai.voice).toBe("alloy")
    expect(openai.python).toContain('voice="alloy",')
    expect(requestFor("/embeddings").voice).toBe("")
  })

  it("reads a key given as a shell reference from the environment", () => {
    const fromEnv = { ...INPUT, apiKey: "$OTARI_API_KEY" }
    const key = 'os.environ["OTARI_API_KEY"]'

    const responses = requestFor("/responses", fromEnv)
    expect(responses.python).toMatch(/^import os\n\nfrom openai import OpenAI/)
    expect(responses.python).toContain(`api_key=${key})`)
    // The shell expands it, so cURL keeps the reference.
    expect(responses.curl).toContain('-H "Otari-Key: $OTARI_API_KEY"')

    expect(requestFor("/images/generations", fromEnv).python).toMatch(
      /^import base64\nimport os\n\nfrom openai import OpenAI/,
    )
    expect(requestFor("/messages", fromEnv).python).toContain(`api_key=${key})`)
    expect(requestFor("/rerank", fromEnv).python).toContain(
      `headers={"Otari-Key": ${key}}`,
    )
    expect(requestFor("/videos", fromEnv).python).toMatch(
      /^import os\nimport time\n\nimport httpx/,
    )
    expect(requestFor("/chat/completions", fromEnv).python).toContain(
      `api_key=${key})`,
    )
  })

  it("keeps a real key as a string, with no import for it", () => {
    const { python } = requestFor("/responses")
    expect(python).toMatch(/^from openai import OpenAI/)
    expect(python).toContain('api_key="gw-abc123")')
  })

  it("calls each route through the SDK that has it", () => {
    expect(requestFor("/responses").python).toContain(
      "client.responses.create(",
    )
    expect(requestFor("/embeddings").python).toContain(
      "client.embeddings.create(",
    )
    expect(requestFor("/moderations").python).toContain(
      "client.moderations.create(",
    )
    expect(requestFor("/images/generations").python).toContain(
      "client.images.generate(",
    )
    expect(requestFor("/images/edits").python).toContain("client.images.edit(")
    expect(requestFor("/audio/speech").python).toContain(
      "client.audio.speech.with_streaming_response.create(",
    )
  })

  it("points the Anthropic client one level above the API root", () => {
    // The Anthropic SDK appends `/v1/messages` to its base URL itself.
    const messages = requestFor("/messages")
    expect(messages.pythonPackage).toBe("anthropic")
    expect(messages.python).toContain(
      `client = Anthropic(base_url="https://otari.example.com/api", api_key="gw-abc123")`,
    )
    expect(`/api/v1/messages`).toBe(`${API_ROOT}/messages`)
  })

  it("posts a rerank with plain httpx, which has no OpenAI SDK call", () => {
    const rerank = requestFor("/rerank")
    expect(rerank.pythonPackage).toBe("httpx")
    expect(rerank.python).toContain(`"${ROOT}/rerank"`)
    expect(rerank.python).toContain('headers={"Otari-Key": "gw-abc123"}')
  })

  it("quotes the base URL as one shell word on every route", () => {
    const baseUrl = "https://gw.example/a b'; echo pwned; '"
    const quoted = `'https://gw.example/a b'\\''; echo pwned; '\\''${API_ROOT}`
    for (const endpoint of ["/embeddings", "/images/edits", "/videos"] as const)
      expect(requestFor(endpoint, { ...INPUT, baseUrl }).curl).toContain(
        `curl ${quoted}${endpoint}' \\`,
      )
  })

  it("gives no request for a route that takes a decision selector", () => {
    expect(buildEndpointRequest("/decisions", INPUT)).toBeUndefined()
    expect(buildEndpointRequest("/systemone", INPUT)).toBeUndefined()
  })
})
