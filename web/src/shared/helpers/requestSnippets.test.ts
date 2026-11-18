import { afterEach, describe, expect, it, vi } from "vitest"
import { API_ROOT } from "@/shared/api/client"
import type { RequestPolicy } from "@/shared/api/requestPolicy"
import {
  buildAgentPrompt,
  buildCurlSnippet,
  buildPythonSnippet,
  buildTypescriptSnippet,
  resolveSnippetBaseUrl,
  SNIPPET_MODEL_PLACEHOLDER,
} from "@/shared/helpers/requestSnippets"

// Mutable so a case can stand in for a build that reaches several deployments.
const policy = vi.hoisted(
  (): RequestPolicy => ({ origin: "", credentials: "same-origin" }),
)
vi.mock("@/shared/api/overlayRequestPolicy", () => ({
  prepareRequests: async () => {},
  requestPolicy: () => policy,
}))

afterEach(() => {
  policy.origin = ""
  policy.credentials = "same-origin"
  policy.publicOrigin = undefined
})

const INPUT = {
  baseUrl: "https://otari.example.com",
  apiKey: "gw-abc123",
  model: "openai:gpt-4o-mini",
}

/**
 * The `-d` argument as the shell would hand it to cURL: outer single quotes
 * dropped, and each `'\''` sequence back to one apostrophe.
 */
function shellPayload(snippet: string): string {
  const quoted = snippet.slice(snippet.indexOf("-d '") + 3)
  return quoted.slice(1, quoted.lastIndexOf("'")).replaceAll(`'\\''`, "'")
}

describe("buildCurlSnippet", () => {
  it("posts to the completions path on the base URL it was given", () => {
    expect(buildCurlSnippet(INPUT)).toContain(
      `https://otari.example.com${API_ROOT}/chat/completions`,
    )
  })

  it("exports the key, then sends the variable as a Bearer token", () => {
    const lines = buildCurlSnippet(INPUT).split("\n")
    expect(lines[0]).toBe("export OTARI_API_KEY=gw-abc123")
    expect(lines).toContain('  -H "Authorization: Bearer $OTARI_API_KEY" \\')
    expect(lines.slice(1).join("\n")).not.toContain("gw-abc123")
  })

  it("sets nothing when the reader has already exported the key", () => {
    const snippet = buildCurlSnippet({ ...INPUT, apiKey: "$OTARI_API_KEY" })
    expect(snippet).not.toContain("export")
    expect(snippet.split("\n")[0]).toMatch(/^curl /)
  })

  it("quotes the concealed stand-in", () => {
    const snippet = buildCurlSnippet({ ...INPUT, apiKey: "otk_ab…••••" })
    expect(snippet.split("\n")[0]).toBe("export OTARI_API_KEY='otk_ab…••••'")
  })

  it("quotes a key the shell would otherwise split", () => {
    const snippet = buildCurlSnippet({ ...INPUT, apiKey: "a b'c" })
    expect(snippet.split("\n")[0]).toBe(`export OTARI_API_KEY='a b'\\''c'`)
  })

  it("names the model it was given", () => {
    expect(buildCurlSnippet(INPUT)).toContain('"model": "openai:gpt-4o-mini"')
  })

  it("falls back to the placeholder when the deployment has no model to name", () => {
    const snippet = buildCurlSnippet({ ...INPUT, model: undefined })
    expect(snippet).toContain(`"model": "${SNIPPET_MODEL_PLACEHOLDER}"`)
  })

  it("escapes a model id that would otherwise break the JSON body", () => {
    // A model id is whatever the provider's catalog says it is. Quotes,
    // backslashes and newlines are the three that would leave a snippet the
    // guide advertises as runnable unable to parse.
    const hostile = 'weird"model\\name\nv2'
    const snippet = buildCurlSnippet({ ...INPUT, model: hostile })

    expect(JSON.parse(shellPayload(snippet))).toEqual({
      model: hostile,
      messages: [{ role: "user", content: "Hello" }],
    })
  })

  it("survives an apostrophe, which would otherwise close the shell quote", () => {
    const snippet = buildCurlSnippet({ ...INPUT, model: "someone's-model" })

    expect(snippet).toContain(`someone'\\''s-model`)
    expect(JSON.parse(shellPayload(snippet))).toEqual({
      model: "someone's-model",
      messages: [{ role: "user", content: "Hello" }],
    })
  })

  it("quotes the base URL as one shell word, whatever it holds", () => {
    // A space would otherwise split the URL into two shell words and a quote
    // would end the argument early (otari#823).
    const snippet = buildCurlSnippet({
      ...INPUT,
      baseUrl: "https://gw.example/a b'; echo pwned; '",
    })

    // The whole URL is one shell word, so the `;` never reaches the shell as a
    // separator: `'\''` closes the quote, emits a literal apostrophe, reopens.
    expect(snippet.split("\n")[1]).toBe(
      `curl 'https://gw.example/a b'\\''; echo pwned; '\\''${API_ROOT}/chat/completions' \\`,
    )
  })

  it("leaves an ordinary payload unquoted beyond its own single quotes", () => {
    // The escaping is invisible for every value anyone will actually see, which
    // is what keeps the snippet readable.
    expect(buildCurlSnippet(INPUT)).toContain(
      `-d '{"model": "openai:gpt-4o-mini", "messages": [{"role": "user", "content": "Hello"}]}'`,
    )
  })
})

describe("buildPythonSnippet", () => {
  it("installs the client and exports the key, then reads it from the environment", () => {
    expect(buildPythonSnippet(INPUT)).toBe(
      [
        "# pip install openai",
        "# Set OTARI_API_KEY in your environment first",
        "",
        "import os",
        "",
        "from openai import OpenAI",
        "",
        `client = OpenAI(base_url="https://otari.example.com${API_ROOT}", api_key=os.environ["OTARI_API_KEY"])`,
        "resp = client.chat.completions.create(",
        '    model="openai:gpt-4o-mini",',
        '    messages=[{"role": "user", "content": "Hello"}],',
        ")",
        "print(resp.choices[0].message.content)",
      ].join("\n"),
    )
  })

  it("reads a key given as a shell reference from the environment", () => {
    const snippet = buildPythonSnippet({ ...INPUT, apiKey: "$OTARI_API_KEY" })

    // The reader has exported it already, so there is nothing to set up.
    expect(snippet).toMatch(/^import os\n\nfrom openai import OpenAI\n/)
    expect(snippet).toContain('api_key=os.environ["OTARI_API_KEY"])')
  })

  it("points the OpenAI client at the gateway's API root", () => {
    expect(buildPythonSnippet(INPUT)).toContain(
      `client = OpenAI(base_url="https://otari.example.com${API_ROOT}", api_key=`,
    )
  })

  it("carries the model and the message into the call", () => {
    const snippet = buildPythonSnippet({
      ...INPUT,
      message: "Hello from Otari",
    })
    expect(snippet).toContain('model="openai:gpt-4o-mini"')
    expect(snippet).toContain('"content": "Hello from Otari"')
  })

  it("escapes a model id that would otherwise break the Python literal", () => {
    // A JSON string literal is a valid Python one, which is what lets one
    // escaping helper serve both snippets.
    const snippet = buildPythonSnippet({ ...INPUT, model: 'weird"model\\name' })

    expect(snippet).toContain('model="weird\\"model\\\\name"')
  })

  it("escapes a base URL that would otherwise break the Python literal", () => {
    // Same reason as the curl case: the builder escapes whatever it is given
    // (otari#823).
    const snippet = buildPythonSnippet({
      ...INPUT,
      baseUrl: 'https://gw.example/x"',
    })

    expect(snippet).toContain(`base_url="https://gw.example/x\\"${API_ROOT}"`)
  })
})

describe("buildPythonSnippet's key", () => {
  it("never carries the key, not even in a comment", () => {
    expect(buildPythonSnippet(INPUT)).not.toContain("gw-abc123")
  })
})

describe("buildTypescriptSnippet", () => {
  it("installs the client and exports the key, then reads it from the environment", () => {
    expect(buildTypescriptSnippet(INPUT)).toBe(
      [
        "// npm i openai",
        "// Set OTARI_API_KEY in your environment first",
        "",
        'import OpenAI from "openai"',
        "",
        `const client = new OpenAI({ baseURL: "https://otari.example.com${API_ROOT}", apiKey: process.env.OTARI_API_KEY })`,
        "const resp = await client.chat.completions.create({",
        '  model: "openai:gpt-4o-mini",',
        '  messages: [{ role: "user", content: "Hello" }],',
        "})",
        "console.log(resp.choices[0].message.content)",
      ].join("\n"),
    )
  })

  it("never carries the key, not even in a comment", () => {
    expect(buildTypescriptSnippet(INPUT)).not.toContain("gw-abc123")
  })
})

describe("buildAgentPrompt", () => {
  it("names the variable and the Bearer header, never the key", () => {
    const prompt = buildAgentPrompt(INPUT)
    expect(prompt).toContain("Authorization: Bearer $OTARI_API_KEY")
    expect(prompt).not.toContain("gw-abc123")
  })
})

describe("resolveSnippetBaseUrl", () => {
  const ORIGIN = "https://dashboard.example.com"

  it("uses the browser's own origin", () => {
    // One process serves the dashboard and the API, so whatever address
    // reached this page reaches /api/v1/chat/completions.
    expect(resolveSnippetBaseUrl(ORIGIN)).toBe(ORIGIN)
  })

  it("trims a trailing slash, since the caller suffixes the result", () => {
    expect(resolveSnippetBaseUrl(`${ORIGIN}/`)).toBe(ORIGIN)
  })

  it("defaults to the page's own origin under the same-origin policy", () => {
    expect(resolveSnippetBaseUrl()).toBe(window.location.origin)
  })

  it("defaults to the deployment the dashboard's requests reach", () => {
    // A dashboard that reaches several deployments is served from one host and
    // sends every request to the one chosen, so the key just issued belongs to
    // that deployment and a snippet aimed at the page's host would miss it.
    policy.origin = "https://eu.otari.example"
    policy.credentials = "include"

    expect(resolveSnippetBaseUrl()).toBe("https://eu.otari.example")
  })

  it("prefers the public origin over the page's own", () => {
    // The dashboard's requests stay on the page's host, whose edge cuts a long
    // non-streamed request; a program is pointed at the API's own host.
    policy.publicOrigin = "https://api.otari.example"

    expect(resolveSnippetBaseUrl()).toBe("https://api.otari.example")
  })

  it("prefers the public origin over the deployment the requests reach", () => {
    policy.origin = "https://eu.otari.example"
    policy.credentials = "include"
    policy.publicOrigin = "https://eu.api.otari.example"

    expect(resolveSnippetBaseUrl()).toBe("https://eu.api.otari.example")
  })
})
