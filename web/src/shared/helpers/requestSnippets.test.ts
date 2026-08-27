import { describe, expect, it } from "vitest"

import {
  buildCurlSnippet,
  buildPythonSnippet,
  resolveSnippetBaseUrl,
  SNIPPET_MODEL_PLACEHOLDER,
} from "@/shared/helpers/requestSnippets"

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
      "https://otari.example.com/v1/chat/completions",
    )
  })

  it("authenticates with the header the gateway names itself", () => {
    expect(buildCurlSnippet(INPUT)).toContain('-H "Otari-Key: gw-abc123"')
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

  it("quotes the base URL, which an operator configures and we do not constrain", () => {
    // `data_plane_url` is a server-side setting validated only for scheme and
    // host, so a space in one would otherwise split the URL into two shell
    // words and a quote would end the argument early (otari#823).
    const snippet = buildCurlSnippet({
      ...INPUT,
      baseUrl: "https://gw.example/a b'; echo pwned; '",
    })

    // The whole URL is one shell word, so the `;` never reaches the shell as a
    // separator: `'\''` closes the quote, emits a literal apostrophe, reopens.
    expect(snippet.split("\n")[0]).toBe(
      `curl 'https://gw.example/a b'\\''; echo pwned; '\\''/v1/chat/completions' \\`,
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
  it("points the OpenAI SDK at the gateway", () => {
    expect(buildPythonSnippet(INPUT)).toContain(
      'client = OpenAI(base_url="https://otari.example.com/v1", api_key="gw-abc123")',
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
    // Same reason as the curl case: the value comes from the deployment's
    // config rather than from `window.location.origin` (otari#823).
    const snippet = buildPythonSnippet({
      ...INPUT,
      baseUrl: 'https://gw.example/x"',
    })

    expect(snippet).toContain('base_url="https://gw.example/x\\"/v1"')
  })
})

describe("resolveSnippetBaseUrl", () => {
  const ORIGIN = "https://dashboard.example.com"

  it("uses the browser's own origin on a standalone gateway", () => {
    // One process is both the dashboard and the data plane, so whatever address
    // reached this page reaches /v1/chat/completions.
    expect(
      resolveSnippetBaseUrl(
        { deployment_type: "standalone", data_plane_url: null },
        ORIGIN,
      ),
    ).toBe(ORIGIN)
  })

  it("uses the browser's own origin on a hybrid gateway", () => {
    // A gateway attached to otari.ai *is* the data plane; only its management
    // surface lives elsewhere.
    expect(
      resolveSnippetBaseUrl(
        { deployment_type: "hybrid", data_plane_url: null },
        ORIGIN,
      ),
    ).toBe(ORIGIN)
  })

  it("uses the published data plane on a hosted control plane", () => {
    expect(
      resolveSnippetBaseUrl(
        {
          deployment_type: "hosted",
          data_plane_url: "https://gateway.otari.ai",
        },
        ORIGIN,
      ),
    ).toBe("https://gateway.otari.ai")
  })

  it("answers undefined on a hosted control plane that published none", () => {
    // Never the origin, which is the bug this replaces: the control plane is the
    // one host a request must not be sent to (otari#822).
    expect(
      resolveSnippetBaseUrl(
        { deployment_type: "hosted", data_plane_url: null },
        ORIGIN,
      ),
    ).toBeUndefined()
  })

  it("trims a trailing slash, since the caller suffixes the result", () => {
    expect(
      resolveSnippetBaseUrl(
        {
          deployment_type: "hosted",
          data_plane_url: "https://gateway.otari.ai/",
        },
        ORIGIN,
      ),
    ).toBe("https://gateway.otari.ai")
  })

  it("treats a blank published value as none at all", () => {
    expect(
      resolveSnippetBaseUrl(
        { deployment_type: "hosted", data_plane_url: "   " },
        ORIGIN,
      ),
    ).toBeUndefined()
  })

  it("answers undefined rather than an empty origin off the browser", () => {
    // The default origin is "" where there is no `window`. Answering it would
    // let a caller build `curl /v1/chat/completions`, which reads as a snippet
    // and is not one; absent is the honest answer.
    expect(
      resolveSnippetBaseUrl(
        {
          deployment_type: "standalone",
          data_plane_url: null,
        },
        "",
      ),
    ).toBeUndefined()
  })
})
