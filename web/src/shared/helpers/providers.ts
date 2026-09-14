/**
 * How a provider is named to a reader.
 *
 * A provider id is a wire identifier (`openai`, `mistral`, `together`), and the
 * dashboard has been rendering it verbatim, so vendors appear lowercase and
 * misspelled against their own capitalization (otari#990, otari-ai#2098). The
 * id itself never changes: this is a display layer, and every place a reader
 * needs the id to configure something keeps showing the id.
 *
 * Generic title-casing is not an option and is why this is a map: `openai`
 * title-cases to "Openai", which is wrong in a way that reads as a bug. An id
 * with no entry is returned unchanged rather than guessed at, so a provider
 * added upstream tomorrow is plain rather than misspelled.
 *
 * Names only. Vendor logos carry trademark terms of use, so which marks ship
 * and under what terms is a decision this file does not make.
 */
const PROVIDER_DISPLAY_NAMES: Readonly<Record<string, string>> = {
  ai21: "AI21 Labs",
  anthropic: "Anthropic",
  azure: "Azure OpenAI",
  bedrock: "Amazon Bedrock",
  cerebras: "Cerebras",
  cohere: "Cohere",
  databricks: "Databricks",
  deepseek: "DeepSeek",
  fireworks: "Fireworks AI",
  gemini: "Google Gemini",
  google: "Google",
  groq: "Groq",
  huggingface: "Hugging Face",
  inception: "Inception",
  llamacpp: "llama.cpp",
  llamafile: "llamafile",
  lmstudio: "LM Studio",
  mistral: "Mistral AI",
  moonshot: "Moonshot AI",
  nebius: "Nebius",
  ollama: "Ollama",
  openai: "OpenAI",
  openrouter: "OpenRouter",
  perplexity: "Perplexity",
  portkey: "Portkey",
  sambanova: "SambaNova",
  togetherai: "Together AI",
  vertexai: "Vertex AI",
  voyage: "Voyage AI",
  watsonx: "IBM watsonx",
  xai: "xAI",
}

/**
 * The reader-facing name for a provider id, or the id itself when none is
 * known.
 *
 * Matching is case-insensitive on a trimmed id, because an id reaches the
 * dashboard from several places (a catalog row, a stored credential, a model
 * selector's prefix) and they do not all agree on case.
 */
export function providerDisplayName(providerId: string): string {
  return PROVIDER_DISPLAY_NAMES[providerId.trim().toLowerCase()] ?? providerId
}
