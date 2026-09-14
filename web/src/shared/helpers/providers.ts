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
 * Keys are any-llm's own provider ids, read off `AnyLLM.get_supported_providers()`
 * rather than recalled, because a key that is not a real id is a row that never
 * gets its name and nothing fails. The four ids with no widely-used brand
 * spelling of their own are deliberately absent and fall through unchanged.
 *
 * `mzai` and `otari` are in it for the same reason every vendor is. A deployment
 * serving its own models would otherwise have shown them lowercase while every
 * third party got proper casing, which is the bug this map exists to fix,
 * pointed at us.
 *
 * Names only. Vendor logos carry trademark terms of use, so which marks ship
 * and under what terms is a decision this file does not make.
 */
const PROVIDER_DISPLAY_NAMES: Readonly<Record<string, string>> = {
  anthropic: "Anthropic",
  atlascloud: "AtlasCloud",
  // Two different any-llm providers, and they must not read the same: `azure`
  // is the models Azure sells directly, `azureopenai` is Azure OpenAI. Given one
  // label they produced two identical options in the Models provider filter.
  azure: "Azure AI Foundry",
  azureanthropic: "Azure Anthropic",
  azureopenai: "Azure OpenAI",
  bedrock: "Amazon Bedrock",
  cerebras: "Cerebras",
  cohere: "Cohere",
  dashscope: "DashScope",
  databricks: "Databricks",
  deepinfra: "DeepInfra",
  deepseek: "DeepSeek",
  edenai: "Eden AI",
  fireworks: "Fireworks AI",
  gemini: "Google Gemini",
  github: "GitHub Models",
  groq: "Groq",
  huggingface: "Hugging Face",
  inception: "Inception",
  llama: "Llama API",
  llamacpp: "llama.cpp",
  llamafile: "llamafile",
  lmstudio: "LM Studio",
  meta: "Meta",
  minimax: "MiniMax",
  mistral: "Mistral AI",
  moonshot: "Moonshot AI",
  mzai: "Mozilla AI",
  nebius: "Nebius",
  ollama: "Ollama",
  openai: "OpenAI",
  openrouter: "OpenRouter",
  otari: "Otari",
  ovhcloud: "OVHcloud",
  perplexity: "Perplexity",
  portkey: "Portkey",
  qiniu: "Qiniu",
  requesty: "Requesty",
  sagemaker: "Amazon SageMaker",
  sambanova: "SambaNova",
  telnyx: "Telnyx",
  together: "Together AI",
  vertexai: "Vertex AI",
  vertexaianthropic: "Vertex AI Anthropic",
  vllm: "vLLM",
  voyage: "Voyage AI",
  watsonx: "IBM watsonx",
  xai: "xAI",
  zai: "Z.ai",
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
