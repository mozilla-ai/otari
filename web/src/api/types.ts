// Response shapes mirror the gateway's Pydantic models in
// src/gateway/api/routes/{models,pricing}.py. Keep them in sync.

// Identity of the dashboard bundle the gateway is currently serving. Changes
// when the built app changes, so a tab can tell its own code went stale.
export interface DashboardBuild {
  build: string;
  version: string;
}

export interface ModelPricingInfo {
  input_price_per_million: number;
  output_price_per_million: number;
}

export interface ModelObject {
  id: string;
  object: string;
  created: number;
  owned_by: string;
  pricing: ModelPricingInfo | null;
  // "configured" (DB price), "default" (genai-prices fallback), or "none".
  pricing_source: string;
  // Context-window token limit from the bundled genai-prices dataset, or null
  // when the dataset does not know the model or lists no window for it.
  context_window: number | null;
}

export interface ModelListResponse {
  object: string;
  data: ModelObject[];
}

// One model a provider reports as available. `key` is the selector to send as
// `model`; `id` is the bare id the provider uses.
export interface DiscoverableModel {
  id: string;
  key: string;
}

// A provider's discovery result. `ok` false means the instance could not be
// queried at all, so an empty list is a failure to report rather than a provider
// that genuinely serves nothing.
export interface DiscoverableProvider {
  provider: string;
  ok: boolean;
  error: string | null;
  models: DiscoverableModel[];
}

export interface DiscoverableModelsResponse {
  providers: DiscoverableProvider[];
}

// A model alias. "config" aliases come from config.yml and are read-only here;
// "stored" ones live in the database and can be created and deleted.
export interface AliasResponse {
  name: string;
  target: string;
  source: "config" | "stored";
  created_at: string | null;
  updated_at: string | null;
}

export interface CreateAliasRequest {
  name: string;
  target: string;
}

// Curated capability flags for a provider, from the bundled any-llm metadata.
// True means the provider (not necessarily every model it serves) supports it.
export interface ProviderCapabilities {
  streaming: boolean;
  reasoning: boolean;
  vision: boolean;
  pdf: boolean;
  embeddings: boolean;
  image_generation: boolean;
  audio: boolean;
  rerank: boolean;
  responses_api: boolean;
  moderation: boolean;
  list_models: boolean;
}

// Static, network-free metadata for one configured provider instance. `instance`
// is the configured key (may differ from `provider_type`, the any-llm backend).
export interface ProviderInfo {
  instance: string;
  provider_type: string;
  name: string;
  doc_url: string | null;
  description: string | null;
  env_key: string | null;
  pricing_urls: string[];
  capabilities: ProviderCapabilities;
}

export interface ProvidersResponse {
  providers: ProviderInfo[];
}

// Per-model metadata from the public models.dev catalog, for the detail panel.
// Fields are best-effort: models.dev does not know every model, and unknown
// values come back null/false/[].
export interface ModelMetadata {
  name: string | null;
  description: string | null;
  family: string | null;
  input_modalities: string[];
  output_modalities: string[];
  reasoning: boolean;
  tool_call: boolean;
  structured_output: boolean;
  attachment: boolean;
  temperature: boolean;
  context_window: number | null;
  max_output_tokens: number | null;
  knowledge_cutoff: string | null;
  release_date: string | null;
  last_updated: string | null;
  open_weights: boolean;
  deprecated: boolean;
  cost_input: number | null;
  cost_output: number | null;
}

export interface ModelMetadataResponse {
  source: string;
  // False when enrichment is disabled or models.dev could not be reached; the
  // map is then empty and the UI shows only what the catalog provides.
  available: boolean;
  // Keyed by `provider:model`.
  models: Record<string, ModelMetadata>;
}

export interface PricingResponse {
  model_key: string;
  effective_at: string;
  input_price_per_million: number;
  output_price_per_million: number;
  created_at: string;
  updated_at: string;
}

export interface SetPricingRequest {
  model_key: string;
  input_price_per_million: number;
  output_price_per_million: number;
  effective_at?: string | null;
}

export interface GatewaySettings {
  mode: string;
  version: string;
  model_discovery: boolean;
  default_pricing: boolean;
  require_pricing: boolean;
}

// Toggle one or more runtime settings. Omitted fields are left unchanged.
export interface UpdateSettingsRequest {
  model_discovery?: boolean;
  default_pricing?: boolean;
}
