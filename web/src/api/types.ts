// Response shapes mirror the gateway's Pydantic models in
// src/gateway/api/routes/{models,pricing,usage,health}.py. Keep them in sync.

export interface UsageEntry {
  id: string;
  user_id: string | null;
  api_key_id: string | null;
  timestamp: string;
  model: string;
  provider: string | null;
  endpoint: string;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
  cache_read_tokens: number | null;
  cache_write_tokens: number | null;
  cost: number | null;
  status: string;
  error_message: string | null;
}

// Aggregates computed in the database, so they cover every matching row rather
// than the page /v1/usage returns. Summing a page under-reports as soon as the
// log outgrows the limit, which is why these totals do not come from the client.
export interface UsageTotals {
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost: number;
  errors: number;
}

export interface ModelUsage {
  // "provider:model", or the bare model when no provider was recorded. Matches
  // how /v1/models ids and pricing keys are formed, so callers can join on it.
  key: string;
  model: string;
  provider: string | null;
  requests: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cost: number;
}

export interface UsageSummary {
  totals: UsageTotals;
  // Busiest model first.
  by_model: ModelUsage[];
}

// Identity of the dashboard bundle the gateway is currently serving. Changes
// when the built app changes, so a tab can tell its own code went stale.
export interface DashboardBuild {
  build: string;
  version: string;
}

export interface HealthResponse {
  status: string;
  mode?: string;
  version?: string;
  [key: string]: unknown;
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
  default_pricing: boolean;
  require_pricing: boolean;
}
