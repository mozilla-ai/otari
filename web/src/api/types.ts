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

// A provider configured at runtime through the dashboard (a row in
// provider_credentials). The API key is never returned, only `last4`.
export interface StoredProvider {
  instance: string;
  provider_type: string | null;
  api_base: string | null;
  last4: string | null;
  client_args: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
  // False when the stored key can't be decrypted with the current OTARI_SECRET_KEY.
  decryptable: boolean;
}

export interface CreateStoredProviderRequest {
  instance: string;
  provider_type?: string | null;
  api_base?: string | null;
  api_key?: string | null;
  client_args?: Record<string, unknown> | null;
}

// Omitted fields are left unchanged; `api_key` rotates the stored key in place.
// `expected_updated_at` guards against clobbering a concurrent edit (412).
export interface UpdateStoredProviderRequest {
  provider_type?: string | null;
  api_base?: string | null;
  api_key?: string | null;
  client_args?: Record<string, unknown> | null;
  expected_updated_at?: string | null;
}

// Result of a live provider connection test (lists the provider's models).
export interface TestProviderResult {
  ok: boolean;
  model_count: number;
  error: string | null;
}

// One provider instance's reachability, from the same model-discovery path the
// per-provider "test connection" uses. `ok` false means unreachable; `error`
// carries the sanitized provider error. `checked_at` is the wall-clock time the
// provider was last dialed (null if never), so a cached status shows honest age.
export interface ProviderHealth {
  instance: string;
  ok: boolean;
  model_count: number;
  error: string | null;
  checked_at: string | null;
}

// Provider connectivity across the gateway. The `healthy` / `total` counts and
// most-recent `checked_at` are precomputed so a summary tile (the overview page,
// issue #302) can reuse them without re-deriving.
export interface ProviderHealthResponse {
  providers: ProviderHealth[];
  healthy: number;
  total: number;
  checked_at: string | null;
}

// A known provider offered in the add-provider picker, with autofill hints.
export interface KnownProvider {
  id: string;
  name: string;
  env_key: string | null;
  default_api_base: string | null;
  requires_api_key: boolean;
  // True when env_key is already set on the server, so a pasted key is optional
  // (any-llm falls back to the environment variable).
  env_key_present: boolean;
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

// An API key row. The full secret is never returned after creation; `key_prefix`
// is a display-only fingerprint (leading chars of the key), null for keys minted
// before the prefix was recorded. Note: providers use `last4` while keys use a
// leading `key_prefix` — a deliberate divergence (the gw-/sk- convention is
// recognized by its prefix), not an inconsistency to "fix".
// `allowed_models` is the per-key model access-list: null = any model
// (unrestricted), [] = deny all, or canonical `instance:model` entries with
// `instance:*` / `instance:prefix*` wildcards. Governs both /v1/models visibility
// and inference.
export interface ApiKey {
  id: string;
  key_prefix: string | null;
  key_name: string | null;
  user_id: string | null;
  created_at: string;
  last_used_at: string | null;
  expires_at: string | null;
  is_active: boolean;
  allowed_models: string[] | null;
  metadata: Record<string, unknown>;
}

export interface CreateKeyRequest {
  key_name?: string | null;
  user_id?: string | null;
  expires_at?: string | null;
  allowed_models?: string[] | null;
  metadata?: Record<string, unknown>;
}

// Returned by create and regenerate: the one and only time the plaintext `key`
// is exposed. Shape matches the gateway's CreateKeyResponse (no last_used_at).
export interface CreateKeyResponse {
  id: string;
  key: string;
  key_prefix: string | null;
  key_name: string | null;
  user_id: string | null;
  created_at: string;
  expires_at: string | null;
  is_active: boolean;
  allowed_models: string[] | null;
  metadata: Record<string, unknown>;
}

// Omitted fields are left unchanged. `allowed_models` is tri-state on the wire:
// omit = unchanged, null = clear to unrestricted, [] = deny all, list = restrict.
export interface UpdateKeyRequest {
  key_name?: string | null;
  is_active?: boolean | null;
  expires_at?: string | null;
  allowed_models?: string[] | null;
  metadata?: Record<string, unknown> | null;
}

// A budget: a reusable spending template (a per-user limit plus an optional
// reset period). Multiple users can share one budget, so the usage fields are an
// aggregate rollup over the users currently assigned to it: how many there are
// and their combined spend/reserved. Assigning users lands with user management,
// so a gateway without assigned users reports zeros here.
export interface Budget {
  budget_id: string;
  name: string | null;
  max_budget: number | null;
  budget_duration_sec: number | null;
  created_at: string;
  updated_at: string;
  user_count: number;
  total_spend: number;
  total_reserved: number;
}

export interface CreateBudgetRequest {
  name?: string | null;
  max_budget?: number | null;
  budget_duration_sec?: number | null;
}

// Omitted fields are left unchanged; `name` is tri-state (omit = unchanged,
// null = clear to unnamed, string = rename).
export interface UpdateBudgetRequest {
  name?: string | null;
  max_budget?: number | null;
  budget_duration_sec?: number | null;
}

// One usage-log row: the metadata for a single API request the gateway served.
// No request or response body is stored, only counts and timing. Surfaced by the
// Activity page and by the per-user usage view.
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
  // Total server-side request duration in ms; null for historical rows and for
  // write paths with no synchronous duration (e.g. batch jobs).
  latency_ms: number | null;
}

// Activity-log filters. All optional; an omitted field means "no filter". Sent as
// query params to /v1/usage and /v1/usage/count.
export interface UsageFilters {
  start_date?: string;
  // Upper bound (exclusive). Omitted for a live "up to now" window; set by the
  // analytics previous-period query so its window does not overlap the current one.
  end_date?: string;
  status?: string;
  model?: string;
  endpoint?: string;
  user_id?: string;
}

// Total matching rows for a set of filters (from /v1/usage/count). Kept separate
// from the list so /v1/usage stays a bare array for external export consumers.
export interface UsageCount {
  total: number;
}

// Time-series granularity for the analytics summary.
export type UsageBucket = "hour" | "day";

// Grand totals over the summary window (from /v1/usage/summary).
export interface UsageTotals {
  cost: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  request_count: number;
  error_count: number;
  // Mean server-side latency over rows that recorded one; null when none did.
  avg_latency_ms: number | null;
}

// One breakdown row (a model, a user, or an API key). `key` is null both for the
// synthesized fold row (`is_other: true`) and for usage whose grouping column was
// NULL, e.g. a since-deleted user (`is_other: false`); `is_other` tells them apart.
export interface UsageGroupRow {
  key: string | null;
  cost: number;
  tokens: number;
  requests: number;
  is_other: boolean;
}

// One time bucket. `bucket_start` is canonical ISO-8601 UTC (`...Z`).
export interface UsageSeriesPoint {
  bucket_start: string;
  cost: number;
  tokens: number;
  requests: number;
}

// Aggregated spend/volume for the Usage & analytics page. `start_date`/`end_date`
// echo the (clamped) window the server actually aggregated over.
export interface UsageSummary {
  start_date: string;
  end_date: string;
  bucket: UsageBucket;
  totals: UsageTotals;
  by_model: UsageGroupRow[];
  by_user: UsageGroupRow[];
  by_api_key: UsageGroupRow[];
  series: UsageSeriesPoint[];
}

// One per-user budget reset event (the spend that was cleared and when the next
// reset is due). Surfaced as the budget's reset history.
export interface BudgetResetLog {
  id: number;
  user_id: string | null;
  budget_id: string;
  previous_spend: number;
  reset_at: string;
  next_reset_at: string | null;
}

// A user/customer: the principal keys and budgets attach to, and where the
// per-user model-access default lives. `allowed_models` is the default every one
// of this user's keys inherits (null = unrestricted, [] = deny all, else canonical
// `instance:model` entries). `user_id` is the identifier used by request routing.
export interface User {
  user_id: string;
  alias: string | null;
  spend: number;
  reserved: number;
  budget_id: string | null;
  allowed_models: string[] | null;
  budget_started_at: string | null;
  next_budget_reset_at: string | null;
  blocked: boolean;
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface CreateUserRequest {
  user_id: string;
  alias?: string | null;
  budget_id?: string | null;
  blocked?: boolean;
  allowed_models?: string[] | null;
  metadata?: Record<string, unknown>;
}

// Omitted fields are left unchanged. `allowed_models` is tri-state on the wire
// (omit = unchanged, null = clear to unrestricted, [] = deny all, list = restrict).
export interface UpdateUserRequest {
  alias?: string | null;
  budget_id?: string | null;
  blocked?: boolean | null;
  allowed_models?: string[] | null;
  metadata?: Record<string, unknown> | null;
}

export type ConfigFieldType = "bool" | "int" | "float" | "str" | "list";

// One effective config value in the full config viewer. `settable` fields can be
// changed at runtime (they hot-apply); the rest are startup-only, display only.
export interface ConfigField {
  key: string;
  value: boolean | number | string | string[] | null;
  type: ConfigFieldType;
  settable: boolean;
  group: string;
  description?: string | null;
  options?: string[] | null;
  // Numeric lower bounds (settable numeric fields only), so the input can gate
  // the value the same way the backend validator does.
  minimum?: number | null; // inclusive (ge)
  exclusive_minimum?: number | null; // gt
}

export interface GatewaySettings {
  mode: string;
  version: string;
  model_discovery: boolean;
  default_pricing: boolean;
  require_pricing: boolean;
  config: ConfigField[];
}

export type StreamMissingUsagePolicy = "estimate" | "fail" | "allow_free";
export type VisionStrategy = "describe" | "ocr" | "off";

// Change one or more runtime settings. Omitted fields are left unchanged. Only
// the hot-changeable subset is accepted; startup-only fields are display-only.
// vision_describe_model is nullable: send null to clear it.
export interface UpdateSettingsRequest {
  model_discovery?: boolean;
  default_pricing?: boolean;
  require_pricing?: boolean;
  reject_user_mismatch?: boolean;
  models_dev_metadata?: boolean;
  file_understanding_enabled?: boolean;
  model_cache_ttl_seconds?: number;
  models_dev_cache_ttl_seconds?: number;
  vision_describe_max_tokens?: number;
  budget_estimate_default_output_tokens?: number;
  model_discovery_timeout_seconds?: number;
  model_discovery_negative_ttl_seconds?: number;
  stream_missing_usage_policy?: StreamMissingUsagePolicy;
  vision_strategy?: VisionStrategy;
  vision_describe_model?: string | null;
}

// Built-in tool & guardrail configuration (the service URLs + web-search knobs
// the Settings page keeps display-only). Editable here, standalone-only.
export type ToolServiceName = "web_search" | "sandbox" | "guardrails";
export type ToolSettingType = "url" | "str" | "int" | "bool";

// One editable tool/guardrail field. `value` is the effective value a request
// would use (URL passwords are masked in the response).
export interface ToolSettingField {
  key: string;
  service: ToolServiceName;
  type: ToolSettingType;
  value: boolean | number | string | null;
  description?: string | null;
}

export interface ToolSettingsResponse {
  fields: ToolSettingField[];
}

// Change one or more tool settings. Omitted fields are unchanged; an explicit
// null clears a field back to the configured env/YAML default.
export interface UpdateToolSettingsRequest {
  web_search_url?: string | null;
  web_search_engines?: string | null;
  web_search_max_results?: number | null;
  web_search_extract?: boolean | null;
  web_search_purpose_hint?: string | null;
  sandbox_url?: string | null;
  sandbox_purpose_hint?: string | null;
  guardrails_url?: string | null;
}

export interface TestServiceResponse {
  ok: boolean;
  reason: string;
}
