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
