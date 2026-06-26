// Response shapes mirror the gateway's Pydantic models in
// src/gateway/api/routes/{keys,users,usage,health}.py. Keep them in sync.

export interface KeyInfo {
  id: string;
  key_name: string | null;
  user_id: string | null;
  created_at: string;
  last_used_at: string | null;
  expires_at: string | null;
  is_active: boolean;
  metadata: Record<string, unknown>;
}

export interface CreateKeyResponse extends Omit<KeyInfo, "last_used_at"> {
  key: string;
}

export interface CreateKeyRequest {
  key_name?: string | null;
  user_id?: string | null;
  expires_at?: string | null;
}

export interface UserResponse {
  user_id: string;
  alias: string | null;
  spend: number;
  reserved: number;
  budget_id: string | null;
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
}

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
