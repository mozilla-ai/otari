import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "@/api/client";
import type {
  CreateKeyRequest,
  CreateKeyResponse,
  CreateUserRequest,
  GatewaySettings,
  HealthResponse,
  KeyInfo,
  ModelListResponse,
  PricingResponse,
  SetPricingRequest,
  UsageEntry,
  UserResponse,
} from "@/api/types";

const KEYS = "keys";
const USERS = "users";
const USAGE = "usage";
const HEALTH = "health";
const MODELS = "models";
const PRICING = "pricing";
const SETTINGS = "settings";

export function useHealth() {
  return useQuery({
    queryKey: [HEALTH],
    queryFn: () => apiFetch<HealthResponse>("/health"),
    staleTime: 60_000,
  });
}

export function useKeys() {
  return useQuery({
    queryKey: [KEYS],
    queryFn: () => apiFetch<KeyInfo[]>("/v1/keys?limit=1000"),
  });
}

export function useCreateKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateKeyRequest) =>
      apiFetch<CreateKeyResponse>("/v1/keys", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [KEYS] });
      void queryClient.invalidateQueries({ queryKey: [USERS] });
    },
  });
}

export function useSetKeyActive() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, isActive }: { id: string; isActive: boolean }) =>
      apiFetch<KeyInfo>(`/v1/keys/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify({ is_active: isActive }),
      }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  });
}

export function useDeleteKey() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/v1/keys/${encodeURIComponent(id)}`, { method: "DELETE" }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [KEYS] }),
  });
}

export function useUsers() {
  return useQuery({
    queryKey: [USERS],
    queryFn: () => apiFetch<UserResponse[]>("/v1/users?limit=1000"),
  });
}

export function useCreateUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateUserRequest) =>
      apiFetch<UserResponse>("/v1/users", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [USERS] }),
  });
}

export function useDeleteUser() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (userId: string) =>
      apiFetch<void>(`/v1/users/${encodeURIComponent(userId)}`, { method: "DELETE" }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [USERS] });
      void queryClient.invalidateQueries({ queryKey: [KEYS] });
    },
  });
}

export function useUsage(limit = 500) {
  return useQuery({
    queryKey: [USAGE, limit],
    queryFn: () => apiFetch<UsageEntry[]>(`/v1/usage?limit=${limit}`),
  });
}

export interface BackfillResult {
  model_key: string;
  rows_updated: number;
  cost_added: number;
  users_updated: number;
}

export function useBackfillUsageCost() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (modelKey: string) =>
      apiFetch<BackfillResult>("/v1/usage/backfill", {
        method: "POST",
        body: JSON.stringify({ model_key: modelKey }),
      }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [USAGE] });
      void queryClient.invalidateQueries({ queryKey: [USERS] });
    },
  });
}

export function useModels() {
  return useQuery({
    queryKey: [MODELS],
    queryFn: () => apiFetch<ModelListResponse>("/v1/models"),
    staleTime: 60_000,
  });
}

export function useSettings() {
  return useQuery({
    queryKey: [SETTINGS],
    queryFn: () => apiFetch<GatewaySettings>("/v1/settings"),
    staleTime: 60_000,
  });
}

export function usePricing() {
  return useQuery({
    queryKey: [PRICING],
    queryFn: () => apiFetch<PricingResponse[]>("/v1/pricing?limit=1000"),
  });
}

export function useSetPricing() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: SetPricingRequest) =>
      apiFetch<PricingResponse>("/v1/pricing", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PRICING] });
      void queryClient.invalidateQueries({ queryKey: [MODELS] });
    },
  });
}

export function useDeletePricing() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (modelKey: string) =>
      apiFetch<void>(`/v1/pricing/${encodeURIComponent(modelKey)}`, { method: "DELETE" }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [PRICING] });
      void queryClient.invalidateQueries({ queryKey: [MODELS] });
    },
  });
}
