import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { apiFetch } from "@/api/client";
import type {
  GatewaySettings,
  HealthResponse,
  ModelListResponse,
  PricingResponse,
  SetPricingRequest,
  UsageEntry,
} from "@/api/types";

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
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: [USAGE] }),
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
