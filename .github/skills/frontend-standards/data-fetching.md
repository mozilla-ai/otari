# Data fetching: TanStack Query

All server state flows through TanStack Query hooks in `web/src/api/hooks.ts`, which call
`apiFetch` from `web/src/api/client.ts`. Never call `fetch()` for the management API directly
and never mirror server state into `useState`.

## The API boundary

`apiFetch<T>(path, init)` is the single door to the gateway's management API. It:

- attaches the master key as `Authorization: Bearer …`,
- sets JSON headers, extracts `detail` from error bodies into an `ApiError`,
- treats **401 and 403** as "this session can't use the management API", it calls the
  registered unauthorized handler (drops the key, bounces to sign-in) and throws.

Because 401/403 are handled centrally, hooks don't need to. The query client
(`web/src/provider.tsx`) also **never retries** an `ApiError` with status 401/403 (they won't
fix themselves) and retries other failures twice.

## Query conventions

- **Query keys are module constants**, not inline literals:

  ```ts
  const MODELS = "models";
  const PRICING = "pricing";
  // ...
  export function useModels() {
    return useQuery({ queryKey: [MODELS], queryFn: () => apiFetch<ModelListResponse>("/v1/models"), staleTime: 60_000 });
  }
  ```

- **Set `staleTime` deliberately**, sized to how fast the data actually moves: seconds for
  mutable lists (`aliases`, `settings`, `models`), several minutes for near-static gateway
  metadata (`providers`, discovery, model metadata). Add a one-line comment when the choice is
  non-obvious, as the existing hooks do.

- **Keep independent keys independent.** `discoverable` is deliberately *not* nested under
  `models` because a pricing change can't alter which models a provider serves, sharing the
  key would fire a live provider call on every save. Think about invalidation blast radius when
  you pick a key.

## Mutations invalidate what they change

Every mutation invalidates exactly the keys its write affects, no more, no less:

```ts
export function useCreateAlias() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateAliasRequest) =>
      apiFetch<AliasResponse>("/v1/aliases", { method: "POST", body: JSON.stringify(body) }),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [ALIASES] });
      void queryClient.invalidateQueries({ queryKey: [MODELS] }); // an alias shows up as a model
    },
  });
}
```

- Invalidate the primary key **and** any derived view the write touches (creating an alias
  changes the model catalog too).
- Use `setQueryData` when the mutation already returns the fresh object (`useUpdateSettings`
  seeds `[SETTINGS]` from the response, then invalidates the derived model lists).
- Prefix fire-and-forget invalidations with `void` so the floating-promise lint stays happy.

## Bounded pagination

When a hook fetches "everything," cap the walk so a backend or proxy that ignores `skip`
can't turn it into an unbounded request loop. Copy the `fetchAllPricing` shape:

```ts
const PRICING_PAGE_SIZE = 1000;   // matches the server-side cap
const PRICING_MAX_PAGES = 100;    // hard stop: 100k rows, far beyond any real history

async function fetchAllPricing(): Promise<PricingResponse[]> {
  const all: PricingResponse[] = [];
  for (let page = 0; page < PRICING_MAX_PAGES; page += 1) {
    const rows = await apiFetch<PricingResponse[]>(`/v1/pricing?skip=${page * PRICING_PAGE_SIZE}&limit=${PRICING_PAGE_SIZE}`);
    all.push(...rows);
    if (rows.length < PRICING_PAGE_SIZE) break;
  }
  return all;
}
```

## Polling

Use `refetchInterval` for anything that must stay live, not a hand-rolled `setInterval`.
`useDashboardBuild` polls a small build hash every 60s (and on window focus, with `retry:
false`) so an open tab notices a redeploy, model that when you need liveness.
