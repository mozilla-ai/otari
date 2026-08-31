# Data fetching: TanStack Query

All *authenticated* server state flows through TanStack Query hooks in `web/src/shared/api/hooks.ts`,
which call `apiFetch` from `web/src/shared/api/client.ts`. Don't call `fetch()` directly for
authenticated management requests, and never mirror server state into `useState`.

## The API boundary

`apiFetch<T>(path, init)` is the single door to the gateway's management API. It:

- lets the browser attach the HttpOnly session cookie on same-origin requests,
- sets JSON headers, extracts `detail` from error bodies into an `ApiError`,
- treats **401** as an expired or revoked session, calls the registered unauthorized handler,
  and throws. A 403 is an authorization refusal for a still-valid session.

The query client (`web/src/app/provider.tsx`) does not retry an `ApiError` with status 401 or
403, because neither will fix itself, and retries other failures twice.

Pre-authentication helpers such as `createSession`, `signInWithPasskey`, and the OAuth helpers
use public requests. They cannot use `apiFetch`, because a 401 while somebody is signing in
must report a refused credential rather than bounce them back to the page they are already on.
Every authenticated management request after sign-in goes through `apiFetch`.

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

## Loading and error guards carry `&& !data`

`isPending` is true again every time the key changes, so a guard on the flag alone blanks a
populated page whenever a filter moves:

```tsx
if (isPending && !data) return <Skeleton />
if (isError && !data) return <ErrorBanner error={error} />
return <Table rows={data} />
```

`isLoading` (`isPending && isFetching`) is worse: it is false for a disabled or
cache-restoring query, so it flashes an empty state at exactly the wrong moment. And for a
query whose key carries filters or pagination, add `placeholderData: (previous) => previous`
so the last result stays on screen through the refetch. [layout-stability.md](./layout-stability.md)
has the rest of the flicker rules.

## Independent requests run together

Two awaits that do not depend on each other should not be sequential:

```ts
const [models, pricing] = await Promise.all([fetchModels(), fetchPricing()])
```

Inside a component this is usually not a question, because two `useQuery` calls already run in
parallel. It comes up in a `queryFn` that assembles from more than one endpoint, and in the
bounded walks below.

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

### An error has to go somewhere

A mutation's failure is the operator's business. Either let it surface where the action was
taken (the call-site `mutate(vars, { onError })`, which is what the pages here do, rendering
it through `ErrorBanner` and `errorMessage(error)`), or handle it in the hook. What is not
acceptable is an `onError` that swallows: a delete that quietly did nothing is worse than one
that says why it could not.

401 sign-out and the query client's no-retry handling for 401 and 403 are centralized.

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
