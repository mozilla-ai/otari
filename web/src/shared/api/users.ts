import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import type { CreateUserRequest, UpdateUserRequest, User } from "@/client"
import { apiFetch } from "@/shared/api/client"
import { BUDGETS, KEYS, USERS } from "@/shared/api/queryKeys"

const USERS_PAGE_SIZE = 1000
const USERS_MAX_PAGES = 100

async function fetchAllUsers(): Promise<User[]> {
  const all: User[] = []
  for (let page = 0; page < USERS_MAX_PAGES; page += 1) {
    const rows = await apiFetch<User[]>(
      `/users?skip=${page * USERS_PAGE_SIZE}&limit=${USERS_PAGE_SIZE}`,
    )
    all.push(...rows)
    if (rows.length < USERS_PAGE_SIZE) {
      break
    }
  }
  return all
}

// Gated for the same reason as `useBudgets` above.
export function useUsers(enabled = true) {
  return useQuery({
    queryKey: [USERS],
    queryFn: fetchAllUsers,
    staleTime: 60_000,
    enabled,
  })
}

// Assigning a budget to a user changes that budget's usage rollup, so a user
// write invalidates the budgets list too.
function invalidateUserViews(
  queryClient: ReturnType<typeof useQueryClient>,
): void {
  void queryClient.invalidateQueries({ queryKey: [USERS] })
  void queryClient.invalidateQueries({ queryKey: [BUDGETS] })
}

export function useCreateUser() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (body: CreateUserRequest) =>
      apiFetch<User>("/users", {
        method: "POST",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateUserViews(queryClient),
  })
}

export function useUpdateUser() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({ id, body }: { id: string; body: UpdateUserRequest }) =>
      apiFetch<User>(`/users/${encodeURIComponent(id)}`, {
        method: "PATCH",
        body: JSON.stringify(body),
      }),
    onSuccess: () => invalidateUserViews(queryClient),
  })
}

export function useDeleteUser() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) =>
      apiFetch<void>(`/users/${encodeURIComponent(id)}`, {
        method: "DELETE",
      }),
    onSuccess: () => {
      invalidateUserViews(queryClient)
      // Deleting a user deactivates its keys server-side.
      void queryClient.invalidateQueries({ queryKey: [KEYS] })
    },
  })
}
