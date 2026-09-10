import type { Decorator } from "@storybook/react-vite"

import type { DeploymentBootstrap } from "@/client"
import { DeploymentProvider } from "@/shared/hooks/useDeployment"
import { SelectedWorkspaceProvider } from "@/shared/hooks/SelectedWorkspace"
import { bootstrap } from "@/tests/fixtures"

/**
 * The two app-level contexts a feature component may read.
 *
 * Neither is a fetch, so `apiMock` cannot stand in for them: `useDeployment`
 * throws outside its provider, and `useSelectedWorkspace` answers null. A story
 * of `SetupGuideCard`, `PasswordCard` or the code-execution policy card needs
 * them mounted, and mounting them per story would mean every such story
 * restating the same wrapper.
 *
 * `.storybook/` may import `@/shared/hooks` and `@/tests` freely: it sits outside
 * `src/`, so none of biome.jsonc's layer overrides apply to it. This is the same
 * exemption `src/tests/providers.tsx` documents for itself -- a harness exists to
 * mount what the app mounts.
 *
 * The default bootstrap is the fixture's, so it describes a standalone deployment
 * with every standalone surface hosted. A story overrides it through
 * `parameters.deployment`, which is how the hybrid and claimed-key cases are
 * reached:
 *
 *   parameters: { deployment: { sign_in_methods: ["password"] } }
 */
export const withAppContext: Decorator = (Story, context) => {
  const overrides = (context.parameters.deployment ?? {}) as Partial<DeploymentBootstrap>

  return (
    <DeploymentProvider value={bootstrap(overrides)}>
      {/* Inside the deployment, and inside the query client `apiMock` provides:
          it seeds itself from `useOrganizationContext()`, so a story that cares
          which workspace is selected mocks /v1/organization/context. */}
      <SelectedWorkspaceProvider>
        <Story />
      </SelectedWorkspaceProvider>
    </DeploymentProvider>
  )
}
