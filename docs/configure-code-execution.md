---
icon: code
---

# Configure code execution

Each workspace decides whether its requests can run code in a sandbox. These
steps are in the dashboard; the settings they change are described in
[Built-in tools](tools.md).

A workspace setting can only narrow what the deployment allows: it cannot turn
on code execution when the deployment has no sandbox.

1. Select the workspace in the top-left corner.
2. Go to **Tools**, then **Code execution**.
3. Turn **Allow in <workspace>** on or off.
4. Under **Who runs provider code tools**, choose whether a provider's own code
   tool or Otari's sandbox runs the code.

The other fields of the policy, such as the iteration cap, the execution timeout
and the sandbox image, are set through the API.

The policy these steps set is described in
[Per-workspace code policy](tools.md#per-workspace-code-policy), and who runs the
code in [Code-execution executor](tools.md#code-execution-executor).
