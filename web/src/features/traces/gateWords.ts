// How a guardrail gate's vocabulary reads to a person, shared by every place that shows a gate.

export const GATE_KIND: Record<string, string> = {
  path: "Forbids files by path",
  command: "Forbids shell commands",
  command_if_changed: "Requires a command when matching files change",
  judge: "A model reviews the change against a rubric",
  verifier: "A script decides",
}

export const GATE_MOMENT: Record<string, string> = {
  "pre_tool_use.edit_target": "Before the agent edits a file",
  "pre_tool_use.read_target": "Before the agent reads a file",
  "pre_tool_use.command": "Before the agent runs a shell command",
  "stop.working_tree": "When the agent stops, on the files it changed",
  "stop.session": "When the agent stops, on the commands it ran",
  stop: "When the agent stops",
}

export function says(table: Record<string, string>, key: string): string {
  return table[key] ?? key
}
