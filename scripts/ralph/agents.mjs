import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";

const efforts = {
  codex: ["minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
  claude: ["low", "medium", "high", "xhigh", "max"],
};

// Existing PRD model fields belong to Codex. Switching tools never forwards them to Claude.
export function selectAgent(config, { tool = "codex", model, reasoningEffort } = {}) {
  if (!Object.hasOwn(efforts, tool)) throw new Error("--tool deve ser codex ou claude.");
  const selectedModel = model ?? (tool === "codex" ? config.model : undefined);
  const effort = reasoningEffort ?? (tool === "codex" ? config.reasoningEffort : undefined);
  if (selectedModel !== undefined && (typeof selectedModel !== "string" || !selectedModel.trim())) {
    throw new Error("--model não pode ser vazio.");
  }
  if (effort !== undefined && !efforts[tool].includes(effort)) {
    throw new Error(`Esforço inválido para ${tool}: ${effort}. Use ${efforts[tool].join(", ")}.`);
  }
  return { tool, model: selectedModel, reasoningEffort: effort };
}

export function codexArgs(root, model, reasoningEffort) {
  return ["exec", "--dangerously-bypass-approvals-and-sandbox", "--ephemeral", "--color", "never",
    "--cd", root, ...(model ? ["--model", model] : []),
    ...(reasoningEffort ? ["--config", `model_reasoning_effort=${JSON.stringify(reasoningEffort)}`] : []), "-"];
}

export function agentCommand(root, agent) {
  if (agent.tool === "codex") return ["codex", ...codexArgs(root, agent.model, agent.reasoningEffort)];
  return ["claude", "--print", "--dangerously-skip-permissions", "--no-session-persistence",
    "--output-format", "stream-json", "--verbose",
    ...(agent.model ? ["--model", agent.model] : []),
    ...(agent.reasoningEffort ? ["--effort", agent.reasoningEffort] : [])];
}

// Inspect only the chosen CLI. Never print auth output (it may contain account details).
export function checkAgent(root, agent, { execute = spawnSync, exists = existsSync, home = process.env.HOME ?? "/home/node" } = {}) {
  const run = (args) => execute(agent.tool, args, { cwd: root, encoding: "utf8", timeout: 15_000 });
  if (run(["--version"]).status !== 0) throw new Error(`${agent.tool} CLI indisponível; instale no devcontainer.`);
  if (agent.tool === "codex") {
    if (!exists(resolve(home, ".codex/auth.json"))) throw new Error("credencial Codex ausente em ~/.codex/auth.json");
    return;
  }
  const help = run(["--help"]);
  const required = ["--print", "--dangerously-skip-permissions", "--no-session-persistence", "--output-format", "--verbose",
    ...(agent.reasoningEffort ? ["--effort"] : [])];
  if (help.status !== 0 || required.some((flag) => !help.stdout?.includes(flag))) {
    throw new Error("Claude Code incompatível com o Ralph; atualize a CLI no devcontainer.");
  }
  if (run(["auth", "status"]).status !== 0) {
    throw new Error("Claude Code sem autenticação disponível; execute claude auth login dentro do devcontainer e repita o preflight.");
  }
}
