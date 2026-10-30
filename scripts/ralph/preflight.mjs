// Confere o ambiente antes do loop: container, Python e checks, branch, árvore limpa, dados e a CLI do agente.
import { existsSync, readdirSync, realpathSync } from "node:fs";
import { resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { parseArgs } from "node:util";
import { checkAgent, selectAgent } from "./agents.mjs";
import { isDevContainer, loadPlan, localPath } from "./loop.mjs";

function run(root, executable, args) {
  const result = spawnSync(executable, args, { cwd: root, encoding: "utf8", timeout: 120_000 });
  return { ok: result.status === 0, out: `${result.stdout ?? ""}${result.stderr ?? ""}`.trim() };
}

const root = realpathSync(run(process.cwd(), "git", ["rev-parse", "--show-toplevel"]).out);
const { values } = parseArgs({ options: {
  config: { type: "string", default: "scripts/ralph/prd.noite1.json" }, tool: { type: "string", default: "claude" },
  model: { type: "string" }, "reasoning-effort": { type: "string" },
} });
const { config } = loadPlan(root, localPath(root, values.config));
const agent = selectAgent(config, { tool: values.tool, model: values.model, reasoningEffort: values["reasoning-effort"] });
console.log(`Ferramenta: ${agent.tool}; modelo: ${agent.model ?? "padrão da CLI"}.`);

const failures = [];
if (!isDevContainer()) failures.push("ambiente: execute dentro do container (scripts/ralph/container.sh)");

const branch = run(root, "git", ["branch", "--show-current"]).out;
if (branch !== config.branchName) failures.push(`branch: esperado ${config.branchName}, encontrado ${branch}`);
if (run(root, "git", ["status", "--porcelain"]).out) failures.push("árvore Git suja: faça commit ou descarte antes do loop");
if (!run(root, "git", ["config", "user.email"]).out) failures.push("git user.email não configurado no repositório");

for (const [description, args] of [
  ["pytest", ["-m", "pytest", "-q"]],
  ["ruff", ["-m", "ruff", "check", "src", "tests", "scripts"]],
  ["notebooks", ["scripts/check_notebooks.py"]],
]) {
  const result = run(root, "python", args);
  if (!result.ok) failures.push(`${description} já falha antes do loop:\n${result.out.split("\n").slice(-15).join("\n")}`);
  else console.log(`✓ ${description}`);
}

const raw = resolve(root, "data/raw");
for (const file of ["RePro.csv", "olist_order_reviews_dataset.csv", "olist_orders_dataset.csv"]) {
  if (!existsSync(resolve(raw, file))) failures.push(`dados ausentes: data/raw/${file}`);
}
const secrets = existsSync(root) ? readdirSync(root).filter((name) => /^\.env/.test(name)) : [];
if (secrets.length) failures.push(`remova arquivos de segredo da raiz: ${secrets.join(", ")}`);

try {
  checkAgent(root, agent);
  console.log(`✓ ${agent.tool} CLI autenticada`);
} catch (error) {
  failures.push(error.message);
}

if (failures.length) {
  console.error(`\nPreflight reprovado:\n- ${failures.join("\n- ")}`);
  process.exitCode = 1;
} else {
  console.log("\nPreflight aprovado.");
}
