import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmdirSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { spawn, spawnSync } from "node:child_process";
import { parseArgs } from "node:util";
import { z } from "zod";
import { agentCommand, checkAgent, selectAgent } from "./agents.mjs";
export { codexArgs } from "./agents.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const text = z.string().trim().min(1);
const command = z.array(text).min(1);
const reasoningEffortSchema = z.enum(["minimal", "low", "medium", "high", "xhigh", "max", "ultra"]);
const designSchema = z.object({ id: text, prd: text, buildReport: text, dependsOn: z.array(text) }).strict();
const storySchema = z.object({
  id: text, title: text, designIds: z.array(text).min(1), scope: text,
  checks: z.array(command), passes: z.boolean(), finalReview: z.boolean().default(false),
}).strict();
const configurationSchema = z.object({
  version: z.literal(2), branchName: text, progressPath: text,
  model: text.optional(), reasoningEffort: reasoningEffortSchema.optional(),
  designs: z.array(designSchema).min(1), userStories: z.array(storySchema).min(2),
}).strict();

function git(root, args) {
  const result = spawnSync("git", args, { cwd: root, encoding: "utf8" });
  if (result.status !== 0) throw new Error(`Git falhou: ${args.join(" ")}\n${result.stderr.trim()}`);
  return result.stdout.trim();
}

// Resolve também os ancestrais existentes de relatórios ainda não criados.
export function localPath(root, value) {
  root = realpathSync(root);
  if (isAbsolute(value) || value.split(/[\\/]/).includes("..")) throw new Error(`Caminho fora do repositório: ${value}`);
  const path = resolve(root, value);
  let ancestor = path;
  while (!existsSync(ancestor)) ancestor = dirname(ancestor);
  const actual = relative(root, realpathSync(ancestor));
  if (actual === ".." || actual.startsWith(`..${sep}`) || isAbsolute(actual)) throw new Error(`Symlink fora do repositório: ${value}`);
  return path;
}

function stablePlan(config) {
  return JSON.stringify({ ...config, userStories: config.userStories.map((story) => {
    const stableStory = { ...story };
    delete stableStory.passes;
    return stableStory;
  }) });
}

function committedConfiguration(root, configPath) {
  const relativeConfigPath = relative(root, configPath);
  if (relativeConfigPath === "" || relativeConfigPath === ".." || relativeConfigPath.startsWith(`..${sep}`)) {
    throw new Error("O PRD do Ralph precisa estar dentro da raiz Git.");
  }
  return configurationSchema.parse(JSON.parse(git(root, ["show", `HEAD:${relativeConfigPath}`])));
}

// O commit, e não uma escrita interrompida no working tree, é a fonte de verdade de passes.
export function reconcilePassesWithHead(root, configPath) {
  const working = loadPlan(root, configPath);
  const committed = committedConfiguration(root, configPath);
  if (working.stable !== stablePlan(committed)) {
    throw new Error("O PRD possui alterações estruturais não commitadas; durante o loop apenas passes pode mudar.");
  }

  const committedPasses = new Map(committed.userStories.map((story) => [story.id, story.passes]));
  let changed = false;
  for (const story of working.config.userStories) {
    const committedValue = committedPasses.get(story.id);
    if (story.passes !== committedValue) {
      story.passes = committedValue;
      changed = true;
    }
  }
  if (changed) writeFileSync(configPath, `${JSON.stringify(working.config, null, 2)}\n`, "utf8");
  return loadPlan(root, configPath);
}

function commitTransition(root, headBefore) {
  const headAfter = git(root, ["rev-parse", "HEAD"]);
  if (headAfter === headBefore) return { headAfter, count: 0, direct: false };
  const count = Number(git(root, ["rev-list", "--count", `${headBefore}..${headAfter}`]));
  const parents = git(root, ["show", "-s", "--format=%P", headAfter]).split(/\s+/u).filter(Boolean);
  return { headAfter, count, direct: count === 1 && parents.length === 1 && parents[0] === headBefore };
}

export function loadPlan(root, configPath) {
  const config = configurationSchema.parse(JSON.parse(readFileSync(configPath, "utf8")));
  const designIds = new Set();
  for (const design of config.designs) {
    if (designIds.has(design.id)) throw new Error(`Design duplicado: ${design.id}`);
    for (const dependency of design.dependsOn) {
      if (!designIds.has(dependency)) throw new Error(`Dependência deve preceder ${design.id}: ${dependency}`);
    }
    designIds.add(design.id);
    if (!design.prd.endsWith(".md")) throw new Error("prd deve referenciar o DESIGN Markdown canônico.");
    readFileSync(localPath(root, design.prd), "utf8");
    localPath(root, design.buildReport);
  }
  localPath(root, config.progressPath);

  const storyIds = new Set();
  const reviews = config.userStories.filter((story) => story.finalReview);
  if (reviews.length !== 1 || !config.userStories.at(-1).finalReview) {
    throw new Error("A única história finalReview deve ser a última da fila.");
  }
  for (const story of config.userStories) {
    if (storyIds.has(story.id)) throw new Error(`História duplicada: ${story.id}`);
    storyIds.add(story.id);
    if (story.designIds.some((id) => !designIds.has(id))) throw new Error(`História ${story.id} referencia design desconhecido.`);
  }
  if (config.designs.some((design) => !config.userStories.some((story) => !story.finalReview && story.designIds.includes(design.id)))) {
    throw new Error("Cada design precisa de ao menos uma história de implementação.");
  }
  if (reviews[0].designIds.length !== config.designs.length || config.designs.some((design) => !reviews[0].designIds.includes(design.id))) {
    throw new Error("A revisão final deve referenciar todos os designs.");
  }
  return { config, stable: stablePlan(config) };
}

export function isDevContainer(dockerenv = existsSync("/.dockerenv"), containerenv = existsSync("/run/.containerenv")) {
  return dockerenv || containerenv;
}

// Grupo próprio permite encerrar também subprocessos no timeout/Ctrl+C (macOS/Linux).
export function runCommand(argv, { root, input = "", timeoutMs, log }) {
  return new Promise((resolveResult, reject) => {
    const child = spawn(argv[0], argv.slice(1), { cwd: root, shell: false, detached: process.platform !== "win32", stdio: ["pipe", "pipe", "pipe"] });
    let timedOut = false;
    let interrupted = false;
    const kill = () => {
      if (!child.pid) return;
      try {
        if (process.platform === "win32") child.kill("SIGKILL");
        else process.kill(-child.pid, "SIGKILL");
      } catch (error) { if (error.code !== "ESRCH") throw error; }
    };
    const interrupt = () => { interrupted = true; kill(); };
    process.once("SIGINT", interrupt);
    process.once("SIGTERM", interrupt);
    const timer = setTimeout(() => { timedOut = true; kill(); }, timeoutMs);
    const cleanup = () => {
      clearTimeout(timer);
      process.removeListener("SIGINT", interrupt);
      process.removeListener("SIGTERM", interrupt);
    };
    const output = (data) => appendFileSync(log, data, { mode: 0o600 });
    child.stdout.on("data", output);
    child.stderr.on("data", output);
    child.stdin.on("error", (error) => { if (error.code !== "EPIPE") kill(); });
    child.once("error", (error) => { cleanup(); reject(error); });
    child.once("close", (code, signal) => { cleanup(); resolveResult({ code, signal, timedOut, interrupted }); });
    child.stdin.end(input);
  });
}

export async function executeLoop({ root, configPath, dryRun = false, tool = "claude", model, reasoningEffort, maxIterations = 30,
  timeoutMs = 1_800_000, run = runCommand, print = console.log, containerized = isDevContainer(), ensureAgent = checkAgent }) {
  root = realpathSync(root);
  let initial = loadPlan(root, configPath);
  const agent = selectAgent(initial.config, { tool, model, reasoningEffort });
  if (![maxIterations, timeoutMs].every((n) => Number.isSafeInteger(n) && n > 0)) throw new Error("Limites devem ser inteiros positivos.");
  print(`Ferramenta: ${agent.tool}`);
  print(`Modelo: ${agent.model ?? "padrão da CLI"}${agent.reasoningEffort ? ` (${agent.reasoningEffort})` : ""}`);
  for (const design of initial.config.designs) print(`${design.id}: ${design.prd}`);
  for (const story of initial.config.userStories) print(`${story.passes ? "✓" : "○"} ${story.id}: ${story.title}`);
  if (dryRun) return { status: "dry-run", code: 0 };
  if (!containerized) throw new Error("Autonomia total recusada fora de um container real. Abra o projeto pelo Dev Container.");
  if (git(root, ["branch", "--show-current"]) !== initial.config.branchName) throw new Error(`Use a branch ${initial.config.branchName}; o loop não troca branches.`);
  await ensureAgent(root, agent);
  initial = reconcilePassesWithHead(root, configPath);

  const progressPath = localPath(root, initial.config.progressPath);
  if (!existsSync(progressPath)) writeFileSync(progressPath, "# Ralph Progress Log\n\n", { mode: 0o600 });
  const runtime = localPath(root, ".ralph");
  mkdirSync(runtime, { recursive: true, mode: 0o700 });
  const lockPath = resolve(runtime, "loop.lock");
  try { mkdirSync(lockPath); } catch { throw new Error("Há outro loop ativo ou um lock residual em .ralph/loop.lock."); }
  let runDir;
  try {
    runDir = mkdtempSync(resolve(runtime, "run-"));
    for (let iteration = 1; iteration <= maxIterations; iteration++) {
      const before = reconcilePassesWithHead(root, configPath);
      if (before.stable !== initial.stable) return { status: "blocked", code: 2, runDir, reason: "Estrutura do PRD mudou durante o loop." };
      const story = before.config.userStories.find((item) => !item.passes);
      if (!story) return { status: "complete", code: 0, runDir };
      const storyIndex = before.config.userStories.findIndex((item) => item.id === story.id);
      if (story.finalReview && before.config.userStories.slice(0, storyIndex).some((item) => !item.passes)) {
        return { status: "blocked", code: 2, runDir, reason: "A revisão final não pode antecipar histórias pendentes." };
      }

      const headBefore = git(root, ["rev-parse", "HEAD"]);
      const progressSize = readFileSync(progressPath).length;
      const log = resolve(runDir, `${iteration}-${story.id}.log`);
      const outcomePath = resolve(runDir, `${iteration}-${story.id}.outcome.json`);
      // Só o que a história usa: a revisão final já lista todos os designs em designIds.
      const prompt = `${readFileSync(resolve(here, "PROMPT.md"), "utf8")}\n\n## Current run\n${JSON.stringify({
        iteration, initialHead: headBefore, agent, outcomePath: relative(root, outcomePath), prdPath: relative(root, configPath), progressPath: before.config.progressPath, story,
        designs: before.config.designs.filter((design) => story.designIds.includes(design.id)),
      }, null, 2)}\n`;
      writeFileSync(resolve(runDir, `${iteration}-${story.id}.prompt.md`), prompt, { mode: 0o600 });
      print(`Iteração ${iteration}/${maxIterations}: ${story.id} — ${story.title}`);
      const execution = await run(agentCommand(root, agent), { root, input: prompt, timeoutMs, log });
      if (git(root, ["branch", "--show-current"]) !== initial.config.branchName) return { status: "blocked", code: 2, runDir, reason: "O agente trocou a branch." };

      const after = loadPlan(root, configPath);
      if (after.stable !== initial.stable) return { status: "blocked", code: 2, runDir, reason: "O agente alterou a estrutura, os critérios ou as referências do PRD." };
      const completed = after.config.userStories.find((item) => item.id === story.id)?.passes;
      const otherPassChanged = after.config.userStories.some((item, index) => index !== storyIndex && before.config.userStories[index].passes !== item.passes);
      if (otherPassChanged) return { status: "blocked", code: 2, runDir, reason: "A iteração alterou passes de outra história." };
      const transition = commitTransition(root, headBefore);

      if (execution.code !== 0 && !execution.timedOut && !execution.interrupted && completed) {
        return { status: "blocked", code: 2, runDir, reason: `${story.id} marcou passes apesar da falha do processo.` };
      }
      if (!completed) {
        if (transition.headAfter !== headBefore) return { status: "blocked", code: 2, runDir, reason: `${story.id} criou commit antes de concluir a história.` };
        reconcilePassesWithHead(root, configPath);
        if (execution.interrupted) return { status: "interrupted", code: 130, runDir };
        if (execution.timedOut) return { status: "timeout", code: 1, runDir };
        if (existsSync(outcomePath)) {
          let outcome;
          try {
            outcome = z.object({ status: z.enum(["BLOCKER", "CONTEXT", "TIME"]), reason: z.string().trim().min(1) })
              .strict().parse(JSON.parse(readFileSync(outcomePath, "utf8")));
          } catch {
            return { status: "blocked", code: 2, runDir, reason: `${story.id} deixou resultado de iteração inválido em ${outcomePath}.` };
          }
          if (outcome.status === "BLOCKER") {
            return { status: "blocked", code: 2, runDir, reason: `${story.id}: BLOCKER — ${outcome.reason}` };
          }
        } else if (execution.code === 0) {
          return { status: "blocked", code: 2, runDir, reason: `${story.id} terminou pendente sem resultado em ${outcomePath}.` };
        }
        print(`${story.id} continua pendente; a próxima instância retomará o working tree e o progress.txt.`);
        continue;
      }

      if (!transition.direct || transition.count !== 1) {
        reconcilePassesWithHead(root, configPath);
        if (execution.interrupted && transition.count === 0) return { status: "interrupted", code: 130, runDir };
        if (execution.timedOut && transition.count === 0) return { status: "timeout", code: 1, runDir };
        return {
          status: "blocked", code: 2, runDir,
          reason: `${story.id} exige exatamente um commit direto a partir do HEAD inicial; encontrados ${transition.count}.`,
        };
      }
      if (git(root, ["status", "--porcelain"])) return { status: "blocked", code: 2, runDir, reason: `${story.id} deixou alterações fora do commit.` };
      if (readFileSync(progressPath).length <= progressSize) return { status: "blocked", code: 2, runDir, reason: `${story.id} não acrescentou memória ao progress.txt.` };
      const subject = git(root, ["log", "-1", "--pretty=%s"]);
      if (!subject.includes(story.id)) return { status: "blocked", code: 2, runDir, reason: `O commit da história não menciona ${story.id}.` };
      if (/co-authored-by|generated with/i.test(git(root, ["log", "-1", "--pretty=%B"]))) {
        return { status: "blocked", code: 2, runDir, reason: `O commit de ${story.id} contém atribuição de autoria a IA; refaça sem trailers.` };
      }
      if (execution.timedOut || execution.interrupted) print(`${story.id} reconciliada como concluída pelo commit ${transition.headAfter}.`);
      print(`✓ ${story.id} concluída em ${git(root, ["rev-parse", "--short", "HEAD"])}.`);
    }
    return { status: "iteration-limit", code: 1, runDir };
  } finally {
    rmdirSync(lockPath);
  }
}

async function main() {
  const { values } = parseArgs({ options: {
    config: { type: "string", default: "scripts/ralph/prd.noite1.json" }, "dry-run": { type: "boolean", default: false },
    tool: { type: "string", default: "claude" }, model: { type: "string" }, "reasoning-effort": { type: "string" },
    "max-iterations": { type: "string", default: "30" }, "timeout-minutes": { type: "string", default: "30" },
  } });
  const root = realpathSync(git(process.cwd(), ["rev-parse", "--show-toplevel"]));
  const result = await executeLoop({ root, configPath: localPath(root, values.config), dryRun: values["dry-run"], tool: values.tool, model: values.model,
    reasoningEffort: values["reasoning-effort"],
    maxIterations: Number(values["max-iterations"]), timeoutMs: Number(values["timeout-minutes"]) * 60_000 });
  if (result.reason) console.error(result.reason);
  process.exitCode = result.code;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  main().catch((error) => { console.error(error.message); process.exitCode = 2; });
}
