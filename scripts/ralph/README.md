# Ralph loop — PLN ABSA Olist

Adaptação do loop do CRM Marrocos (por sua vez baseado em [snarktank/ralph](https://github.com/snarktank/ralph)):
um agente novo por história, fila e `passes` no `prd.<fila>.json`, memória em Git + `progress.<fila>.txt`, e o
próprio agente rodando os checks e fazendo **um commit por história**.

| Arquivo | Papel |
|---|---|
| `loop.mjs` | Controlador: pega a 1ª história com `passes: false`, chama o Claude Code, confere as invariantes de Git |
| `agents.mjs` | Montagem do comando da CLI (Claude ou Codex) e checagem de autenticação |
| `preflight.mjs` | Confere container, branch, árvore limpa, checks verdes, dados e login da CLI |
| `PROMPT.md` | Única instrução do agente |
| `prd.noite1.json` | Fila da noite 1 (E1, E2-S1/S2, E3-S1, E4-S1/S3 + revisão final) |
| `progress.noite1.txt` | Memória curta entre agentes |
| `container.sh` | Sobe o container (Python CPU + Node + Claude Code) com o repositório montado |

**Garantias do controlador:** branch certa, uma história por iteração, só o `passes` daquela história muda no
PRD, exatamente um commit direto mencionando o ID, `progress` cresceu, árvore limpa e **nenhum trailer de
autoria de IA** no commit. Ele recusa rodar fora de um container (autonomia total só em ambiente descartável).

**O que o loop não faz:** GPU/Colab, anotação humana, push, chamadas pagas. Isso fica pendente nos
`docs/build/BUILD_REPORT_*.md`.

## Primeira vez

1. Abra o Docker Desktop. Em *Settings → Resources*, deixe pelo menos 4 GB de memória.
2. Entre no container e faça login no Claude **uma vez**. O login fica num volume do Docker, nunca no
   repositório:

   ```bash
   scripts/ralph/container.sh
   claude auth login        # abre um link; conclua no navegador
   exit
   ```

## Rodar a fila (à noite)

Com a árvore limpa e na branch da fila:

```bash
git switch ralph/noite-1
scripts/ralph/container.sh node scripts/ralph/preflight.mjs
scripts/ralph/container.sh node scripts/ralph/loop.mjs --config scripts/ralph/prd.noite1.json --timeout-minutes=60 --max-iterations=20
```

`--model` e `--reasoning-effort` (`low|medium|high|xhigh|max`) escolhem o modelo e o esforço do Claude. Os logs
ficam em `.ralph/run-*` (ignorado pelo git).

**Deixe o Mac ligado e na tomada.** Para ele não dormir durante a noite:

```bash
caffeinate -dimsu
```

## De manhã

```bash
git log --oneline main..ralph/noite-1     # um commit por história
cat scripts/ralph/progress.noite1.txt     # estado e lições
```

- Tudo verde: revise o diff e faça o merge na `main`.
- Parou com `blocked`: leia o motivo impresso e o último log em `.ralph/run-*`.
- `timeout`, `interrupted` ou limite de iterações: rode o mesmo comando do loop de novo. Ele retoma a
  história pendente a partir do working tree, então **não** faça `pull`, `stash` nem commit antes.
