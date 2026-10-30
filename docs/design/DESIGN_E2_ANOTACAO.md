# DESIGN E2 — Ferramenta e lotes de anotação humana

**Status: Pronto para execução** · aprovado em 30/10/2026 · roadmap: [`docs/ROADMAP.md`](../ROADMAP.md) §4 E2 · decisão D1 = enxuta

## Objetivo

Permitir que o autor faça as três tarefas de anotação em cerca de 2 h 45 min, com qualidade mensurável e sem
pré-anotação por IA.

| Tarefa | Itens | Origem |
|---|---|---|
| T1 · Olist do zero | 100 | `data/annotated/olist_para_anotar.csv` (já existe; não sortear de novo) |
| T2 · Casos mistos do RePro | 100 | 976 avaliações com polaridade mista e vários tópicos |
| T3 · Auditoria cega | 50 | Teste convertido do RePro, estratificado: 35 com vários tópicos, 15 com um |

## Referências obrigatórias

- `pesquisa/melhorias/05_processo_de_anotacao.md` §3.2–3.4, §6.1–6.3, §7
- `pesquisa/melhorias/02_polaridade_mista.md` §4.1–4.4 e §5.2–5.3

## E2-S1 · Página de anotação

Arquivo único `anotacao/anotar.html`, sem dependências externas nem rede, conforme 05 §6.2:

- Carrega um CSV escolhido pelo usuário (`<input type=file>`). Nenhum texto do RePro fica embutido na página.
- Detecta a tarefa pelo cabeçalho do CSV (coluna `tarefa` com valor `T1`, `T2` ou `T3`).
- Atalhos: `P E R A O` selecionam o aspecto; `1 2 3 0` = positivo, negativo, neutro, ausente; `Enter` salva e
  avança; `Backspace` volta; `D` marca dúvida; `Tab` vai para `obs`.
- **T2:** mostra só os tópicos marcados no RePro; os outros ficam fixos em "ausente". `Enter` é bloqueado
  sem ao menos um positivo e um negativo, a menos que `X` (indecidível) tenha sido marcado.
- **T3:** primeiro às cegas; depois do `Enter`, revela o rótulo convertido e pede o veredito (`C` concordo
  comigo · `R` RePro certo · `A` ambíguo).
- A nota em estrelas **nunca** aparece.
- Salvamento automático em `localStorage` (com `try/catch`), aviso para exportar a cada 25 itens, botão
  **Exportar CSV**, `t_inicio`/`t_fim` por item, contador de progresso, e funcionamento em tela de notebook.
- Exporta os formatos de 05 §6.3: T1 = mesmas colunas de `olist_para_anotar.csv` (compatível com
  `load_olist_gold`) + arquivo de log separado; T2 e T3 = `review_id`, `hash_texto`, `asp_*` e campos da tarefa.
- Teste automatizado sem navegador (`tests/test_anotar_html.py`): o HTML existe, não referencia URLs externas
  (`http`), contém os atalhos e os nomes de coluna exigidos. A lógica de validação de T2 fica numa função JS
  pura, testada por um script `node` chamado do pytest **se** `node` existir (senão `skip`).

## E2-S2 · Geração dos lotes

Novo `src/absa/annotation.py` + `scripts/make_annotation_batches.py`:

1. **T1:** reaproveita `olist_para_anotar.csv`, acrescenta a coluna `tarefa=T1` e embaralha com semente
   fixa. Grava `data/annotated/lote_T1.csv`.
2. **T2:** amostra estratificada por nota de 100 dos 976 casos mistos com vários tópicos, com semente fixa.
   Grava `data/annotated/lote_T2.csv` (fora do git; ver licença) com `review_id`, `hash_texto`, `titulo`,
   `texto`, `topicos` e `tarefa=T2`.
3. **T3:** 50 do **teste** convertido (35 com vários tópicos e 15 com um), com `asp_*_repro`. Grava
   `data/annotated/lote_T3.csv` (fora do git) com `tarefa=T3`.
4. `.gitignore`: `lote_T2.csv` e `lote_T3.csv` ficam fora do git. Os arquivos **anotados** de T2/T3 guardam só
   `review_id` + `hash_texto` + rótulos e entram no git.
5. `load_repro_annotations(arquivo)`: junta o arquivo anotado com o RePro pelo `review_id`, confere o hash e
   falha se não bater.
6. Testes: tamanhos exatos, estratificação, determinismo, nenhuma sobreposição entre T3 e casos mistos, hash
   conferido.

## Fora do escopo (próximas noites)

Análise de concordância (E2-S6), a anotação em si (humana), o reteste.
