# DESIGN E1 — Fundação correta (métricas, entrada, taxonomia, vazamento, bugs do LLM)

**Status: Pronto para execução** · aprovado em 30/10/2026

## Objetivo

Corrigir os defeitos encontrados pela pesquisa **antes** de qualquer treino na GPU. Nada aqui usa GPU nem rede.

## Referências obrigatórias (ler só as seções citadas)

- `pesquisa/melhorias/04_avaliacao_estatistica_xai.md` §2.1–2.4, §3.3, §8 P0 (itens 1–4) e P2 item 11
- `pesquisa/melhorias/03_llm_anotador.md` §1 item 1, §7.1, §7.2 itens 1, 2 e 6
- `pesquisa/melhorias/02_polaridade_mista.md` Resumo executivo item 5
- `pesquisa/melhorias/01_adaptacao_de_dominio.md` §5 Passo 0 (só a parte de vazamento)

## E1-S1 · Métricas

Arquivo: `src/absa/metrics.py` (+ `src/absa/models/encoder.py` só na seleção de época).

1. `labels_from_gold(y_true, include_absent=True) -> dict[col, list[str]]`: classes com suporte no gabarito,
   na ordem de `POLARITIES`.
2. `macro_f1(y_true, y_pred, labels=None, include_absent=False)`: usa `labels` fixos (calculados do gabarito
   se `None`). Por padrão **sem** "ausente" na média, conforme 04 §3.3.
3. `_polarity_f1`: só polaridades do gabarito nas posições mencionadas, com `labels=` explícito; uma falha de
   detecção não pode virar classe "ausente" na média.
4. `per_aspect_report(y_true, y_pred, labels=None)`: mesmas regras; as colunas continuam as atuais.
5. `tuple_f1` vira a **métrica principal**. `PRIMARY_METRIC = tuple_f1` exportado no módulo.
6. `bootstrap_ci(..., metric=PRIMARY_METRIC)` e `paired_bootstrap(..., metric=PRIMARY_METRIC, n=10_000)`: fixam
   `labels` **antes** do laço quando a métrica aceitar `labels`. P-valor **bilateral**
   (`2·min(P(d≤0), P(d≥0))`, limitado a 1) e erro padrão `ep`.
7. `holm(pvals: dict) -> dict` conforme 04 §8 P0 (3).
8. `encoder.py`: a seleção de época usa `tuple_f1` no dev (não mais `macro_f1`).

**Testes obrigatórios** (`tests/test_metrics.py`, novo arquivo):
- (a) uma predição espúria de classe ausente do gabarito muda `macro_f1` em < 1e-3;
- (b) `_polarity_f1` não muda quando o modelo deixa de detectar um aspecto mencionado;
- (c) `tuple_f1` = 1 para predição perfeita e 0 para tudo "ausente";
- (d) `holm` reproduz um exemplo calculado à mão;
- (e) p-valor bilateral ≈ 1 para dois modelos idênticos.

## E1-S2 · Entrada título + corpo

1. `src/absa/preprocess.py`: `model_text(titulo, texto) -> str`. Junta `"{titulo}. {texto}"` quando o título
   tem pelo menos uma letra e não é prefixo do corpo; senão devolve só o corpo. Títulos só de asteriscos ou
   pontuação são ignorados (04/02: 99,9% dos títulos do RePro têm texto).
2. `src/absa/repro.py` (`convert`) e `src/absa/data.py` (`build_reviews_table`) criam a coluna `texto_modelo`.
   `texto` continua existindo e inalterado (é a chave de deduplicação).
3. Notebooks 03–08 e `scripts/` passam a usar `texto_modelo` como entrada dos modelos. Amostragens e
   deduplicação continuam usando `texto`.
4. Testes: título vazio, título só "****", título igual ao início do corpo, título normal.

## E1-S3 · Taxonomia alinhada ao guia do RePro

1. `src/absa/config.py`: novo `ASPECT_GUIDE` (texto longo para o prompt) seguindo a tabela de 03 §7.1:
   custo-benefício e elogio geral → produto; não entrega, frete, retirada → entrega; "atendeu à expectativa" →
   recebimento; especificações técnicas → anúncio; estoque, pagamento, experiência de compra, dúvidas ao
   vendedor, texto sem sentido → outros. **Remover "preço" de outros.** `ASPECT_DESCRIPTIONS` continua curto,
   para tabelas, e consistente com o guia.
2. `src/absa/llm_labeler.py` (`build_prompt`) usa `ASPECT_GUIDE`.
3. `anotacao/guia_de_anotacao.md`: tabela dos 5 aspectos atualizada com as mesmas regras.
4. Teste: o prompt contém "custo-benefício" na linha de produto e não contém "preço" na linha de outros.

## E1-S4 · Anti-vazamento e divisão da Olist

1. `src/absa/sampling.py`:
   - `gold_exclusion(df)`: devolve a máscara booleana das linhas cujo `review_id` **ou** `texto` está em
     `data/annotated/olist_para_anotar.csv` (ou em `olist_anotado.csv`, se existir).
   - `olist_split(df, seed=SEED)`: divide a Olist em `"dev"` (50%) e `"relato"` (50%) **por `order_id`**, de
     forma determinística (hash de `order_id` + seed), sem depender da ordem das linhas.
2. Notebook 04 (seção da Olist): a amostra para rotulagem é tirada só de `olist_split == "dev"` e sem
   `gold_exclusion`. O notebook 07 reporta a validação extrínseca na metade `"relato"`.
3. Testes: (a) nenhum `review_id` nem texto do gabarito sobrevive à exclusão; (b) o split é determinístico e
   estável quando as linhas são embaralhadas; (c) nenhum `order_id` aparece nos dois lados.

## E1-S5 · Bugs do rotulador LLM

1. Chave do cache = `sha1(model_id | PROMPT_VERSION | prompt)`. O JSONL grava `model`, `prompt_version` e o
   `raw`. `label_texts(..., model_id=...)` é obrigatório.
2. Resposta inválida **não** vira "ausente": `label_texts` devolve `None` e o notebook 04 **descarta** a linha
   do treino, registrando a contagem.
3. `openai_compatible_generator`: padrão `model="sabiazinho-4"`; *retry* (até 3 tentativas) quando o parse
   falha.
4. Testes: mesma entrada com `model_id` diferente gera chaves diferentes; resposta inválida → `None`; caches
   antigos (sem `model`) continuam legíveis.

## E1-REVISAO · Baselines refeitos e README

1. Novo `scripts/run_baselines.py`: regras + TF-IDF (LogReg e RF), treino humano do RePro, entrada
   `texto_modelo`, métricas novas (principal, macro fixo e IC por bootstrap com 1.000 reamostras). Grava
   `results/baselines_repro.json` (versionado). Roda em CPU em menos de 2 min.
2. README: tabela "Resultados até agora" refeita a partir desse JSON, com a métrica principal na primeira
   coluna e uma frase explicando a troca de métrica (04 §2.1).
3. Conferir que todas as histórias E1 batem com este design; corrigir na origem o que divergir.

## Fora do escopo

GPU, download de modelos, chamadas pagas, notebooks 05–08 além da troca de coluna de entrada, anotação.
