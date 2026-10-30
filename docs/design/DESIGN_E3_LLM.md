# DESIGN E3 — LLM como anotador: backends, decodificação restrita e notebook de piloto

**Status: Pronto para execução** · aprovado em 30/10/2026 · roadmap: [`docs/ROADMAP.md`](../ROADMAP.md) §4 E3 · decisões D2 (piloto escolhe) e D3 (Maritaca sim)

## Objetivo

Deixar o notebook 04 pronto para o autor só apertar "Executar tudo" no Colab: piloto de decisão → configuração
congelada → rotulagem retomável. **Nada roda na GPU nesta história**; o código precisa ser testável em CPU
com geradores falsos.

## Referências obrigatórias

- `pesquisa/melhorias/03_llm_anotador.md` §2.3–2.4, §3, §6.1–6.2, §7.2 (todos os itens), §7.3, §8
- `docs/design/DESIGN_E1_FUNDACAO.md` E1-S3 e E1-S5 (já implementados; não refazer)

## E3-S1 · Backends e notebook 04

`src/absa/llm_labeler.py`:

1. `JSON_SCHEMA` (enum de `POLARITIES` por aspecto, todos obrigatórios, sem campos extras) e
   `PROMPT_VERSION`.
2. `build_prompt(texto, titulo=None, nota=None, examples=None, variant="v1")`, com as regras de 03 §7.2
   item 2 e `PROMPT_VARIANTS` com 3 variantes (v1, v2, v3). O texto do comentário entra como `texto_modelo`.
3. `vllm_generator(model_id, ...)` conforme 03 §7.2 item 4, devolvendo `{"raw", "probs"}`. Importa `vllm`
   **dentro** da função. Como a API do vLLM varia entre versões, a função tenta `StructuredOutputsParams` e cai
   para `GuidedDecodingParams`, com mensagem clara se nenhum existir.
4. `_aspect_probs(output)` conforme 03 §7.2 item 5: probabilidade das 4 classes por aspecto a partir dos
   logprobs. Teste com um objeto falso que imita a saída do vLLM.
5. `label_texts(...)` devolve `(rotulos, probs, raws)`. `probs` é `None` quando o backend não fornece.
   Mantém o cache retomável do E1-S5.
6. `openai_compatible_generator`: concorrência (ThreadPool, 4 workers) e leitura da chave **só** de variável
   de ambiente (`MARITACA_API_KEY`); nenhuma chave em código, notebook ou log.
7. `hf_generator` continua como plano B (prompts ordenados por comprimento).
8. `select_fixed_examples(pool, k=12, seed)`: exemplos fixos que cobrem cada aspecto ao menos uma vez com
   polaridade não ausente. kNN fica fora do escopo (prazo).

`notebooks/04_rotulagem_llm.ipynb` (reescrever, conforme 03 §7.3, com estes cortes de prazo):

1. Divide a validação: `val_demo` (200, pool de exemplos) e `val_dev` (599), estratificados e com semente fixa.
2. **Piloto (decisão D2):** 200 textos do `val_dev` × {Qwen2.5-7B-Instruct-AWQ, Qwen2.5-14B-Instruct-AWQ
   (vLLM), Sabiazinho-4 (API, se `MARITACA_API_KEY` existir)} × variante v1. Uma tabela com F1 de tupla, F1 por
   aspecto, κ por aspecto contra o humano, segundos por texto e % de respostas válidas. A célula seguinte tem
   `MODELO_ESCOLHIDO = "..."` para o autor preencher.
3. Rotula com a configuração congelada: treino RePro (5.589), `val_dev`, teste RePro (o LLM como modelo) e a
   amostra Olist (só `olist_split == "dev"`, sem o gabarito, 1.200 por nota). Grava CSVs com rótulo, `p_*` (5×4
   colunas), `modelo`, `prompt_version` e `valido`. Linhas inválidas são descartadas e contadas.
4. Registra o tempo por 1.000 textos e o total de tokens quando disponível.
5. Toda célula pesada é retomável pelo cache no Drive.

Testes (`tests/test_llm_labeler.py`): schema válido; `_aspect_probs` soma 1 por aspecto; `label_texts` com
gerador falso devolve probabilidades; variantes geram prompts diferentes; `select_fixed_examples` cobre os 5
aspectos; nenhuma string parecida com chave de API no repositório (`grep` por `sk-` e `MARITACA_API_KEY=`).

## Fora do escopo

Rodar qualquer LLM, kNN few-shot, voto de 3 prompts (cortado pelo prazo), teste de contaminação.
