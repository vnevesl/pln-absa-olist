# Tema 8 — Análise de Sentimentos Baseada em Aspectos (ABSA) em avaliações de e-commerce (Olist)

*Relatório de pesquisa preparado em 08/10/2026 como base para o pré-projeto.*

Convenção usada: **[verificado]** significa que o número foi conferido em fonte primária ou no próprio artigo. **[fonte secundária]** significa que o número aparece em análises públicas, mas não na documentação oficial. **[não verificado]** marca estimativas ou valores que o estudante deve confirmar com código no primeiro dia de projeto.

---

## 1. Resumo executivo

A análise de sentimento "por nota" no Olist é um tema saturado. Há centenas de notebooks no Kaggle, e a classificação binária já alcança ROC-AUC de 96,5 com TF-IDF + Regressão Logística e de 97,9 com BERTimbau Large (SOUZA; FILHO, 2021; 2022). Um pré-projeto que só repita isso não se destaca. O caminho para se destacar é a **ABSA por categoria de aspecto (ACSA)**: identificar, em cada avaliação, *sobre o que* o cliente fala (produto, entrega, condições de recebimento, fidelidade ao anúncio, atendimento) e *com qual polaridade*.

O Olist não tem rótulos de aspecto, e esse é o ponto que torna o projeto original e ainda viável. Recomendamos uma proposta com quatro peças:

1. **Taxonomia de aspectos reaproveitada do corpus RePro** (PROPOR 2024). São 10.000 avaliações da B2W anotadas por humanos com 6 tópicos e 4 polaridades, com kappa de 0,68 para tópico e 0,71 para polaridade (SILVA et al., 2024).
2. **Rotulagem fraca assistida por LLM** sobre os cerca de 41 mil comentários do Olist, feita com um LLM aberto (Qwen/Gemma/Llama) ou com o Sabiá-3, e validada contra um **conjunto-ouro anotado pelo estudante** (~600 a 800 avaliações, com dupla anotação parcial e kappa).
3. **Destilação para um encoder leve** (BERTimbau ou Albertina PT-BR) em classificação multirrótulo aspecto×polaridade, comparada com baselines que o estudante já domina (TF-IDF + LR/RF) e com o LLM *zero/few-shot*.
4. **Validação extrínseca com metadados estruturados do Olist.** Os aspectos de "Entrega negativa" previstos pelo modelo devem se correlacionar com atraso real, calculado pelas datas `order_delivered_customer_date` e `order_estimated_delivery_date`. Esse recurso transforma o projeto em uma aplicação de negócio mensurável.

Os ângulos extras que o estudante já domina se encaixam naturalmente: XAI (SHAP/LIME por aspecto), fairness por região (`customer_state`), LLM-as-a-Judge para auditar rótulos e auditoria do ruído nota→sentimento (no RePro, cerca de 17% das avaliações com 3 estrelas são positivas).

**Viabilidade: 8,5/10. Potencial de nota no rubric: 9/10.**

---

## 2. Datasets

### 2.1 Tabela-resumo

| Dataset | Domínio | Volume | Rótulos | Licença | Link |
|---|---|---|---|---|---|
| **Brazilian E-Commerce Public Dataset by Olist** | Marketplace, pedidos de 2016 a 2018 | ~100 mil pedidos [verificado]. Tabela de reviews com **99.224 linhas** [fonte secundária]. **~40.977 linhas com texto de comentário** [fonte secundária]. Arquivo `olist_order_reviews_dataset.csv` de 14,45 MB [verificado, API Kaggle] | Nota de 1 a 5, título, comentário e datas. Junção com pedidos (datas de entrega), itens, categorias e estado do cliente | **CC BY-NC-SA 4.0** [verificado, API Kaggle] | https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce |
| **B2W-Reviews01** | Americanas.com, jan. a mai. de 2018 | >130 mil avaliações [verificado] | Nota de 1 a 5, "recomendaria a um amigo", gênero, idade e UF do avaliador | CC BY-NC-SA 4.0 [verificado] | https://github.com/americanas-tech/b2w-reviews01 |
| **RePro** (PROPOR 2024) | Amostra da B2W-Reviews01, estratificada por nota (~2 mil por nota) | **10.000** avaliações [verificado] | **Multirrótulo de 6 tópicos**: ENTREGA, PRODUTO, CONDIÇÕES DE RECEBIMENTO, ANÚNCIO, OUTROS, INADEQUADA. **Polaridade em 4 classes**: positivo, negativo, neutro, positivo/negativo. Kappa: tópico 0,68 e polaridade 0,71 | O artigo declara CC BY-NC-SA 4.0. O card do HF mostra cc-by-4.0 (divergência a citar) | https://huggingface.co/datasets/lucasnil/repro e https://github.com/lucasnil/repro |
| **B2W-Reviews02** | Amostra da B2W-Reviews01 | 250 avaliações anotadas como ABSA [verificado, via RePro] | Tópico + polaridade | [não verificado] | citado em Silva et al. (2024) |
| **Brands.Br** | Mesmas 250 da B2W-Reviews02 | 250 [verificado, via RePro] | Tópicos multirrótulo | [não verificado] | https://github.com/metalmorphy/Brands.Br |
| **ABSAPT 2022 (IberLEF)** | TripAdvisor (hotéis) | Treino com 77 termos de aspecto únicos. Os 15 mais frequentes cobrem 79% das ocorrências, com média de 3,7 aspectos por review [verificado, Gomes et al.] | Extração de termo de aspecto (ATE) + polaridade por aspecto (SOE), em formato SemEval-2014 | Uso acadêmico [não verificado] | Overview em *Procesamiento del Lenguaje Natural* e CEUR Vol-3202 |
| **Corpora unificados de Souza & Filho** (Olist, B2W, Buscapé, UTLC-Apps, UTLC-Movies) | Avaliações em PT-BR | Olist pré-processado com ~38 mil amostras (30k/4k/4k), B2W com ~117 mil, Buscapé com ~73 mil, UTLC-Apps com ~969 mil, UTLC-Movies com ~1,19 milhão [verificado no artigo] | Polaridade binária e nota | Ver Kaggle | https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets |
| **UTLCorpus** (STIL 2019) | Filmes e apps, com votos de utilidade | Quase 3 milhões de reviews [verificado, via RePro] | Nota + *helpfulness* | [não verificado] | Sousa, Brum e Nunes (2019) |

### 2.2 Detalhes do Olist e o que precisa ser confirmado

- **O que está confirmado.** A página oficial descreve cerca de 100 mil pedidos feitos entre 2016 e 2018 em vários marketplaces brasileiros. Os nomes de empresas e parceiros citados nos textos das reviews foram substituídos por nomes de casas de *Game of Thrones*. A licença é CC BY-NC-SA 4.0 (API pública do Kaggle, consultada em 08/10/2026). A documentação sugere explicitamente "NLP nas reviews" como um dos usos.
- **Número de linhas.** Análises públicas reportam **99.224** reviews na versão atual (v2, 2021). Cópias antigas, como o espelho do Mode e a análise em R da NSYSU, mostram 100.000 linhas, o que indica que houve mais de uma versão. Os `review_id` não são todos únicos: na cópia de 100 mil linhas há 777 ids duplicados e 25 triplicados. **Ação no dia 1:** `df.shape`, `df.review_id.nunique()`.
- **Comentários com texto.** Uma análise pública reporta **40.977** linhas com `review_comment_message` preenchido depois de remover nulos [fonte secundária]. Os títulos (`review_comment_title`) faltam em ~88% das linhas [fonte secundária]. **Ação:** `df.review_comment_message.notna().sum()`.
- **Distribuição das notas.** A forte assimetria positiva é consistente entre as fontes: média ~4,0 a 4,1 e predominância de 5 estrelas. Souza e Filho (2021) relatam **70% positivo** no Olist já pré-processado. A distribuição aproximada mais citada é 5★ ≈ 57,3 mil, 4★ ≈ 19,1 mil, 3★ ≈ 8,2 mil, 2★ ≈ 3,2 mil e 1★ ≈ 11,4 mil (≈57,8% / 19,3% / 8,2% / 3,2% / 11,5%) **[não verificado em fonte primária; confirmar com `value_counts()`]**. Nos subconjuntos com texto, a proporção de notas baixas tende a ser maior, porque clientes insatisfeitos escrevem mais [hipótese a verificar].
- **Por que a junção com pedidos importa.** O arquivo de pedidos tem data de compra, data de entrega ao cliente e data estimada. Análises públicas, não revisadas por pares, relatam que a taxa de avaliações ruins sobe de ~9% para ~78% em pedidos atrasados, e que a nota média cai de ~4,3 para ~2,6 [fonte secundária]. Isso sustenta o uso do atraso como **sinal fraco e validação extrínseca** do aspecto "Entrega".

**Volume para o pré-projeto (critério 1,5 pt).**

- **Não rotulado:** cerca de 41 mil comentários do Olist, curtos, de uma a três frases em geral [não verificado].
- **Rotulado por fraca supervisão (LLM):** de 8 a 15 mil comentários, estratificados por nota e categoria.
- **Ouro humano no Olist:** de 600 a 800 comentários.
- **Rotulado externo:** RePro, com 10 mil avaliações da B2W, para treino e transferência de domínio.

Esses números são adequados. O ABSAPT 2022 inteiro tem escala de alguns milhares de frases, e o RePro mostra que 10 mil avaliações bastam para F1 entre 0,88 e 0,91 com BERTimbau.

---

## 3. Estado da arte (2023–2026), com foco em português

### 3.1 Tarefa e formulação

A ABSA abrange várias tarefas: extração de termo de aspecto (ATE), classificação de polaridade por aspecto (ASC), detecção de categoria (ACD), ACSA e tarefas compostas como triplas e quádruplas (ASQP). A taxonomia de referência é a de Zhang et al. (2022). Para avaliações curtas de e-commerce, em que muitos aspectos são **implícitos** ("chegou antes do prazo" se refere a Entrega sem citar a palavra), a formulação **ACSA**, com categoria predefinida + polaridade, é mais robusta e mais fácil de avaliar do que a ATE com spans. O próprio RePro adota categorias amplas e explica que elas são mais largas do que aspectos de produto (SILVA et al., 2024).

### 3.2 Resultados reportados em português

| Trabalho | Dados | Modelo | Resultado |
|---|---|---|---|
| Souza e Filho (2021, LA-CCI) | Olist, B2W, Buscapé, UTLC | TF-IDF + Regressão Logística | ROC-AUC de **96,5** no Olist e 98,2 na B2W |
| Souza e Filho (2022, arXiv) | Mesmos | BERTimbau Large *fine-tuned* | ROC-AUC de **97,9** no Olist. O ajuste fino **não superou** as *features* pré-treinadas (98,1) no Olist, o menor dataset |
| Silva et al. (2024, PROPOR), RePro | 10 mil da B2W | BERTimbau, lr 4e-5, batch 8, 7 a 10 épocas, split 70/30 | Tópicos: F1 macro médio **0,88** (Entrega 0,97, Produto 0,97, Cond. receb. 0,90, Anúncio 0,91, Inadequada 0,62). Polaridade: F1 **0,91** (Neutro 0,84, Pos/Neg 0,88) |
| Silva e Real (2024, STIL) | RePro | **Sabiá-3** como anotador de tópicos | O LLM vai bem nos tópicos frequentes e mal nos nuançados ou raros. Modelos treinados com rótulos do Sabiá-3 são promissores nas categorias comuns. **A supervisão humana continua essencial** (números no PDF, não extraídos aqui) |
| ABSAPT 2022 (IberLEF) | TripAdvisor | Ensemble RoBERTa-PT + mDeBERTa (ATE). Ensemble PTT5-large gerativo (SOE) | Melhor resultado: **acurácia 0,671 em ATE** e **acurácia balanceada 0,824 em SOE** (GOMES et al., 2022) |
| Eiras e Brito (2026, IEEE Access) | ~60 mil reviews de cerveja em PT, ouro com 1.712 itens | Sabiá-3 vs. GPT-4o mini, zero/one/few-shot, ABSA **não supervisionada** | F1 de **0,857** na extração de aspecto, **0,826** na categoria e apenas **0,552** no sentimento. O sentimento foi o mais sensível ao *prompt* |
| Schuck et al. (2025, JBCS) | 12 datasets de sentimento em PT-BR | 23 LLMs (13 multilíngues, 10 ajustados para PT), *in-context* | Claude-3.5-Sonnet, GPT-4o, DeepSeek-V3 e Sabiá-3 passam de **92% de acurácia**. Modelos de 7 a 13B passam de 90%. **A especialização em PT teve efeito misto** |

### 3.3 LLMs para ABSA: lições gerais

- **Ajuste fino supera *prompting* em tarefas estruturadas.** O GPT-3.5 ajustado atingiu F1 de 83,8 na tarefa conjunta ATE+polaridade do SemEval-2014, 5,7% acima do InstructABSA, mas com cerca de 1000× mais parâmetros (SIMMERING; HUOVIALA, 2023). Modelos LLaMA/Orca-2 ajustados superam o SOTA em 4 tarefas ABSA e se saem mal em zero/few-shot (ŠMÍD; PŘIBÁŇ; KRÁL, 2024).
- **LLMs vão bem em tarefas simples e perdem em tarefas estruturadas,** mas são superiores em cenários de poucos rótulos (ZHANG et al., 2023). Isso justifica usar o LLM como **anotador ou professor** e não como modelo final.
- **Transferência entre línguas e domínios.** Para encoders pequenos, a estratégia ótima depende da arquitetura, e encoders seguem competitivos em configurações simples (FEHLE et al., 2026, preprint, sem português).
- **LLM como anotador.** Gilardi, Alizadeh e Kubli (2023, PNAS) mostram que o ChatGPT zero-shot supera *crowd workers* em várias tarefas de anotação, a um custo cerca de 30× menor. No português de e-commerce, porém, o trabalho de Silva e Real (2024) com Sabiá-3 mostra degradação nas categorias raras.

### 3.4 Modelos candidatos (todos executáveis no Colab gratuito ou por API barata)

| Papel | Modelo | Observação |
|---|---|---|
| Encoder principal | **BERTimbau base** (110M) / large (335M) | Referência em PT-BR (SOUZA; NOGUEIRA; LOTUFO, 2020). Base cabe folgado em T4 |
| Encoder alternativo | **Albertina PT-BR** (DeBERTa; versões de 100M e 900M) | Arquitetura DeBERTa (RODRIGUES et al., 2023). A versão de 100M é comparável em custo ao BERTimbau |
| Encoder multilíngue | mDeBERTa-v3-base | Foi o melhor em ATE no ABSAPT 2022 |
| Gerativo pequeno em PT | **Tucano** (160M a 2,4B; Tucano 2 com 0,5 a 3,7B em 2026) | Treinado em GigaVerbo, com ~200B tokens (CORRÊA et al., 2024). Opcional como *baseline* gerativo |
| LLM anotador ou juiz | **Sabiá-3 / Sabiazinho-3** (API Maritaca) ou Qwen2.5-7B / Llama-3.1-8B / Gemma em 4 bits | O Sabiá-3 tem custo por token 3 a 4× menor que modelos de fronteira (ABONIZIO et al., 2024). Os abertos rodam em T4 com quantização |
| Ajuste fino de LLM (opcional) | QLoRA com Unsloth (Qwen2.5-7B tem notebook gratuito para T4) | O T4 não tem bfloat16, então é preciso usar fp16 |

---

## 4. Pipeline proposto (concreto, para o Colab)

**Título sugerido:** *"Do que o cliente reclama? ABSA por categoria em avaliações do Olist com rotulagem assistida por LLM, destilação para encoders em português e validação com dados logísticos"*.

**Etapa 0. Ingestão e EDA (semana 1).**
- Juntar `order_reviews` com `orders` (datas), `order_items`, `products` (categoria), `customers` (UF) e `sellers`.
- Calcular `atraso = delivered_customer − estimated` (em dias), o tamanho do comentário e a distribuição das notas.
- Fazer deduplicação de `review_id` e de textos idênticos, como "ok" e "recebi o produto".

**Etapa 1. Pré-processamento leve (semana 1).**
- Normalizar caixa, URLs e repetições ("muitooo") e corrigir espaços e pontuação.
- **Não remover stopwords nem negação** para os transformers. Usar o pré-processamento clássico só no baseline TF-IDF.
- Filtrar textos com menos de 3 tokens, que irão para uma análise à parte.

**Etapa 2. Taxonomia e guia de anotação (semana 2).**
- Adaptar as 6 categorias do RePro ao Olist: **Produto, Entrega, Condições de recebimento (avaria, item faltando, item errado), Anúncio/descrição, Atendimento/Vendedor, Outros**. Para cada categoria a polaridade é {pos, neg, neutro}. Tratar "não mencionado" como ausência.
- Escrever um guia de uma a duas páginas com exemplos, incluindo aspectos implícitos e adversativas. O artigo do RePro aponta que **mais de 20% dos erros** de polaridade vêm de conjunções adversativas.

**Etapa 3. Conjunto-ouro (semanas 2 a 4).**
- Selecionar de 600 a 800 comentários, estratificados por nota (sobreamostrando 2 e 3 estrelas) e pelos atrasos.
- O estudante anota tudo. Um segundo anotador anota ~150 para medir o **kappa de Cohen**, com meta ≥0,6, coerente com os 0,68 e 0,71 do RePro.
- Dividir em 300 para validação e calibração de *prompt* e o restante como **teste congelado**.

**Etapa 4. Rotulagem fraca (semanas 3 a 5).**
- **(a) LLM zero/few-shot com saída JSON estruturada:** `{aspecto: polaridade}`, com 5 a 8 exemplos do ouro de validação e temperatura 0.
- **(b) Funções de rotulagem heurísticas, no estilo Snorkel:** léxico de entrega ("chegou", "prazo", "correios", "atras*"), avaria ("quebrado", "amassado") e o sinal de **atraso real > 0 ⇒ Entrega negativa provável**.
- **(c) Combinação** por voto ponderado ou por modelo de rótulos.
- Volume: de 8 a 15 mil comentários. Em LLM aberto de 7B a 4 bits no T4, a vazão estimada é de ~0,5 a 2 s por comentário com geração curta em lote, ou seja, de ~2 a 8 h divididas em sessões [estimativa, não verificado]. Pela API do Sabiazinho/Sabiá o custo deve ser baixo e o tempo de minutos a poucas horas [estimativa, não verificado].

**Etapa 5. Modelos (semanas 5 a 8).**
- **B1, baseline clássico:** TF-IDF (1–2-gramas) + Regressão Logística / Random Forest **one-vs-rest por aspecto×polaridade**. É o que o estudante já domina.
- **B2:** Embeddings fastText + LightGBM/MLP.
- **M1:** BERTimbau-base multirrótulo (cabeça sigmoide com 6×3 saídas, ou 6 cabeças softmax de 4 classes incluindo "ausente"). Hiperparâmetros típicos, seguindo o RePro e o ABSAPT: lr de 2e-5 a 4e-5, batch de 16 a 32, max_len de 128, 3 a 5 épocas, AdamW, warmup de 10% e fp16. Custo estimado em T4 para ~15 mil textos curtos: **~15 a 40 min por execução** [estimativa].
- **M2:** Albertina PT-BR 100M, com a mesma configuração.
- **M3, transferência:** pré-ajuste no **RePro** (tópicos da B2W) seguido de ajuste no Olist fraco. Comparar com treino só no Olist e com **zero-shot de domínio** (RePro→Olist).
- **M4, LLM zero/few-shot** como teto de referência (já calculado na etapa 4).
- **Opcional (M5):** QLoRA em Qwen2.5-7B ou Gemma, se sobrar tempo.

**Etapa 6. Avaliação (semanas 8 a 10).**
- Métricas no ouro:
  - F1 micro/macro por aspecto (detecção);
  - F1 macro da polaridade condicionada ao aspecto;
  - F1 da tupla (aspecto, polaridade);
  - matriz de confusão por aspecto.
- **Intervalos de confiança por bootstrap** (1.000 reamostragens) e teste de McNemar entre modelos.
- **Validação extrínseca:**
  - correlação ponto-bisserial ou AUC entre "Entrega negativa prevista" e atraso real;
  - regressão logística da nota baixa em função dos aspectos previstos, para estimar o *driver* de insatisfação por aspecto.
- **Auditoria de ruído nota→texto:** aplicar *confident learning* (NORTHCUTT; JIANG; CHUANG, 2021; biblioteca `cleanlab`) para listar reviews com nota incoerente com o texto e checar uma amostra manualmente.

**Etapa 7. XAI, fairness e LLM-as-a-Judge (semanas 9 a 11).**
- SHAP/LIME por aspecto, para confirmar que "Entrega" ativa "prazo/chegou" e não a categoria do produto.
- Fairlearn `MetricFrame` por **região ou UF do cliente** e por **categoria de produto**, para ver se o modelo erra mais com regionalismos (por exemplo, Norte/Nordeste com menos dados).
- LLM-as-a-Judge para auditar uma amostra das discordâncias entre modelo e LLM anotador, calibrado contra o ouro (ZHENG et al., 2023).

**Etapa 8. Aplicação e escrita (semanas 11 e 12).**
- Painel simples (Streamlit ou notebook) com a matriz **aspecto × categoria de produto × UF** e a evolução temporal das queixas.
- Relatório final.

**Recursos:** Colab T4 gratuito (~15 GB de VRAM, sessões de até ~12 h com limites variáveis) e Google Drive para *checkpoints*. Nenhuma etapa obrigatória exige mais que uma T4.

---

## 5. Desafios documentados e mitigações

| Desafio | Evidência | Mitigação |
|---|---|---|
| **Ruído nota→sentimento** | No RePro, ~17% das avaliações de 3★ são positivas, 154 avaliações de 5★ são mistas e 4 de 1★ são positivas (SILVA et al., 2024) | Não usar a nota como rótulo de aspecto. Usar a nota só como atributo ou sinal fraco. Auditar com cleanlab |
| **Desbalanceamento** | Olist com ~70% de positivos (SOUZA; FILHO, 2021). Na ABSA, os aspectos raros têm poucos exemplos (no RePro, "Inadequada" tem F1 de 0,62) | Amostragem estratificada no ouro, *class weights* ou *focal loss*, F1 macro como métrica principal e limiares ajustados por aspecto |
| **Aspectos implícitos e textos curtos** | Motivo para preferir ACSA a ATE (ZHANG et al., 2022) | Formular como ACSA. Usar *few-shot* com exemplos implícitos no *prompt* |
| **Adversativas e polaridade mista** | Mais de 20% dos erros de polaridade no RePro vêm de adversativas. "Pos/Neg" e "Neutro" são as classes mais difíceis (F1 de 0,88 e 0,84) | A polaridade por aspecto resolve parte da mistura. Fazer análise de erros dedicada |
| **Erros de digitação, gírias e regionalismos** | O texto do RePro foi mantido "exatamente como escrito". O mesmo vale para o Olist | Tokenização *subword* robusta. Normalização leve. Análise de fairness por UF |
| **Qualidade do rótulo do LLM** | O Sabiá-3 é fraco em categorias raras ou nuançadas (SILVA; REAL, 2024). No ABSA de cervejas, o sentimento teve F1 de apenas 0,55 e foi sensível ao *prompt* (EIRAS; BRITO, 2026) | Ouro humano obrigatório, calibração de *prompt* no conjunto de validação, voto entre LLM e heurísticas, e relatório de concordância LLM↔humano |
| **Contaminação de LLM** | O Olist é público desde 2018 e muito usado, então pode estar no pré-treino dos LLMs [risco plausível, não medido] | O teste final é o **ouro novo** anotado pelo estudante, cujos rótulos de aspecto não existem publicamente. Reportar isso explicitamente |
| **Anonimização "Game of Thrones"** | Nomes de empresas trocados por casas de GoT (documentação do Kaggle) | Normalizar as entidades para um *token* genérico, como `<LOJA>` |
| **Domínio cruzado B2W→Olist** | Os tópicos da B2W são parecidos, mas o Olist concentra queixas logísticas [hipótese] | Esse é um experimento do projeto, não um problema: medir a queda zero-shot e o ganho com o ajuste |
| **Licença NC** | Olist e B2W são CC BY-NC-SA | Uso acadêmico é permitido. Declarar a licença e não comercializar |

---

## 6. Ângulos para se destacar

1. **Taxonomia validada + ouro novo para o Olist.** Seria o primeiro conjunto de avaliação de ABSA no Olist com kappa reportado. É uma contribuição concreta e publicável em workshop do STIL ou do PROPOR.
2. **Validação extrínseca com logística.** Usar o atraso real como verdade de campo para o aspecto Entrega. Poucos trabalhos de PLN têm esse "sensor" externo, e ele conversa diretamente com o negócio.
3. **"Professor LLM → aluno encoder" (destilação de rótulos)** com análise de custo e benefício: F1, tempo de inferência e custo por mil reviews. Mostra maturidade de engenharia.
4. **Transferência de domínio B2W(RePro)→Olist,** respondendo a uma pergunta de pesquisa clara.
5. **Auditoria de ruído nota×texto com *confident learning*** e quantificação do quanto a nota engana.
6. **XAI por aspecto + fairness por região e categoria**, usando ferramentas que o estudante já domina, aplicadas de forma nova.
7. **LLM-as-a-Judge calibrado:** medir a concordância do juiz com o ouro antes de confiar nele, em vez de usá-lo cegamente.
8. **Aplicação:** um painel de "motivos de insatisfação" por categoria, UF e mês, e a priorização de vendedores ou transportadoras.


---

## 7. Viabilidade: **8,5/10**

**A favor:**
- Os dados são públicos, pequenos (o CSV de reviews tem 14,5 MB) e licenciados para uso acadêmico.
- Os textos são curtos, o que deixa o ajuste fino de BERTimbau-base em T4 na casa de dezenas de minutos.
- Existem baselines publicados para comparação: RePro, ABSAPT e Souza e Filho.
- O RePro fornece uma taxonomia pronta e dados de treino humanos.
- Todas as etapas obrigatórias rodam no Colab gratuito.
- O LLM entra apenas como anotador de um subconjunto, o que limita o custo.

**Riscos (−1,5):**
- **(i)** A anotação manual de 600 a 800 itens consome de 15 a 25 h [estimativa], o que é o maior custo humano.
- **(ii)** As cotas e a instabilidade do Colab gratuito podem atrasar a inferência do LLM 7B. Plano B: Sabiazinho via API, LLM de 3 a 4B ou um subconjunto menor.
- **(iii)** A qualidade do rótulo fraco pode ser baixa em categorias raras. Plano B: fundir "Anúncio" e "Outros" e focar em 4 aspectos.
- **(iv)** Os números exatos do Olist ainda precisam ser confirmados no dia 1 (sem risco real).

**Cronograma (12 semanas):**

| Semanas | Atividade |
|---|---|
| S1 | EDA e junções |
| S2 | Taxonomia e guia |
| S2 a S4 | Ouro e kappa |
| S3 a S5 | Rotulagem fraca |
| S5 a S8 | Modelos B1, M1, M2 e M3 |
| S8 a S10 | Avaliação e validação extrínseca |
| S9 a S11 | XAI, fairness e juiz |
| S11 e S12 | Painel e escrita |

Há uma folga de cerca de 1 semana.

---

## 8. Potencial de nota no rubric: **9/10**

- **Fundamentação técnica (4,0): forte.** A tarefa é bem definida (ACSA, multirrótulo), com justificativa para cada escolha:
  - encoder PT versus LLM, apoiado por Simmering e Huoviala, Šmíd et al. e Zhang et al.;
  - supervisão fraca versus anotação manual, apoiada por Gilardi et al. e Silva e Real;
  - escada de baselines e métricas apropriadas.
  
  Combina supervisionado e fraco e menciona clássicos (TF-IDF) e o estado da arte. Estimativa: 3,6 a 3,9.
- **Coleta e modelagem de dados (1,5): forte.** Origem, licença e volumes (99 mil / 41 mil / 10 mil / 600 a 800) e a junção relacional com os dados logísticos. Estimativa: 1,4 a 1,5.
- **Desafios e viabilidade (1,5): forte**, com riscos documentados na literatura e planos B. Estimativa: 1,3 a 1,5.
- **Contexto, objetivos e justificativa (1,5): forte.** O problema de negócio é claro (o "porquê" da nota baixa), há lacuna de dados ABSA em PT e as perguntas de pesquisa são explícitas. Estimativa: 1,4 a 1,5.
- **Texto (1,5):** depende da redação. O tema rende um texto dissertativo coeso.

**Risco de nota:** se o pré-projeto ficar genérico ("sentimento no Olist com BERT"), cai para ~7. O diferencial está em **ABSA + ouro + validação logística + destilação**.

---

## 9. Referências (ABNT)

ABONIZIO, H. et al. **Sabiá-3 Technical Report**. arXiv:2410.12049, 2024. Disponível em: https://arxiv.org/abs/2410.12049. Acesso em: 8 out. 2026.

CORRÊA, N. K.; SEN, A.; FALK, S.; FATIMAH, S. **Tucano: Advancing Neural Text Generation for Portuguese**. arXiv:2411.07854, 2024. Disponível em: https://arxiv.org/abs/2411.07854. Acesso em: 8 out. 2026.

EIRAS, D. M. A.; BRITO, A. C. de. Unsupervised aspect-based sentiment analysis through LLM: a case study of an unlabeled Portuguese beer database. **IEEE Access**, 2026. Disponível em: https://ieeexplore.ieee.org/abstract/document/11505998. Notícia institucional: https://ipt.br/2026/05/07/unsupervised-aspect-based-sentimento-analysisthrough-llm-a-case-study-of-na-unlabeled-portuguese-beer-database/. Acesso em: 8 out. 2026. [Volume e páginas não verificados.]

FEHLE, J.; HELLWIG, N. C.; KRUSCHWITZ, U.; WOLFF, C. **Zero-Shot to Full-Resource: Cross-lingual Transfer Strategies for Aspect-Based Sentiment Analysis**. arXiv:2604.26619, 2026. Disponível em: https://arxiv.org/abs/2604.26619. Acesso em: 8 out. 2026.

GILARDI, F.; ALIZADEH, M.; KUBLI, M. ChatGPT outperforms crowd workers for text-annotation tasks. **Proceedings of the National Academy of Sciences**, v. 120, n. 30, e2305016120, 2023. DOI: 10.1073/pnas.2305016120. Disponível em: https://pmc.ncbi.nlm.nih.gov/articles/PMC10372638. Acesso em: 8 out. 2026.

GOMES, J. R. S. et al. Deep Learning Brasil at ABSAPT 2022: Portuguese Transformer Ensemble Approaches. In: IBERIAN LANGUAGES EVALUATION FORUM (IberLEF 2022). **Proceedings** [...]. CEUR Workshop Proceedings, v. 3202, 2022. Disponível em: https://arxiv.org/abs/2311.05051. Acesso em: 8 out. 2026.

IBERLEF 2022. **Preface – Iberian Languages Evaluation Forum 2022**. CEUR Workshop Proceedings, v. 3202, 2022. Disponível em: https://ceur-ws.org/Vol-3202/preface.pdf. Acesso em: 8 out. 2026.

NORTHCUTT, C.; JIANG, L.; CHUANG, I. Confident Learning: Estimating Uncertainty in Dataset Labels. **Journal of Artificial Intelligence Research**, v. 70, p. 1373-1411, 2021. Disponível em: https://arxiv.org/abs/1911.00068. Acesso em: 8 out. 2026. [Páginas não verificadas nesta pesquisa.]

OLIST. **Brazilian E-Commerce Public Dataset by Olist**. Kaggle, 2018 (v2 atualizada em 2021). Licença CC BY-NC-SA 4.0. Disponível em: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce. Acesso em: 8 out. 2026.

REAL, L.; OSHIRO, M.; MAFRA, A. B2W-Reviews01: an open product reviews corpus. In: SYMPOSIUM IN INFORMATION AND HUMAN LANGUAGE TECHNOLOGY (STIL), 12., 2019. **Proceedings** [...]. 2019. p. 200-208. Dados: https://github.com/americanas-tech/b2w-reviews01. Acesso em: 8 out. 2026.

RODRIGUES, J. et al. **Advancing Neural Encoding of Portuguese with Transformer Albertina PT-\***. arXiv:2305.06721, 2023. Disponível em: https://arxiv.org/abs/2305.06721. Acesso em: 8 out. 2026.

SCHUCK, A. da F.; GARCIA, G. L.; MANESCO, J. R. R.; PAIOLA, P. H.; PAPA, J. P. Evaluating Large Language Models for Brazilian Portuguese Sentiment Analysis: A Comparative Study of Multilingual State-of-the-Art vs. Brazilian Portuguese Fine-Tuned LLMs. **Journal of the Brazilian Computer Society**, v. 31, n. 1, p. 884-916, 2025. Disponível em: https://journals-sol.sbc.org.br/index.php/jbcs/article/view/5793. Acesso em: 8 out. 2026.

SILVA, L. N. dos S. et al. RePro: A Benchmark Dataset for Opinion Mining in Brazilian Portuguese. In: INTERNATIONAL CONFERENCE ON COMPUTATIONAL PROCESSING OF PORTUGUESE (PROPOR), 16., 2024, Santiago de Compostela. **Proceedings** [...]. ACL, 2024. v. 1, p. 432-440. Disponível em: https://aclanthology.org/2024.propor-1.44/. Acesso em: 8 out. 2026.

SILVA, L. N. dos S.; REAL, L. Automated Topic Annotation in Brazilian Product Reviews: A Case Study of Adversarial Examples with Sabia-3. In: SIMPÓSIO BRASILEIRO DE TECNOLOGIA DA INFORMAÇÃO E DA LINGUAGEM HUMANA (STIL), 15., 2024, Belém. **Anais** [...]. Porto Alegre: SBC, 2024. p. 484-492. DOI: 10.5753/stil.2024.31167. Disponível em: https://sol.sbc.org.br/index.php/stil/article/view/31167. Acesso em: 8 out. 2026.

SIMMERING, P. F.; HUOVIALA, P. **Large language models for aspect-based sentiment analysis**. arXiv:2310.18025, 2023. Disponível em: https://arxiv.org/abs/2310.18025. Acesso em: 8 out. 2026.

ŠMÍD, J.; PŘIBÁŇ, P.; KRÁL, P. LLaMA-Based Models for Aspect-Based Sentiment Analysis. In: WORKSHOP ON COMPUTATIONAL APPROACHES TO SUBJECTIVITY, SENTIMENT & SOCIAL MEDIA ANALYSIS (WASSA), 14., 2024. **Proceedings** [...]. 2024. Disponível em: https://arxiv.org/abs/2508.08649. Acesso em: 8 out. 2026.

SOUSA, R. F. de; BRUM, H. B.; NUNES, M. G. V. A bunch of helpfulness and sentiment corpora in Brazilian Portuguese. In: SYMPOSIUM IN INFORMATION AND HUMAN LANGUAGE TECHNOLOGY (STIL), 2019. **Proceedings** [...]. SBC, 2019. [Citado via Silva et al. (2024); páginas não verificadas.]

SOUZA, F. D.; FILHO, J. B. de O. e S. **Sentiment Analysis on Brazilian Portuguese User Reviews**. In: IEEE LATIN AMERICAN CONFERENCE ON COMPUTATIONAL INTELLIGENCE (LA-CCI), 2021. arXiv:2112.05459. Disponível em: https://arxiv.org/abs/2112.05459. Acesso em: 8 out. 2026.

SOUZA, F. D.; FILHO, J. B. de O. e S. **BERT for Sentiment Analysis: Pre-trained and Fine-Tuned Alternatives**. arXiv:2201.03382, 2022. Disponível em: https://arxiv.org/abs/2201.03382. Acesso em: 8 out. 2026.

SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: pretrained BERT models for Brazilian Portuguese. In: CERRI, R.; PRATI, R. C. (ed.). **Intelligent Systems – BRACIS 2020**. Cham: Springer, 2020. p. 403-417. DOI: 10.1007/978-3-030-61377-8_28. Modelo: https://huggingface.co/neuralmind/bert-base-portuguese-cased. Acesso em: 8 out. 2026.

ZHANG, W.; DENG, Y.; LIU, B.; PAN, S. J.; BING, L. **Sentiment Analysis in the Era of Large Language Models: A Reality Check**. arXiv:2305.15005, 2023. Disponível em: https://arxiv.org/abs/2305.15005. Acesso em: 8 out. 2026.

ZHANG, W.; LI, X.; DENG, Y.; BING, L.; LAM, W. **A Survey on Aspect-Based Sentiment Analysis: Tasks, Methods, and Challenges**. arXiv:2203.01054, 2022. (Publicado na IEEE TKDE segundo Silva et al. (2024); não verificado diretamente.) Disponível em: https://arxiv.org/abs/2203.01054. Acesso em: 8 out. 2026.

ZHENG, L. et al. **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena**. arXiv:2306.05685, 2023 (NeurIPS 2023 Datasets and Benchmarks). Disponível em: https://arxiv.org/abs/2306.05685. Acesso em: 8 out. 2026. [Referência clássica, não reconsultada nesta pesquisa.]

**Fontes secundárias usadas apenas para os números do Olist (não revisadas por pares):** espelho do Mode, com 100.000 linhas (https://app.mode.com/brooklyndata/tables/olist_order_reviews_dataset/schema); análise em R da NSYSU (https://bap.cm.nsysu.edu.tw/msrc/2020rpb/6/olist01.html); artigo no Medium de I. Formiga sobre os 40.977 comentários (https://medium.com/@igorformiga125/processamento-de-linguagem-natural-nlp-tratamento-de-dados-32d3e94a322); notebook Zerve sobre atraso × avaliação ruim (https://www.zerve.ai/gallery/2bfc5bdc-a385-499d-a388-5e1c82406b73).
