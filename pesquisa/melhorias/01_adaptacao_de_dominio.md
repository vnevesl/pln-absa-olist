# Ponto fraco: mudança de domínio Americanas/B2W (RePro) → Olist

*Levantamento feito em 30/10/2026. Os números marcados como **[medido aqui]** foram calculados por mim nos dados locais do repositório (`data/raw/RePro.csv` e `data/interim/reviews.csv.gz`), com o código do pacote `absa` e sem alterar nenhum arquivo. Os scripts ficaram fora do repositório e o essencial deles está no Apêndice A. O que não consegui confirmar na fonte está marcado com [não verificado].*

---

## 1. Resumo executivo

1. **A mudança de domínio existe, mas é moderada e quase nada lexical.** 96% dos tokens da Olist já aparecem no RePro, e o *proxy A-distance* (PAD) com comprimentos casados é 0,93 numa escala de 0 a 2. Para comparar, a distância entre comentários 1★ e 5★ da própria Olist é 1,76. O que mais muda é o tamanho dos textos (mediana de 9 contra 16 palavras), o assunto (logística e casa contra eletrônicos) e a frequência dos rótulos: a entrega é prevista em 50% dos comentários da Olist e aparece em 30% do RePro. **[medido aqui]**
2. **Por isso, o DAPT/TAPT por MLM deve render pouco**, entre 0 e 1,5 ponto de F1. A literatura mostra os menores ganhos justamente no domínio de avaliações. Ainda assim é barato (cerca de 1 h de T4) e vale **como ablação** ("TAPT na Olist"), não como a melhoria principal.
3. **Prioridade 1 é medir direito.** Anote 200 comentários da Olist em vez de 100 (com 100, o IC95% do F1 macro tem cerca de 13 pontos de largura e o poder para detectar 5 pontos de diferença fica perto de 30%). Use a AUC de atraso, com IC, separando uma metade da Olist para desenvolvimento e outra para o relato. **Corrija antes um vazamento:** 26 dos 100 comentários do gabarito Olist estão na amostra que o notebook 04 manda o LLM rotular para treino. **[medido aqui]**
4. **Prioridade 2 é o treino misto RePro (humano) + Olist (LLM), sem o gabarito, com peso nos rótulos fracos.** É a única intervenção que ataca a mudança de frequência dos rótulos e a polaridade mista ("produto bom, mas atrasou"). Esses casos foram excluídos do RePro (2.018 avaliações descartadas, cerca de 20%) **[medido aqui]**, e o modelo treinado só no RePro nunca os vê.
5. **Não implementar DANN** (instável e, no estudo mais próximo, pior que não adaptar), nem SetFit (o ganho dele se concentra com 8 a 64 exemplos, e o projeto tem 5.589). Trocar de encoder (mmBERT, NorBERTo, BERTomelo, moBERTo) não muda a conclusão sobre domínio. A adaptação mais promissora depois do treino misto é o UDALM (MLM na Olist junto com a classificação), que fica como extra opcional.

---

## 2. Evidências

### 2.1 Diagnóstico: quanto a Olist difere do RePro? [medido aqui]

| Medida | RePro (7.985) | Olist (40.641) | Leitura |
|---|---|---|---|
| Palavras por texto (média / mediana / p95) | 21,9 / 16 / 57 | 11,7 / 9 / 33 | Olist tem textos com cerca de metade do tamanho |
| Tokens da Olist fora do vocabulário do RePro (após `clean_classic`) | — | 4,2% | quase nenhum vocabulário novo |
| Sobreposição dos 1.000 / 10.000 unigramas mais frequentes (sem stopwords) | — | 74% / 54% | Na escala de Gururangan et al. (2020, Fig. 2), 54% é como "pré-treino × notícias" (54,1%) e mais que "notícias × reviews" (40%). Os corpora aqui são menores, então o top-10k subestima a semelhança |
| Divergência de Jensen-Shannon de unigramas (amostras de 40 mil tokens) | intra-RePro 0,111 | intra-Olist 0,085 | **entre RePro e Olist: 0,19** (cerca de 2× o ruído de amostragem) |
| PAD, TF-IDF de palavras + LogReg, todos os textos | — | — | 1,07 (acurácia do discriminador 0,77) |
| **PAD com comprimento casado** (RePro reamostrado para o comprimento da Olist) | — | — | **0,93** (palavras) / 0,97 (caracteres) |
| PAD de referência: RePro × RePro / Olist × Olist | — | — | −0,02 / 0,06 (sem mudança) |
| PAD de referência: Olist atrasado × no prazo / Olist 1★ × 5★ | — | — | 1,26 / 1,76 |

Os termos mais sobre-representados na Olist são logísticos e de categorias de casa: *correios, entregue, pedidos, faltou, unidades, nf, greve, pacote, cortina, colcha, luminárias*. Os mais sobre-representados no RePro são de eletrônicos e da marca: *americanas, tv, samsung, lg, bateria, smartphone, aparelho, memória*.

A mudança de rótulo também aparece com o baseline TF-IDF + LogReg treinado no RePro. Na Olist ele prevê menção a **entrega em 49,8%** dos comentários, contra 30,4% no gabarito do teste do RePro, e **recebimento em 28,6%**, contra 15,8%. Ao mesmo tempo, prevê **produto negativo em só 8,9%**, contra 23,1% no RePro.

Como validação extrínseca, esse mesmo baseline já tem **AUC de 0,841** (IC95% de Hanley-McNeil 0,833–0,849; 38.796 pedidos entregues, 3.747 atrasados) entre P(entrega negativa) e atraso real. A AUC entre "algum aspecto negativo" e nota ≤ 2 é 0,914. **Esses são os patamares que qualquer encoder ou adaptação precisa superar.**

**Conclusão do diagnóstico.** Segundo o critério de Gururangan et al. (2020), "quanto mais distante o domínio, maior o potencial do DAPT", aqui o potencial do DAPT é baixo. A mudança relevante é de **frequência dos rótulos e estrutura** (textos curtos, mais logística, polaridade mista), e isso o MLM não corrige.

### 2.2 DAPT/TAPT com corpora pequenos

- **Referência clássica** ([Gururangan et al., 2020](https://aclanthology.org/2020.acl-main.740/); números extraídos de [ar5iv](https://ar5iv.labs.arxiv.org/html/2004.10964)):
  - O domínio REVIEWS teve os menores ganhos. No IMDB, RoBERTa ficou em 95,0, DAPT em 95,4, TAPT em 95,5 e DAPT+TAPT em 95,6. No Helpfulness, as mesmas etapas deram 65,1 ± 3,4, 66,5, 68,5 ± 1,9 e 68,7.
  - O MLM do RoBERTa em reviews caiu só de 2,10 para 1,93 com DAPT.
  - Receita do TAPT: **100 épocas**, máscara de 15% (dinâmica), lr de pico 1e-4 com batch 256, warmup de 6%. O TAPT é cerca de 60× mais rápido que o DAPT.
  - O "Curated-TAPT" com dados não rotulados do próprio domínio dá ganhos adicionais pequenos (IMDB: 95,5 → 95,7).
  - Desvios-padrão entre sementes (± 1 a 3 pontos) **são do mesmo tamanho dos ganhos**. Logo, sem várias sementes o efeito do TAPT é indistinguível do ruído.
- **Adaptação de domínio não supervisionada com cerca de 20 mil avaliações-alvo** (UDALM; [Karouzos; Paraskevopoulos; Potamianos, 2021](https://aclanthology.org/2021.naacl-main.203/), números de [ar5iv](https://ar5iv.labs.arxiv.org/html/2104.07078)). Este é o cenário mais parecido com o do projeto: Amazon, 12 pares de domínio e cerca de 20 mil textos-alvo não rotulados. A acurácia média foi:
  - BERT só na fonte: 89,93
  - **pré-treino de domínio (3 épocas de MLM): 90,78 (+0,85)**
  - adversarial (DAT/DANN): **87,68 (−2,25)**
  - **UDALM (MLM no alvo + classificação, ao mesmo tempo): 91,74 (+1,81)**

  Hiperparâmetros: lr 1e-5, máscara de 15% com 80/10/10, λ = n/(n+m). Os autores também observam que o PAD **não** se correlacionou de forma consistente com o erro no alvo.
- **ABSA em avaliações** ([Xu et al., 2019](https://aclanthology.org/N19-1242/)): o pós-treino do BERT em avaliações (BERT-DK) melhora a extração de aspecto e a classificação de sentimento por aspecto. Mas o pós-treino era sobre corpora de avaliações muito maiores [não verificado o tamanho exato] e partia do BERT genérico (Wikipedia e livros). O BERTimbau foi treinado no brWaC, um corpus da web com **whole-word masking** e 1 milhão de passos ([wiki do BERTimbau](https://github.com/neuralmind-ai/portuguese-bert/wiki)), e já deve ter visto muito texto de e-commerce [hipótese].
- **Quando o TAPT não ajuda:**
  - [Shi e Lipani (NeurIPS 2023)](https://arxiv.org/abs/2305.01711) mostram que o TAPT pode **piorar** o ajuste fino com cabeça [CLS], com queda de até 9,2% em tarefas de pares de sentenças, e é instável com ajuste fino por prompt.
  - [Ladkat et al. (AACL 2022)](https://aclanthology.org/2022.aacl-short.39/) relatam *overfitting* e esquecimento catastrófico com TAPT em bases pequenas. A saída proposta é atualizar só a camada de embeddings.
- **Evidência em português.** Não encontrei estudo de DAPT/TAPT do BERTimbau em **avaliações ou sentimento**. A evidência existente é em outros domínios e é mista:
  - Literatura ([Silva; Moro, PROPOR 2024](https://aclanthology.org/2024.propor-1.39/)): o corpus de MLM tinha 583.788 tokens, tamanho comparável aos cerca de 300 mil tokens de conteúdo da Olist. O treino usou 3 épocas, lr 5e-5, máscara de 15%, batch 16 e max_len 512. Ainda assim, o **LitBERTimbau (DAPT) ficou abaixo** do BERT-CRF com ajuste fino comum no F1 estrito de NER (0,753 contra 0,770, tabela 8 do artigo).
  - Jurídico: no LegalBert-pt o ganho do DAPT em recuperação foi marginal ([fonte secundária, ResearchGate](https://www.researchgate.net/publication/374645610_LegalBert-pt_A_Pretrained_Language_Model_for_the_Brazilian_Portuguese_Legal_Domain), [não verificado no texto completo]).
- **Self-training e TAPT se somam.** [Li et al. (Findings EMNLP 2021)](https://aclanthology.org/2021.findings-emnlp.86/) relatam ganhos aditivos com a ordem TAPT → ajuste fino → self-training em 6 tarefas, entre elas sentimento.

**Custo na T4** [estimativa, medir no Colab]:
- O notebook 05 informa cerca de 10 min para 4 épocas sobre 5.589 textos, ou seja, por volta de 22 mil sequências.
- O TAPT com cerca de 36 mil textos distintos da Olist, mais os 5,6 mil do treino do RePro, por 10 épocas dá cerca de 420 mil sequências. Como os textos têm metade do tamanho (max_len 64), devem ser **40 a 90 min**.
- Medir o tempo dos primeiros 200 passos antes de rodar tudo.

### 2.3 Rótulos de LLM, peso e mistura

- [Wang et al. (Findings EMNLP 2021)](https://aclanthology.org/2021.findings-emnlp.354/): rótulos do GPT-3 custam de 50% a 96% menos para atingir o mesmo desempenho. **Misturar rótulos humanos e de LLM** é melhor que qualquer um sozinho quando o orçamento é limitado, sobretudo se os humanos revisam os rótulos de baixa confiança do LLM.
- [Gilardi; Alizadeh; Kubli (PNAS 2023)](https://doi.org/10.1073/pnas.2305016120) já estão no projeto como justificativa para o LLM anotador.
- Esses trabalhos não dão uma receita de **peso** (ex.: 0,5 na perda dos exemplos do LLM). O peso é heurístico e deve ser tratado como hiperparâmetro, com no máximo 2 ou 3 valores fixados *a priori* [hipótese do autor].

### 2.4 Encoders mais novos

- **Albertina PT-BR 100M** ([Santos et al., SIGUL 2024](https://aclanthology.org/2024.sigul-1.14/); [card](https://huggingface.co/PORTULAN/albertina-100m-portuguese-ptbr-encoder)): DeBERTa com 100M de parâmetros, licença MIT, treinado em cerca de 3,7 bilhões de tokens do OSCAR. Já está no plano.
- **mmBERT** ([Marone et al., ICML 2026](https://arxiv.org/abs/2509.06888)): multilíngue, arquitetura ModernBERT, 3 trilhões de tokens em mais de 1.800 línguas. Não encontrei números específicos de português.
- **Encoders ModernBERT para português de 2026, todos *preprints* e sem avaliação em avaliações de produto ou ABSA:**
  - [NorBERTo](https://arxiv.org/abs/2605.00086), 331 bilhões de tokens: segundo o resumo, perde para o Albertina-900M e o BERTimbau-large em parte do ASSIN 2.
  - [BERTomelo](https://arxiv.org/abs/2606.28999).
  - [moBERTo](https://arxiv.org/abs/2606.22722): continuação do ModernBERT em 60 bilhões de tokens.
- **Na T4:** o FlashAttention-2 não suporta Turing ([repositório oficial](https://github.com/Dao-AILab/flash-attention)), então o ModernBERT roda com `attn_implementation="sdpa"` ([docs HF](https://huggingface.co/docs/transformers/main/en/model_doc/modernbert)) e perde parte da vantagem de velocidade.
- **Conclusão:** um encoder mais novo pode subir o F1 no RePro, mas não resolve a mudança de frequência dos rótulos nem a polaridade mista. Ele não muda a recomendação sobre domínio. Se sobrar tempo, mmBERT-base ou BERTomelo-base entram como **uma linha a mais na tabela**, nunca como substituto da análise de domínio.

---

## 3. Alternativas comparadas

O ganho esperado é em F1 macro na Olist, em relação a "BERTimbau treinado só no RePro". Todos os valores são **ordens de grandeza tiradas da literatura, não promessas**.

| Técnica | Ganho esperado | Custo (T4) | Risco | Evidência |
|---|---|---|---|---|
| **(c) Gabarito Olist + AUC de atraso** (medição, não melhoria) | — (é o que torna qualquer ganho mensurável) | 4–8 h humanas para 200 textos; GPU zero | Amostra estratificada não representa a Olist: reponderar | [Card et al., 2020](https://aclanthology.org/2020.emnlp-main.745/); simulação na §4 |
| **(b) Misto RePro + Olist-LLM (com peso)** | +2 a +6 pts na Olist [hipótese]; ≈0 no RePro | Rotular cerca de 6 mil textos com Qwen 4-bit (horas, [medir]); treino cerca de 2× o atual | Herdar vieses do LLM; vazamento do gabarito (já detectado) | [Wang et al., 2021](https://aclanthology.org/2021.findings-emnlp.354/); mudança de rótulo [medido aqui] |
| (a) TAPT/DAPT por MLM na Olist | 0 a +1,5 pt | 40–90 min + ajustes finos | Esquecimento, *overfitting*; ganho do tamanho do ruído entre sementes | [Gururangan et al., 2020](https://aclanthology.org/2020.acl-main.740/); [UDALM](https://aclanthology.org/2021.naacl-main.203/) (+0,85); [Silva; Moro, 2024](https://aclanthology.org/2024.propor-1.39/) (negativo) |
| UDALM (MLM no alvo + classificação, juntos) | +1 a +2 pts | Mudar `encoder.py` (cabeça MLM); treino 2–3× | Implementação; λ | [Karouzos et al., 2021](https://aclanthology.org/2021.naacl-main.203/) (+1,81 sobre só-fonte) |
| Self-training com o próprio encoder (professor RePro → pseudo-rótulos Olist com filtro de confiança) | +0,5 a +2 pts | 2–3 ajustes finos + inferência em 40 mil textos (minutos) | Viés de confirmação; classes raras (neutro) somem | [Du et al., 2021](https://aclanthology.org/2021.naacl-main.426/) (+1,2 em média); [Li et al., 2021](https://aclanthology.org/2021.findings-emnlp.86/) |
| Filtro por concordância LLM ∩ encoder (variação de (b)) | ganha precisão nos rótulos fracos, perde cobertura | Desprezível | Descarta justamente os casos difíceis | [hipótese], combina as duas linhas acima |
| Correção de *prior*/mudança de rótulo (EM de Saerens; BBSE) | ≈0 de F1; **melhora as prevalências** do mapa de reclamações do nb 07 | Minutos (CPU) | Supõe que P(x\|y) não muda, o que aqui é falso em parte | [Saerens et al., 2002](https://doi.org/10.1162/089976602753284446); [Lipton et al., 2018](https://proceedings.mlr.press/v80/lipton18a.html) |
| DANN / treino adversarial | negativo ou instável | Mudar o código + busca de λ | Alto: −2,25 pts no estudo mais próximo; soluções triviais | [Ganin et al., 2016](https://jmlr.org/papers/v17/15-239.html); [UDALM](https://ar5iv.labs.arxiv.org/html/2104.07078) |
| SetFit / embeddings de sentença + cabeça | ≤ 0 com 5,6 mil rótulos | Baixo | Ganho só no regime de poucos exemplos | [Tunstall et al., 2022](https://arxiv.org/abs/2209.11055): +19,3 com N=8, cai para +5,6 com N=64 |
| Trocar encoder (mmBERT, NorBERTo, BERTomelo, moBERTo) | incerto no RePro; não ataca o domínio | 1 ajuste fino cada; sem FA2 na T4 | *Preprints* de 2026 sem avaliação em reviews | §2.4 |

---

## 4. Como medir

### 4.1 Tamanho do gabarito Olist e intervalos

Simulei 300 réplicas de 5 aspectos × 4 classes, com prevalências parecidas com as da Olist, comparando um modelo com 85% de acerto por célula e outro com 80% **[medido aqui, simulação]**:

| n anotados | Largura média do IC95% (bootstrap) do F1 macro | Poder para detectar a diferença (cerca de 5 pts) com bootstrap pareado |
|---|---|---|
| 100 | **0,13** | 0,32 |
| 200 | 0,09 | 0,60 |
| 300 | 0,075 | 0,75 |
| 500 | 0,057 | 0,91 |

A simulação considera erros independentes entre os modelos. Na prática eles são correlacionados, o que aumenta o poder do teste pareado. Para uma proporção de 80% (ex.: acurácia de detecção de "entrega"), o IC de Wilson vai de 71,1% a 86,7% com n = 100 e de 73,9% a 85,0% com n = 200.

**Consequências:**
- Com 100 comentários dá para medir **a queda grosseira** (ex.: 0,80 no RePro contra 0,65 na Olist). **Não** dá para afirmar que o TAPT ganhou 1 a 2 pontos.
- Recomendo **200 comentários**. Se não houver tempo, 100 bastam desde que se use: (i) F1 de detecção por aspecto e F1 de tupla, que são mais estáveis que o macro de 4 classes; (ii) bootstrap pareado e McNemar, que já estão em `metrics.py`; (iii) a AUC de atraso como evidência principal de comparação.
- Isso reproduz o alerta de [Card et al. (2020)](https://aclanthology.org/2020.emnlp-main.745/): conjuntos de teste pequenos são subpotentes para diferenças pequenas.

**Reponderação.** `sampling.py` sobre-representa notas 1–3 e pedidos atrasados (30%). Reporte (i) o F1 "na amostra estratificada" e (ii) um F1 reponderado pelo inverso da probabilidade de inclusão (nota × atraso), com as frações calculadas sobre o `pool` da Olist. Sem isso, "a queda na Olist" mistura domínio com composição da amostra.

**Concordância.** Reanote 30 textos depois de uma semana, ou peça a um colega, e reporte o κ de Cohen. O RePro tem κ médio de 0,71 (segundo o PDF do [RePro](https://aclanthology.org/2024.propor-1.44/)). Sem esse número não dá para separar "o modelo errou" de "o gabarito Olist é ruidoso".

### 4.2 Métricas proxy (sem rótulo)

1. **PAD com controle de comprimento** (§2.1). É barato, mas o UDALM mostra que o PAD **não prevê** o ganho de uma adaptação específica. Use-o só para **descrever** a distância entre os domínios. Kashyap et al. ([NAACL 2021](https://aclanthology.org/2021.naacl-main.147/)) também encontram que medidas simples sobre distribuições de palavras são *baselines* fortes de correlação com a queda de desempenho.
2. **Sobreposição de vocabulário e Jensen-Shannon com amostras de mesmo tamanho**, sempre com a referência intra-domínio. Sem a referência, o JS não tem escala.
3. **Perda MLM do BERTimbau** no RePro-val e numa amostra de 2 mil textos da Olist, antes e depois do TAPT, como na tabela 1 de Gururangan et al. (2020). Se a perda na Olist já for próxima da do RePro, o TAPT tende a não ajudar. Se a perda no RePro-val subir muito depois do TAPT, houve esquecimento.
4. **Distribuição prevista dos rótulos** (Olist contra RePro) e a estimativa de *prior* por EM/BBSE. Isso documenta a mudança de rótulo, que é a mais importante aqui.

### 4.3 Validação extrínseca com o atraso (AUC)

- O baseline TF-IDF já atinge 0,841. **Relate todo modelo como ΔAUC sobre esse patamar**, com IC por bootstrap pareado por pedido ou pelo teste de [DeLong et al. (1988)](https://doi.org/10.2307/2531595). Com cerca de 39 mil pedidos o erro-padrão é de cerca de 0,004, o que permite detectar diferenças de cerca de 0,01.
- **Separe a Olist em duas metades por `order_id`**: "desenvolvimento" (escolha de peso, limiar e checkpoint) e "relato" (números finais). Escolher o modelo olhando a AUC e reportar a mesma AUC é vazamento.
- A AUC mede só um aspecto (entrega negativa) e só a sensibilidade ao atraso, e pode subir com um modelo que só aprendeu que "nota baixa = entrega ruim". Como controle, reporte também a AUC **dentro das notas 1–2** (atrasado × no prazo entre insatisfeitos). Ela isola se o modelo distingue o motivo da insatisfação.

---

## 5. Plano de implementação concreto

A ordem é por custo e benefício. Os nomes de arquivos referem-se ao repositório, e nada foi alterado.

**Passo 0. Corrigir o vazamento e fixar a medição** (CPU, cerca de meio dia)
- `notebooks/04_rotulagem_llm.ipynb`, seção 4: excluir de `pool` os `review_id` **e os textos** de `data/annotated/olist_para_anotar.csv`, já que existem textos duplicados. Hoje há **26 de 100** em comum. Fazer a mesma exclusão no corpus de TAPT.
- `src/absa/sampling.py`: acrescentar `pesos_inclusao(df, gold)` para o F1 reponderado. Se possível, aumentar `QUOTA_POR_NOTA` para 200 mantendo os 100 atuais (amostrar mais 100 com outra semente, excluindo os já sorteados).
- Novo `src/absa/shift.py`: `vocab_overlap`, `js_divergence(n_tokens=40_000)`, `proxy_a_distance(casar_comprimento=True)` e `mlm_loss(model, textos)`. O código mínimo está no Apêndice A.
- `src/absa/metrics.py`: `delivery_validation` deve devolver também o IC (bootstrap ou DeLong), e é preciso um `paired_auc_bootstrap(atrasado, p_a, p_b)`.

**Passo 1. Medir a queda zero-shot** (T4, cerca de 15 min)
- `notebooks/07_validacao_olist.ipynb`: para TF-IDF, BERTimbau-RePro e Albertina-RePro, reportar F1 no RePro-teste, F1 no gabarito Olist (amostra e reponderado, com IC) e AUC de atraso na metade "relato" (com IC). **Esse é o resultado central sobre domínio.**

**Passo 2. Treino misto RePro + Olist-LLM** (T4: rotulagem em horas [medir]; treinos de cerca de 15 min cada)
- Os rótulos já são gerados no notebook 04 (6 mil textos, 1.200 por nota), depois de aplicar o Passo 0.
- `src/absa/models/encoder.py`:
  - `_TextDataset` passa a aceitar a coluna `peso`.
  - `_loss` usa `reduction="none"` e uma média ponderada.
  - `fit` passa a aceitar `sample_weight`.
- Variantes fixadas *a priori*: peso do LLM ∈ {1,0; 0,5}, e "filtro de concordância", que mantém só os textos Olist em que LLM e BERTimbau-RePro concordam na detecção de entrega e de produto.
- Seleção de época: validação **humana** do RePro, como já é feito.
- 3 sementes por variante.

**Passo 3. TAPT como ablação** (T4: cerca de 1 h de MLM + 3 ajustes finos de cerca de 10 min)
- Novo `notebooks/05b_tapt_olist.ipynb`, com `AutoModelForMaskedLM` + `Trainer`. Configuração:
  - Corpus: textos distintos da Olist (sem o gabarito) + textos do treino do RePro. Separar 5% para validar a perda MLM.
  - `DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15, whole_word_mask=True)`. O BERTimbau foi pré-treinado com WWM; o parâmetro `whole_word_mask` existe na versão atual da [doc HF](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator), e em versões antigas o equivalente é `DataCollatorForWholeWordMask`. Verificar a versão no Colab.
  - `max_length=64` (p95 de 33 palavras), `per_device_train_batch_size=64`, `learning_rate=5e-5` (mais conservador que o 1e-4/batch 256 de Gururangan, porque o batch é menor), `warmup_ratio=0.06`, `weight_decay=0.01`, `num_train_epochs=10` (cerca de 6,5 mil passos), `fp16=True`, salvar nas épocas 3 e 10.
  - Equivalência: 10 épocas × cerca de 42 mil textos dá aproximadamente o mesmo número de exposições que as 100 épocas × cerca de 4 mil textos de Gururangan.
  - Parar se a perda MLM do RePro-val subir mais de cerca de 5% em relação ao início [heurística].
- `model.save_pretrained("models/bertimbau-tapt-olist")` + tokenizer. Em `ENCODERS` (`encoder.py`) basta acrescentar `"bertimbau_tapt": "<caminho>"`, porque `MultiAspectModel` usa `AutoModel.from_pretrained` e descarta a cabeça MLM. Não é preciso mudar mais nada.
- Combinar TAPT + misto só se cada um, sozinho, mostrar sinal.

**Passo 4 (opcional, se sobrar tempo)**
- Self-training: o professor é o melhor modelo do Passo 2. Ele gera pseudo-rótulos para os cerca de 30 mil textos Olist restantes, com limiar de confiança por aspecto e classe e no máximo k exemplos por classe, para não apagar as classes "neutro". Depois, 1 rodada de estudante.
- Ou UDALM: perda = CE + λ·MLM(Olist), com λ = n/(n+m).
- Escolher **um** dos dois.

**Custo total estimado de GPU:** de 6 a 10 h de T4, das quais a maior parte é a rotulagem do LLM já prevista no notebook 04. Os passos 1 a 3 somam cerca de 3 h [estimativa].

### Como reportar

1. **Seção "Mudança de domínio"**: a tabela §2.1 (comprimento, JS, PAD com referências, distribuição prevista dos rótulos).
2. **Tabela principal**, com uma linha por modelo/adaptação e colunas F1 RePro-teste (IC), F1 Olist-gold (IC; amostra/reponderado), F1 de detecção da entrega na Olist e AUC de atraso (IC) com Δ sobre o TF-IDF. Média ± desvio-padrão de 3 sementes.
3. **Frase honesta**: "com n = 100 (ou 200) o IC na Olist tem cerca de 13 (ou 9) pontos; diferenças menores que isso não são conclusivas, e a AUC de atraso é a evidência de comparação mais potente."
4. **Resultado negativo também é resultado**: se o TAPT não ajudar, isso confirma o diagnóstico de que o domínio está próximo lexicalmente (Gururangan et al., 2020; Silva; Moro, 2024).

---

## 6. Riscos

| Risco | Probabilidade | Mitigação |
|---|---|---|
| Vazamento do gabarito Olist no treino misto (26/100 hoje) | **certo, se nada for feito** | Passo 0 |
| Escolher o modelo pela AUC e reportar a mesma AUC | média | metades dev/relato por `order_id` |
| Ganho do TAPT menor que a variação entre sementes | alta | 3 sementes; reportar como ablação |
| O misto melhora a Olist mas piora o RePro-teste | média | reportar os dois; peso 0,5 |
| O LLM herda a taxonomia do RePro de forma diferente na Olist (ex.: "vendedor" em "outros") | média | medir a concordância LLM × gabarito Olist nos 100/200 |
| Polaridade mista: o RePro não tem exemplos, a Olist tem | alta | o misto com LLM cobre em parte; ver também `02_polaridade_mista.md` desta pasta |
| Esquecimento catastrófico no TAPT | baixa a média | lr 5e-5, perda MLM no RePro-val, checkpoint da época 3 |
| ModernBERT/mmBERT lentos ou incompatíveis na T4 | média | `sdpa`; ficam fora do caminho crítico |
| Gabarito Olist anotado só pelo autor | alta | κ intra-anotador em 30 textos |

**O que é marginal, dito com honestidade:** TAPT/DAPT, troca de encoder e SetFit. **O que é essencial:** medir com IC e evitar vazamento. **O que tem mais chance de mover o número na Olist:** o treino misto com rótulos do LLM no domínio-alvo.

---

## Apêndice A. Código mínimo das medidas (usado aqui; sugestão para `src/absa/shift.py`)

```python
from collections import Counter
import numpy as np, pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def js_divergence(textos_a, textos_b, n_tokens=40_000, seed=0):
    rng = np.random.default_rng(seed)
    amostra = lambda c: [c[i] for i in rng.permutation(len(c))[:n_tokens]]
    a = Counter(amostra([t for x in textos_a for t in x.split()]))
    b = Counter(amostra([t for x in textos_b for t in x.split()]))
    v = sorted(set(a) | set(b))
    pa, pb = (np.array([d[t] for t in v], float) for d in (a, b))
    return jensenshannon(pa / pa.sum(), pb / pb.sum(), base=2) ** 2

def proxy_a_distance(textos_a, textos_b, n=2000, seed=1):
    a = pd.Series(textos_a).sample(n, random_state=seed)
    b = pd.Series(textos_b).sample(n, random_state=seed + 1)
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(pd.concat([a, b]))
    y = np.r_[np.zeros(n), np.ones(n)]
    acc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=5).mean()
    return 2 * (1 - 2 * (1 - acc))   # 0 = indistinguíveis, 2 = separáveis
# Casar comprimento: reamostrar o RePro por faixas de nº de palavras com as frações da Olist antes do PAD.
```

---

## Referências (ABNT)

BEN-DAVID, S. et al. A theory of learning from different domains. *Machine Learning*, v. 79, p. 151–175, 2010. DOI: 10.1007/s10994-009-5152-4.

BLITZER, J.; DREDZE, M.; PEREIRA, F. Biographies, Bollywood, boom-boxes and blenders: domain adaptation for sentiment classification. In: ANNUAL MEETING OF THE ACL, 45., 2007, Praga. *Proceedings* […]. ACL, 2007. p. 440–447. Disponível em: https://aclanthology.org/P07-1056/.

CARD, D. et al. With little power comes great responsibility. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING (EMNLP), 2020. *Proceedings* […]. ACL, 2020. p. 9263–9274. Disponível em: https://aclanthology.org/2020.emnlp-main.745/.

DELONG, E. R.; DELONG, D. M.; CLARKE-PEARSON, D. L. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, v. 44, n. 3, p. 837–845, 1988. DOI: 10.2307/2531595.

DU, J. et al. Self-training improves pre-training for natural language understanding. In: NAACL-HLT, 2021. *Proceedings* […]. ACL, 2021. p. 5408–5418. Disponível em: https://aclanthology.org/2021.naacl-main.426/.

GANIN, Y. et al. Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, v. 17, n. 59, p. 1–35, 2016. Disponível em: https://jmlr.org/papers/v17/15-239.html.

GILARDI, F.; ALIZADEH, M.; KUBLI, M. ChatGPT outperforms crowd workers for text-annotation tasks. *PNAS*, v. 120, n. 30, e2305016120, 2023. DOI: 10.1073/pnas.2305016120.

GURURANGAN, S. et al. Don't stop pretraining: adapt language models to domains and tasks. In: ANNUAL MEETING OF THE ACL, 58., 2020. *Proceedings* […]. ACL, 2020. p. 8342–8360. Disponível em: https://aclanthology.org/2020.acl-main.740/.

HUGGING FACE. Data collator. *Transformers documentation*. Disponível em: https://huggingface.co/docs/transformers/main/en/main_classes/data_collator. Acesso em: 30 out. 2026.

HUGGING FACE. ModernBERT. *Transformers documentation*. Disponível em: https://huggingface.co/docs/transformers/main/en/model_doc/modernbert. Acesso em: 30 out. 2026.

KAROUZOS, C.; PARASKEVOPOULOS, G.; POTAMIANOS, A. UDALM: unsupervised domain adaptation through language modeling. In: NAACL-HLT, 2021. *Proceedings* […]. ACL, 2021. p. 2579–2590. Disponível em: https://aclanthology.org/2021.naacl-main.203/.

KASHYAP, A. R. et al. Domain divergences: a survey and empirical analysis. In: NAACL-HLT, 2021. *Proceedings* […]. ACL, 2021. p. 1830–1849. Disponível em: https://aclanthology.org/2021.naacl-main.147/.

LADKAT, A. et al. Towards simple and efficient task-adaptive pre-training for text classification. In: AACL-IJCNLP, 2022. *Proceedings* […] (short papers). ACL, 2022. p. 320–325. Disponível em: https://aclanthology.org/2022.aacl-short.39/.

LI, S. et al. Task-adaptive pre-training and self-training are complementary for natural language understanding. In: FINDINGS OF THE ACL: EMNLP 2021. ACL, 2021. p. 1006–1015. Disponível em: https://aclanthology.org/2021.findings-emnlp.86/.

LIPTON, Z.; WANG, Y.-X.; SMOLA, A. Detecting and correcting for label shift with black box predictors. In: ICML, 35., 2018. *Proceedings* […]. PMLR, v. 80, p. 3122–3130, 2018. Disponível em: https://proceedings.mlr.press/v80/lipton18a.html.

MARONE, M. et al. mmBERT: a modern multilingual encoder with annealed language learning. In: ICML, 2026. PMLR, v. 306. Preprint: arXiv:2509.06888. Disponível em: https://arxiv.org/abs/2509.06888.

LAITZ, T. et al. moBERTo: a modern encoder for Portuguese via continued pretraining of ModernBERT. arXiv:2606.22722, 2026. Disponível em: https://arxiv.org/abs/2606.22722.

OLIVEIRA, R. R. A.; VAN ERVEN, G. C. G.; GARCIA, L. P. F. BERTomelo: your Portuguese encoder best friend. arXiv:2606.28999, 2026. Disponível em: https://arxiv.org/abs/2606.28999.

RAMPONI, A.; PLANK, B. Neural unsupervised domain adaptation in NLP: a survey. In: COLING, 28., 2020. *Proceedings* […]. 2020. p. 6838–6855. Disponível em: https://aclanthology.org/2020.coling-main.603/.

SAERENS, M.; LATINNE, P.; DECAESTECKER, C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Computation*, v. 14, n. 1, p. 21–41, 2002. DOI: 10.1162/089976602753284446.

SANTOS, R. et al. Fostering the ecosystem of open neural encoders for Portuguese with Albertina PT* family. In: SIGUL @ LREC-COLING, 2024. *Proceedings* […]. 2024. p. 105–114. Disponível em: https://aclanthology.org/2024.sigul-1.14/.

SHI, Z.; LIPANI, A. Don't stop pretraining? Make prompt-based fine-tuning powerful learner. In: NEURIPS, 37., 2023. *Advances in Neural Information Processing Systems*. 2023. Disponível em: https://arxiv.org/abs/2305.01711.

SILVA, E. S. N. et al. NorBERTo: a ModernBERT model trained for Portuguese with 331 billion tokens corpus. arXiv:2605.00086, 2026. Disponível em: https://arxiv.org/abs/2605.00086.

SILVA, L. N. dos S. et al. RePro: a benchmark dataset for opinion mining in Brazilian Portuguese. In: PROPOR, 16., 2024. *Proceedings* […]. v. 1, p. 432–440. Disponível em: https://aclanthology.org/2024.propor-1.44/.

SILVA, M. O.; MORO, M. M. Evaluating pre-training strategies for literary named entity recognition in Portuguese. In: PROPOR, 16., 2024. *Proceedings* […]. Disponível em: https://aclanthology.org/2024.propor-1.39/.

SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: pretrained BERT models for Brazilian Portuguese. In: BRACIS, 9., 2020. *Intelligent Systems*. Springer, 2020. p. 403–417. Detalhes de pré-treino: https://github.com/neuralmind-ai/portuguese-bert/wiki.

TUNSTALL, L. et al. Efficient few-shot learning without prompts. arXiv:2209.11055, 2022. Disponível em: https://arxiv.org/abs/2209.11055.

WANG, S. et al. Want to reduce labeling cost? GPT-3 can help. In: FINDINGS OF THE ACL: EMNLP 2021. ACL, 2021. p. 4195–4205. Disponível em: https://aclanthology.org/2021.findings-emnlp.354/.

XU, H. et al. BERT post-training for review reading comprehension and aspect-based sentiment analysis. In: NAACL-HLT, 2019. *Proceedings* […]. ACL, 2019. p. 2324–2335. Disponível em: https://aclanthology.org/N19-1242/.
