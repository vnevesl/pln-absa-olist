# Roadmap — TP Final de PLN: ABSA em avaliações de e-commerce

> Versão 2 · 30/10/2026 · consolida o pré-projeto, a adaptação para o RePro e as 5 pesquisas em
> [`pesquisa/melhorias/`](../pesquisa/melhorias/). **Status: aprovado.** Prazo de entrega: **06/11/2026** (1 semana).

## 1. Pergunta de pesquisa e entregas

**Pergunta central.** Num problema de ABSA por categoria em português, treinar um encoder com rótulos
gerados por LLM chega perto de treiná-lo com rótulos humanos, nos mesmos textos? E o modelo resultante
detecta de fato problemas reais de entrega num domínio diferente (Olist)?

**Entregas finais:** repositório reprodutível, tabela de resultados com IC e testes, validação com o atraso
real, LIME/IG e fairness, e o relatório final.

Legenda de execução: 🤖 **Ralph** (loop noturno, CPU, container) · 🖥️ **Colab** (GPU T4, você roda) · 👤 **Humano** (você).

## 2. O que tínhamos × o que a pesquisa encontrou × decisão

| Tema | Plano anterior | O que a pesquisa encontrou | Decisão |
|---|---|---|---|
| **Entrada do modelo** | Só o corpo (`texto`) | O RePro foi anotado vendo **título + corpo**; a parte positiva dos casos mistos costuma estar no título (conferido: 99,9% dos títulos têm texto) [02, 03] | ✅ Entrada = título + corpo |
| **Taxonomia no prompt** | "Preço" em *outros* | No guia do RePro, custo-benefício e elogio geral são *produto*; estoque e pagamento são *outros* [03] | ✅ Alinhar definições ao guia do RePro |
| **Métrica principal** | F1 macro com "classes presentes" | Instável: 1 predição espúria de "neutro" muda 0,759 → 0,717. `_polarity_f1` subestima de 0,17 a 0,31 [04] | ✅ **F1 de pares (aspecto, polaridade)** como principal (padrão SemEval-2016); F1 macro com classes **fixadas pelo gabarito** como secundária; corrigir o bug |
| **Estatística** | 1 semente, bootstrap, McNemar | 1 semente não sustenta conclusão; McNemar tem 41–69% de poder para 2 p.p.; "melhor modelo" escolhido no teste [04] | ✅ 3 sementes nos modelos principais (prazo), bootstrap pareado, **Holm** numa família pré-definida, "melhor" escolhido na validação |
| **Casos mistos** | Descartados (2.018) | O descarte é enviesado (37,5% das avaliações 3★). A literatura usa um **teste separado** (MAMS), não uma classe "conflito". Dá para **treinar** com eles via perda de rótulo parcial [02] | ✅ Teste misto anotado à mão + perda de rótulo parcial no treino (experimento) |
| **Auditoria da conversão** | 50 casos | Com 50, o IC é largo demais (4–21% para 10% de erro); fazer **às cegas** transforma o RePro em "segundo anotador" grátis [02, 05] | ✅ Auditoria cega; tamanho na decisão D1 |
| **LLM anotador** | Qwen2.5-7B, regex, 6 exemplos fixos | Qwen2.5-14B-AWQ é bem melhor em PT (PoETa 71,0 × 63,7); **JSON Schema** com decodificação restrita elimina respostas inválidas; usar a probabilidade como confiança. Há 4 bugs no código [03] | ✅ Piloto 7B × 14B × Sabiá-4 decide (D2); vLLM + JSON Schema; corrigir os bugs |
| **Referência publicada** | — | Silva & Real (STIL 2024): BERTimbau com rótulos do Sabiá-3 = F1 0,62 × 0,68 com humanos no RePro [03] | ✅ Reproduzir e **ir além**: filtragem por confiança, rótulo suave, mistura humano+LLM, "equivalente-humano" |
| **Adaptação de domínio** | DAPT como melhoria principal | Mudança moderada e pouco lexical (96% dos tokens em comum; PAD 0,93/2); DAPT rende 0–1,5 p.p.; DANN piora [01] | ✅ **Treino misto RePro + Olist (LLM)** é a principal; TAPT só como ablação; sem DANN/SetFit |
| **Vazamento** | — | 26 dos 100 comentários do gabarito Olist estavam no conjunto que o LLM rotularia para treino [01] | ✅ Excluir o gabarito de qualquer treino; teste automatizado |
| **Validação com atraso** | AUC crua | A AUC crua (0,841) mal supera a da própria nota (0,817) [04] | ✅ Ganho **incremental** sobre a nota, AUC **dentro de cada nota**, aspecto-placebo, metade da Olist para desenvolvimento e metade para relato |
| **XAI** | LIME em 3 exemplos | Explicação sem medida de fidelidade não convence [04] | ✅ LIME + Integrated Gradients com teste de fidelidade (comprehensiveness/sufficiency) contra explicador aleatório |
| **Fairness** | F1 por grupo | Grupos com classes diferentes não são comparáveis; Norte com 26 casos [04] | ✅ Classes fixas, IC por grupo, sinalizar grupos pequenos |
| **Anotação** | Planilha | Página própria com atalhos e tarefas sob medida; **sem pré-anotação por IA** (ancoragem) [05] | ✅ Página HTML de arquivo único |

## 3. Decisões tomadas

| ID | Decisão | Escolha |
|---|---|---|
| **D1** | Tamanho da anotação humana | **Enxuta (~2 h 45):** Olist 100 · misto 100 · auditoria 50. Reavaliar após o piloto de 15 itens |
| **D2** | LLM anotador | Escolhido pelo **piloto** E3-S2 (Qwen2.5-7B × Qwen2.5-14B-AWQ × Sabiazinho-4) |
| **D3** | API da Maritaca | **Sim**, como segundo anotador e plano B. Chave só em variável de ambiente no Colab |
| **D4** | Prazo | **1 semana** → só histórias **M**; **3 sementes** (não 5); Albertina com 1 semente; TAPT cortado |

## 4. Épicos e histórias

Prioridade: **M** = obrigatório · **S** = importante · **C** = se houver tempo.

### E1 · Fundação correta 🤖 (pré-requisito de tudo)

| ID | Pri | História | Critério de aceite |
|---|---|---|---|
| E1-S1 | M | **Métricas:** F1 de pares como principal; F1 macro com classes fixadas pelo gabarito; corrigir `_polarity_f1`; seleção de época pela métrica principal | Testes reproduzem o caso "1 neutro espúrio" sem mudar a métrica; baseline recalculado |
| E1-S2 | M | **Entrada título + corpo** no RePro e na Olist (`texto_modelo`) | Teste garante a concatenação; baselines refeitos |
| E1-S3 | M | **Taxonomia alinhada ao guia do RePro** (`ASPECT_GUIDE` para o prompt) | Guia e prompt citam custo-benefício → produto etc. |
| E1-S4 | M | **Anti-vazamento:** gabarito Olist fora de qualquer treino; Olist dividida em desenvolvimento e relato | Teste falha se houver interseção de `review_id` |
| E1-S5 | M | **Bugs do LLM:** resposta inválida = descartada (não "ausente"); chave do cache inclui modelo e prompt; Maritaca `sabia-4`/`sabiazinho-4` | Testes unitários de cada bug |
| E1-REVISAO | M | Revisão: rerodar baselines (regras, TF-IDF) e atualizar o README | Tabela nova no README com a métrica principal |

### E2 · Anotação humana 🤖 + 👤

| ID | Pri | História | Tipo |
|---|---|---|---|
| E2-S1 | M | **Página de anotação** (HTML único): T1 Olist do zero, T2 casos mistos (só aspectos marcados pelo RePro, exige 1 positivo e 1 negativo), T3 auditoria cega com revelação depois; atalhos, salvamento automático, CSV no formato `asp_*`, tempo por item | 🤖 |
| E2-S2 | M | Gerar os lotes: T1 (sem vazamento), T2 (amostra estratificada dos 976), T3 (estratificada por nº de tópicos); só `review_id` + hash para textos do RePro | 🤖 |
| E2-S3 | M | **Piloto de 15 itens** → ajustar o guia → congelar | 👤 ~20 min |
| E2-S4 | M | Anotar T3 → T1 → T2 (tamanhos da D1) | 👤 ~2–4,5 h |
| E2-S5 | S | Reteste de 20 itens de T1 depois de 7 dias (consistência intra-anotador) | 👤 ~15 min |
| E2-S6 | M | **Análise de concordância:** κ e α com IC por bootstrap, matriz por aspecto, taxa de erro da conversão com IC de Wilson | 🤖 |

### E3 · LLM como anotador 🤖 + 🖥️

| ID | Pri | História | Tipo |
|---|---|---|---|
| E3-S1 | M | Backend **vLLM com JSON Schema** (enum por aspecto) e probabilidade por aspecto; backend Maritaca; few-shot fixo × por similaridade (kNN); 3 variantes de prompt | 🤖 |
| E3-S2 | M | **Piloto de decisão (D2):** 200 textos da validação × {7B, 14B-AWQ, Sabiazinho-4} × prompts | 🖥️ ~1 h |
| E3-S3 | M | Rotular com o modelo escolhido: treino RePro (5.589), teste RePro (1.597, LLM como modelo), Olist de desenvolvimento (~6.000) | 🖥️ 2–4 h, em várias sessões, retomável pelo cache |

### E4 · Modelos 🤖 + 🖥️

| ID | Pri | História | Tipo |
|---|---|---|---|
| E4-S1 | M | Encoder: sementes, condições de rótulo (humano, LLM bruto, LLM filtrado por confiança, LLM suave), curva de aprendizado (10/25/50/100%), mistura humano + LLM, controle "só validação humana" | 🤖 |
| E4-S2 | S | **Perda de rótulo parcial** para os casos mistos (forma fechada, código no relatório 02) | 🤖 |
| E4-S3 | M | **Treino misto RePro + Olist (LLM)** com peso 1,0 e 0,5 | 🤖 |
| E4-S4 | ~~C~~ cortada | Script de **TAPT** (MLM na Olist, whole-word masking, 10 épocas) como ablação | 🤖 |
| E4-S5 | M | Rodar: BERTimbau humano × LLM bruto × LLM filtrado com 3 sementes; curva de aprendizado (25/50%) com 1 semente; Albertina humano com 1 semente | 🖥️ ~3 h |
| E4-S6 | S | Rodar: rótulo parcial e treino misto RePro + Olist (3 sementes cada) | 🖥️ ~1,5 h |

### E5 · Avaliação 🤖 (+ 🖥️ para inferência)

| ID | Pri | História |
|---|---|---|
| E5-S1 | M | **Comparação:** média ± desvio entre sementes, bootstrap sobre sementes e avaliações, família de hipóteses pré-registrada + Holm, "equivalente-humano", teste misto e gabarito Olist separados |
| E5-S2 | M | **Validação com atraso:** ganho incremental sobre a nota, AUC dentro de cada nota, placebo (produto), regressão com controles, IC em tudo, metade de relato da Olist |
| E5-S3 | S | **XAI:** LIME + Integrated Gradients, fidelidade contra explicador aleatório |
| E5-S4 | S | **Fairness:** classes fixas, IC por grupo, grupos pequenos sinalizados |
| E5-REVISAO | M | Revisão cruzada: todo número do relatório sai de um arquivo de resultados versionado |

### E6 · Relatório final 🤖 + 👤

| ID | Pri | História | Tipo |
|---|---|---|---|
| E6-S1 | M | Rascunho: método, justificativa das mudanças em relação ao pré-projeto, resultados, limitações, checklist de reporte (relatório 04) | 🤖 |
| E6-S2 | M | Revisão, voz própria e entrega | 👤 |

## 5. Como o Ralph roda neste projeto

Adaptação do loop do CRM ([`snarktank/ralph`](https://github.com/snarktank/ralph)): um agente novo por
história, a fila em `scripts/ralph/prd.<épico>.json`, memória em Git + `progress.txt`, e verificações
obrigatórias antes do commit.

- **Checks por história:** `pytest`, `ruff check`, validação de sintaxe dos notebooks e, quando houver, o
  script de reprodução do baseline.
- **Só roda 🤖.** Histórias 🖥️ e 👤 ficam fora da fila; o Ralph deixa notebooks prontos para você só
  apertar "Executar tudo".
- **Container obrigatório** (Docker já instalado), com `claude` autenticado dentro dele. Sem push, sem
  chaves, sem chamadas pagas. Passo a passo em [`scripts/ralph/README.md`](../scripts/ralph/README.md).
- **Restrição do hardware:** M3 com 8 GB de RAM. Não cabe LLM de 7–14B nem o treino com várias sementes. O
  Ralph treina no máximo modelos minúsculos de teste.

**Filas:** `prd.noite1.json` (E1 + E2-S1/S2 + E3-S1 + E4-S1/S3) → `prd.noite2.json` (E2-S6, E4-S2, E5,
E6-S1 esqueleto) → `prd.noite3.json` (E5-REVISAO, E6-S1 com os números finais).

## 6. Calendário (entrega 06/11)

| Dia | Manhã/tarde | Noite (Ralph) |
|---|---|---|
| **Qui 30/10** | Roadmap aprovado; infraestrutura do Ralph | 🤖 Noite 1: E1, página de anotação, backend do LLM, encoder |
| **Sex 31/10** | 👤 Revisar a noite 1 (~20 min); piloto de anotação + auditoria T3 (~45 min) · 🖥️ piloto do LLM (1 h) e rotulagem (2–4 h) | 🤖 Noite 2: concordância, rótulo parcial, avaliação, XAI, fairness |
| **Sáb 01/11** | 👤 Anotar T1 + T2 (~2 h) · 🖥️ treinos E4-S5 (~3 h) | — |
| **Dom 02/11** | 🖥️ E4-S6 (~1,5 h), notebooks 06–08 | — |
| **Seg 03/11** | 👤 Conferir resultados | 🤖 Noite 3: revisão final e relatório com os números |
| **Ter 04/11** | 👤 Revisar o relatório e colocar na sua voz | — |
| **Qua 05/11** | Folga para imprevistos | — |
| **Qui 06/11** | 👤 **Entrega** | — |

**GPU estimada:** cerca de 7–9 h de T4 [estimativa], em sessões de 2–4 h. Se faltar cota: cortar E4-S6, depois
a Albertina.

## 7. Riscos

| Risco | Mitigação |
|---|---|
| 14B-AWQ não cabe ou é lento na T4 | Piloto mede antes; plano B 7B ou Sabiá-4 via API |
| Cota do Colab acaba | Cache do LLM e checkpoints no Drive; cortar histórias C |
| Ralph faz algo errado à noite | Checks obrigatórios, uma história por commit, revisão de manhã; nada é enviado ao GitHub sem você |
| Anotação demora mais que o previsto | A página mede o tempo; D1 permite ficar na versão enxuta |
| Resultado "humano ≈ LLM" sem poder estatístico | Reportar "diferenças abaixo de X p.p. não são detectáveis", nunca "equivalentes" |
