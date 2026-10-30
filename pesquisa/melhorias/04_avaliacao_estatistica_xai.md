# Avaliação, estatística, validação extrínseca, XAI e fairness: auditoria e plano de melhorias

**Projeto:** ABSA em avaliações de e-commerce (RePro → Olist) · **Data:** 30/10/2026
**Escopo auditado:** `src/absa/metrics.py`, `src/absa/models/encoder.py` (seleção de época), `notebooks/06_comparacao.ipynb`, `07_validacao_olist.ipynb`, `08_xai_fairness.ipynb`.
**Método:** leitura do código, experimentos de verificação com o baseline TF-IDF + LogReg/RF reproduzido localmente (os números batem com o README: F1 macro 0,759, IC 0,730–0,783; F1 tupla 0,835) e revisão da literatura com fontes primárias. Os scripts de verificação ficaram fora do repositório (scratchpad da sessão); nenhum arquivo do projeto foi alterado.

---

## 1. Resumo executivo

1. **A métrica principal atual é instável por construção:** o F1 macro usa "as classes presentes no gabarito **ou na predição**". No teste, `recebimento` não tem nenhum "neutro"; **uma única predição espúria de "neutro" derruba o F1 macro médio de 0,759 para 0,717 (−0,042)**, mais que muitas diferenças reais entre modelos. Isso enviesa justamente a pergunta central (rótulos de LLM × humanos), porque o LLM tende a gerar "neutro".
2. **`_polarity_f1` tem um bug:** inclui "ausente" como classe (com F1 = 0) quando o modelo deixa de detectar um aspecto mencionado; o F1 de polaridade reportado sai subestimado em 0,17 a 0,31 por aspecto.
3. **Métrica principal recomendada:** F1 micro de pares (aspecto, polaridade), o padrão de SemEval-2016 e GermEval-2017 para esta formulação; como secundária, F1 macro com **conjunto de classes fixo, definido pelo gabarito**, mais detecção e polaridade condicional por aspecto.
4. **Estatística:** encoders com uma única semente não sustentam conclusões. O plano é rodar 5 sementes, fazer bootstrap pareado ou teste de aleatorização sobre a métrica principal e aplicar Holm na família de comparações. O McNemar sobre a avaliação inteira fica como teste secundário: a n = 1.597, ele só tem ~41–69% de poder para diferenças de 2 p.p.
5. **Validação extrínseca:** a AUC crua de P(entrega negativa) × atraso (0,841 no TF-IDF) mal supera a AUC da própria nota (0,817). A evidência convincente é o **ganho incremental sobre a nota** (AUC 0,815 → 0,862) e a **AUC dentro de cada nota** (0,64–0,74), com um aspecto-placebo (produto: 0,545).

---

## 2. Auditoria do `metrics.py` e dos notebooks (problemas encontrados)

Severidade: **[A]** altera conclusões · **[M]** distorce números reportados · **[B]** boa prática/robustez.

### 2.1 `macro_f1` e `per_aspect_report`: conjunto de classes muda com a predição **[A]**

`f1_score(..., average="macro")` sem `labels=` usa a união das classes de `y_true` e `y_pred` ([documentação do scikit-learn][sk-f1]). Distribuição do teste (n = 1.597), verificada nos dados:

| aspecto | ausente | positivo | negativo | neutro |
|---|---|---|---|---|
| produto | 395 | 807 | 369 | 26 |
| entrega | 1.111 | 295 | 170 | 21 |
| recebimento | 1.345 | 64 | 188 | **0** |
| anúncio | 1.468 | 25 | 96 | **8** |
| outros | 1.192 | 81 | 273 | 51 |

Consequências verificadas no baseline LogReg:

- **Descontinuidade:** basta trocar uma predição de `recebimento` de "ausente" para "neutro" e o F1 macro médio cai de **0,7589 para 0,7173**. A classe "neutro" entra no macro com F1 = 0 e vale 1/3 → 1/4 do aspecto. Com o conjunto de classes fixo (as do gabarito), a mesma perturbação dá 0,7589 → 0,7588.
- **Comparações injustas entre modelos:** o modelo que nunca prevê "neutro" em `recebimento` (LogReg, RF) é avaliado com 3 classes, e o que prevê ao menos uma vez é avaliado com 4. **Isso afeta diretamente a comparação humano × LLM:** um estudante treinado com rótulos de LLM tende a emitir "neutro" em mais aspectos e seria penalizado pela métrica, não pelo desempenho. O fenômeno é um caso do problema geral de agregação do F-score descrito por Forman e Scholz (2010) e Opitz e Burst (2019).
- **O treino nem permite aprender "neutro" em `recebimento`:** são 0 casos no treino, 1 na validação e 0 no teste. Com rótulos humanos, a classe é inalcançável.
- **A seleção de época no `encoder.py` também usa `macro_f1`** (no dev, que tem 1 "neutro" em `recebimento` e 3 em `anúncio`). Por isso a escolha do checkpoint pode ser decidida por uma ou duas predições raras.

**Correção:** fixar, uma vez e antes de olhar os resultados, `LABELS[aspecto]` = classes com suporte > 0 **no teste inteiro** e passar `labels=LABELS[asp]` em todo `f1_score`. Nas reamostras do bootstrap, mantenha o mesmo conjunto (uma classe que some numa reamostra conta como 0, com `zero_division=0`). O risco de isso acontecer é desprezível: com 8 casos, P(0 na reamostra) ≈ e⁻⁸ ≈ 0,03%.

### 2.2 `_polarity_f1`: "ausente" contamina o F1 de polaridade **[M], bug**

A máscara filtra só `t != "ausente"`. Quando o modelo prevê "ausente" num aspecto mencionado (falso negativo de detecção), "ausente" aparece em `p[mask]` e vira classe do macro com precisão 0 e F1 0. Verificado no LogReg:

| aspecto | atual | corrigido (`labels` = polaridades do gabarito) | FN de detecção |
|---|---|---|---|
| produto | 0,509 | 0,679 | 89 |
| entrega | 0,612 | 0,816 | 25 |
| recebimento | 0,617 | 0,926 | 33 |
| anúncio | 0,509 | 0,679 | 31 |
| outros | 0,621 | 0,828 | 58 |

O erro de detecção já entra em `f1_deteccao`. A coluna de polaridade deve medir só a polaridade e não punir esse erro de novo. Corrija com `labels=[l for l in ("positivo","negativo","neutro") if (t[mask]==l).any()]`. Uma alternativa mais limpa é reportar a acurácia/F1 de polaridade **só onde o gabarito e a predição concordam que o aspecto foi mencionado**, como no Slot 3 do SemEval, que avalia a polaridade dada a categoria-ouro ([Pontiki et al., 2016][semeval16]).

### 2.3 `bootstrap_ci` **[B]**

- O IC percentil com 1.000 reamostras é aceitável. Comparei com o BCa no LogReg: F1 macro com percentil [0,730; 0,784] e com BCa [0,734; 0,789]; F1 tupla com percentil [0,821; 0,847] e com BCa [0,821; 0,846]. A diferença é ≤ 0,005, ou seja, **trocar para BCa não muda conclusão**. O BCa corrige viés e assimetria ([Efron, 1987][efron87]) e é o padrão de `scipy.stats.bootstrap` ([SciPy][scipy-boot]), mas exige um jackknife de n = 1.597 avaliações da métrica (~50 s por modelo). Recomendação: manter o percentil e citar a checagem com BCa numa nota de rodapé.
- `alpha` e `n` estão bem parametrizados. Falta devolver também o erro-padrão (desvio das reamostras), que alimenta a análise de poder (seção 4.4).

### 2.4 `paired_bootstrap` **[M]**

- O p-valor `(diffs <= 0).mean()` é **unilateral** e mede a fração de reamostras em que A não supera B. Isso aproxima a confiança em "A > B" ([Koehn, 2004][koehn04]), mas não é um teste de hipótese nula centrada. A forma recomendada por Berg-Kirkpatrick, Burkett e Klein (2012) centra a distribuição: p = P(δ* > 2δ). Para relatório, o mais simples e defensável é reportar **diferença + IC95% pareado** e tratar o p-valor como bilateral, p = 2·min(P(δ*≤0), P(δ*≥0)), declarando isso no texto.
- Com 1.000 reamostras, o menor p possível é 0,001. Após Holm com ~8 comparações, isso ainda basta, mas use 10.000 reamostras para os p-valores finais (é rápido para F1 de tupla).
- O notebook 06 compara **o melhor contra todos**, com o "melhor" escolhido no próprio teste. Isso é seleção pós-hoc. Defina a família de hipóteses **a priori** (seção 4.3).

### 2.5 `mcnemar` sobre a avaliação inteira **[M]**

- Ele mede **acerto exato dos 5 aspectos** (subset accuracy: 64,9% no LogReg), que é uma métrica diferente da principal. Testes e métricas devem casar ([Dror et al., 2018][dror18]). Um modelo pode melhorar o F1 macro (classes raras) sem mudar o acerto exato, e o contrário também acontece.
- Ele é válido como teste da hipótese "as taxas de acerto exato são iguais" ([Dietterich, 1998][dietterich98]) e pode ficar como análise secundária. Também vale rodar o McNemar **por aspecto** (5 testes, com Holm). Verifiquei LogReg × RF: produto p = 0,004; entrega p = 0,047; recebimento p = 0,47; anúncio p = 0,047; outros p = 0,13. A conclusão é mais informativa que o p = 0,006 global.
- O poder é limitado (seção 4.4).

### 2.6 `tuple_f1` **[ok]**

O cálculo está correto e é a métrica do campo: polaridade errada conta como 1 FP + 1 FN, como no F1 de tuplas do SemEval e no script do GermEval. O erro-padrão é baixo (0,0065 contra 0,014 do macro). **Recomendo promovê-la a métrica principal** (seção 3).

### 2.7 `delivery_validation` **[M]**

Devolve só a AUC pontual, sem IC e sem comparação com referências triviais. Os problemas de interpretação estão na seção 5.

### 2.8 Notebooks

| notebook | problema | sev. |
|---|---|---|
| 06 | O "melhor" é escolhido pelo próprio teste e depois comparado com os demais; não há correção de múltiplas comparações; uma semente por encoder | A |
| 06 | A comparação humano × LLM usa `macro_f1` com o problema 2.1, justamente na pergunta central | A |
| 07 | Regressão "nota ≤ 2 ~ aspectos negativos" usa `LogisticRegression()` com penalização L2 padrão (C = 1), o que encolhe as razões de chance, e sem IC; os preditores são **rótulos previstos** (erro de medida) | M |
| 07 | AUC sem IC e sem referência (nota, placebo) | M |
| 07 | `olist.entregue` exclui não entregues (ok), mas há textos duplicados (36.155 distintos em 40.950): reamostrar por pedido ou texto | B |
| 08 | `pred` vem de `clf.predict` (argmax). Sem `labels=` fixo por grupo, o F1 macro de um grupo pequeno (Norte, n = 26, sem nenhum "neutro" em anúncio) usa outro conjunto de classes que o do grupo grande: **não comparável** | A |
| 08 | `idade`: no pandas 3, `pd.cut(...).astype(str)` **mantém NaN** (verificado), então os 83 avaliadores sem ano de nascimento ficam sem grupo, ao contrário de gênero e região, que viram "NI" | B |
| 08 | LIME com `num_samples=500` e 3 exemplos: sem medida de estabilidade nem de fidelidade | M |

---

## 3. Métrica principal

### 3.1 O que a literatura usa

- **SemEval-2016 Task 5** ([Pontiki et al., 2016][semeval16]): categoria (Slot 1) com F1 **micro**. Polaridade (Slot 3) com **acurácia sobre as categorias-ouro**, numa fase separada. Os pares {categoria, alvo} também são avaliados com F1 micro.
- **Formulação idêntica à deste projeto** (cada aspecto ∈ {N/A, pos, neg, neu}): Schmitt et al. (2018) a usam no GermEval-2017 e reportam "F1 micro para categoria e para categoria+polaridade, calculados pelo script oficial do GermEval" ([Schmitt et al., 2018][schmitt18]). A classe N/A **não** entra como classe a ser pontuada: só pares mencionados contam.
- **Survey:** Zhang et al. (2022/2023) classificam ACSA como tarefa composta de pares (categoria, polaridade) ([Zhang et al.][zhang22]). [não verificado: o survey não fixa uma métrica única; nas tarefas compostas, a literatura citada usa majoritariamente F1 micro de tuplas.]
- **RePro:** o artigo original avalia tópico e sentimento **separadamente**, com P/R/F1 por classe e uma média, em um split 70/30 próprio ([Silva et al., 2024][repro]). A polaridade é **da avaliação inteira**, não do tópico, como o próprio `repro.py` documenta, e a concordância entre anotadores foi κ = 0,68 (tópico) e 0,71 (sentimento). Portanto **não há número publicado comparável** ao ACSA deste projeto, e o relatório deve dizer isso. Os κ também dão um teto razoável: diferenças entre modelos menores que o ruído de anotação devem ser interpretadas com cautela.

### 3.2 "Ausente" deve entrar na média?

**Não na métrica principal.** "Ausente" é a classe majoritária em 4 de 5 aspectos (até 92% em anúncio) e tem F1 alto e fácil (0,92–0,97 no LogReg). Incluí-la no macro **infla o número e comprime as diferenças**: no LogReg, o macro com ausente dá 0,759 e sem ausente (só polaridades do gabarito) dá 0,700; no RF, 0,612 e 0,504. A detecção do aspecto já é medida pelo F1 de tupla (um FN de detecção é um FN de par) e, de forma explícita, pelo `f1_deteccao`.

### 3.3 Recomendação

| papel | métrica | justificativa |
|---|---|---|
| **Principal** | **F1 micro de tuplas (aspecto, polaridade)**, já implementado em `tuple_f1` | Padrão de SemEval/GermEval para esta formulação; estável (EP ≈ 0,0065); sem dependência de classes raras; comparável com a literatura |
| Secundária 1 | **F1 macro com classes fixas pelo gabarito**, por aspecto e média (decisão a declarar: com ou sem "ausente"; recomendo **sem**) | Dá peso às classes minoritárias (negativo em anúncio, neutro); captura o que a tupla micro dilui |
| Secundária 2 | F1 de detecção por aspecto + F1 macro de polaridade condicional (corrigido) | Separa "achou o aspecto?" de "acertou a polaridade?", que é o diagnóstico útil para o humano × LLM |
| Diagnóstico | F1 de "neutro" por aspecto **com o suporte ao lado** (26/21/0/8/51) e IC | Mostrar sem tirar conclusão: com 8 casos, o IC do F1 vai de ~0 a ~0,7 |
| Opcional | Acerto exato da avaliação (subset accuracy) | Interpretação de negócio: "% de comentários 100% corretos" |

**Cuidado com o "jardim dos caminhos que se bifurcam":** o README já anuncia o F1 macro como principal. Trocar a métrica **depois** de ver os encoders parece ajuste de resultado. Justifique a troca pelo defeito técnico (2.1), que independe de qual modelo vence, e **reporte as duas** na tabela final.

---

## 4. Testes estatísticos e sementes

### 4.1 Unidade de reamostragem e testes

- **IC de cada modelo:** bootstrap percentil por avaliação, 1.000 a 2.000 reamostras. O BCa é opcional (2.3).
- **Comparação entre dois sistemas na mesma amostra:** **bootstrap pareado** ([Koehn, 2004][koehn04]; [Berg-Kirkpatrick et al., 2012][bk12]) ou **teste de aleatorização aproximada** (troca, por avaliação, das predições de A e B). O teste de aleatorização é o recomendado por Yeh (2000) para métricas não decomponíveis como o F ([Yeh, 2000][yeh00]) e aparece no protocolo de Dror et al. (2018) para métricas não lineares ([Dror et al., 2018][dror18]). Implementado e testado: LogReg × RF no macro fixo deu p ≈ 0,002 (500 permutações, 30 s).
- **McNemar:** só para métricas de acerto binário por item (acerto exato ou acerto por aspecto), como análise secundária.

### 4.2 Variância entre sementes: obrigatória para os encoders

- Mudar só a semente desloca o F1 em ~1 ponto, a ponto de mudar a conclusão sobre o estado da arte ([Reimers; Gurevych, 2017][rg17]). No fine-tuning do BERT, inicialização da cabeça e ordem dos dados contribuem igualmente para a variância, e algumas execuções divergem ([Dodge et al., 2020][dodge20]). A variância de inicialização e de dados afeta comparações de benchmarks ([Bouthillier et al., 2021][bouthillier21]).
- **Protocolo recomendado para a T4:** **5 sementes** por configuração (BERTimbau e Albertina × rótulo humano e LLM = 4 configurações = 20 execuções). Com 5.589 textos, `max_len` = 128, batch 32 e fp16, estimo ~2–4 min por época na T4, o que dá ~10–15 min por execução e **~4–5 h no total**, em 2–3 sessões do Colab gratuito. [Estimativa minha, não verificada: meça a primeira execução e ajuste.] Se não couber, use 3 sementes no mínimo e declare a limitação.
- **Como reportar:** média ± desvio-padrão entre sementes, e também a semente mediana. Para comparar duas configurações com sementes, há duas opções:
  (a) **Almost Stochastic Order** ([Dror; Shlomov; Reichart, 2019][dror19]), implementado no pacote `deepsig` ([Ulmer et al., 2022][ulmer22]). Reporte ε_min; os autores do pacote sugerem o limiar < 0,2. Com só 5 sementes, porém, o ASO tem pouco poder.
  (b) Mais simples: **bootstrap hierárquico**, que reamostra as sementes e, dentro de cada uma, as avaliações. Isso dá um IC que combina a variância do treino e a do teste.
  **Recomendação:** (b) como teste principal e (a) como checagem, citando ambos.
- Para LLM few-shot com temperatura 0 e TF-IDF, uma execução basta. Declare o determinismo.

### 4.3 Múltiplas comparações

Defina a família **antes** de rodar. Sugestão (6 hipóteses):
H1–H2: BERTimbau-humano > TF-IDF-humano e Albertina-humano > TF-IDF-humano;
H3–H4: humano × LLM para BERTimbau e Albertina (bilateral);
H5: melhor encoder × LLM few-shot direto;
H6: BERTimbau × Albertina (bilateral).
Aplique **Holm-Bonferroni** ([Holm, 1979][holm79]), que controla o FWER e é uniformemente mais poderoso que Bonferroni. As comparações por aspecto (5 × k) ficam como exploratórias, sem afirmação de significância, ou com Holm separado. Em Python: `statsmodels.stats.multitest.multipletests(p, method="holm")`, ou 6 linhas de código à mão.

### 4.4 Poder estatístico com n = 1.597

- Card et al. (2020) mostram que o poder do McNemar depende de n, da diferença de acurácia (Δ) e da **taxa de concordância entre modelos (Pa)**, e que comparações subdimensionadas são comuns em PLN ([Card et al., 2020][card20]). Simulei com os parâmetros deles (2.000 réplicas, α = 0,05, n = 1.597):

| Pa \ Δ | 1 p.p. | 2 p.p. | 3 p.p. |
|---|---|---|---|
| 0,80 | 13% | 41% | 76% |
| 0,90 | 23% | 69% | 96% |

- No bootstrap: o EP da diferença pareada LogReg − RF foi 0,016 no macro fixo e 0,0078 no F1 de tupla. Uma regra prática para efeito mínimo detectável com 80% de poder e α = 0,05 bilateral é ≈ 2,8 × EP. Isso dá **≈ 2,2 p.p. no F1 de tupla** e **≈ 4,5 p.p. no macro**. [A regra 2,8·EP supõe normalidade; é aproximação.]
- **Implicação para o relatório:** a diferença humano × LLM esperada é pequena, e se ficar abaixo de ~2 p.p. ela **não será detectável**. Escreva "não detectamos diferença; o IC exclui diferenças maiores que X", e **não** "são equivalentes". Para afirmar equivalência, defina antes uma margem (ex.: 2 p.p.) e verifique se o IC95% pareado cabe dentro dela (lógica TOST).
- Søgaard et al. (2014) mostram que p-valores em PLN são sensíveis à escolha da amostra e da métrica ([Søgaard et al., 2014][sogaard14]) [conteúdo detalhado não verificado; cite-o apenas como alerta geral]. Isso reforça a recomendação de reportar efeitos com IC, e não só p.

### 4.5 Calibração (ECE): vale a pena?

**Sim, como item curto, porque o projeto usa probabilidades**: a AUC do notebook 07 e qualquer limiar operacional na Olist dependem delas. Guo et al. (2017) definem o ECE (bins de confiança) e mostram que redes modernas tendem a ser mal calibradas, com **temperature scaling** como correção simples ([Guo et al., 2017][guo17]). Para BERT e RoBERTa, Desai e Durrett (2020) encontram calibração razoável no domínio e ainda melhor com temperatura ([Desai; Durrett, 2020][desai20]). O efeito da mudança de domínio RePro → Olist é justamente o ponto interessante. Implementação: ECE por aspecto (15 bins) no teste do RePro, antes e depois de ajustar uma temperatura no dev. Mais um diagrama de confiabilidade para `entrega`. **Custo: ~1 hora.** Não é prioridade se o tempo apertar; nesse caso, priorize a AUC (invariante a transformações monótonas, logo insensível à calibração).

---

## 5. Validação extrínseca com o atraso real

### 5.1 O que os números mostram (verificado com o TF-IDF + LogReg treinado no RePro, aplicado a 38.796 avaliações entregues, das quais 3.747 atrasadas)

| preditor do atraso | AUC |
|---|---|
| P(entrega negativa) | **0,841** |
| Só a nota (5 − nota) | 0,817 |
| P(produto negativo) (placebo) | 0,545 |
| Regressão logística: só nota (validação cruzada 5-fold) | 0,815 |
| Regressão logística: nota + P(entrega negativa) (VC 5-fold) | **0,862** |

AUC de P(entrega negativa) **dentro de cada nota**: 1★ 0,744 · 2★ 0,740 · 3★ 0,741 · 4★ 0,711 · 5★ 0,644.

### 5.2 Interpretação

- **A AUC crua sozinha é fraca como evidência.** Quase todo modelo de sentimento geral atinge ~0,82 só por capturar insatisfação, porque a nota já tem AUC de 0,817. O argumento de que o modelo **identifica o aspecto** vem de três peças, todas obrigatórias no relatório:
  1. **Ganho incremental sobre a nota** (0,815 → 0,862, com IC por bootstrap da diferença);
  2. **AUC estratificada por nota**, que fica bem acima de 0,5 mesmo entre clientes igualmente insatisfeitos;
  3. **Controle negativo (placebo):** o aspecto produto quase não prevê atraso (0,545). Isso mostra **validade discriminante**: o sinal está no aspecto certo.
- **Confundidores e controles:** nota, região (o prazo e o atraso variam por UF), categoria e tamanho do texto (textos longos mencionam mais aspectos) e mês da compra (a greve dos caminhoneiros de maio de 2018 concentrou atrasos [não verificado nos dados; checar]). Regressão logística sugerida, **sem penalização**, com IC por erro-padrão robusto:
  `atrasado ~ p_entrega_neg + C(nota) + C(regiao) + C(categoria_top12) + log(n_palavras) + C(mes)`.
  **Não interprete os coeficientes dos controles** (falácia da "Tabela 2": coeficientes de covariáveis não são efeitos totais; [Westreich; Greenland, 2013][westreich13]). E evite linguagem causal: o atraso causa o comentário, e não o contrário. O que se valida é a **validade de construto** da predição.
- **Dose-resposta:** a tabela de faixas de atraso do notebook 07 já está boa. Acrescente: a % de "entrega negativa" deve crescer monotonicamente com os dias de atraso e ser baixa em entregas antecipadas. Plote com IC de Wilson por faixa.
- **Defina o desfecho de forma robusta:** `atrasado` (> 0 dias) é a definição principal; faça uma análise de sensibilidade com > 3 e > 7 dias.
- **Unidade e dependência:** reamostre por pedido. Há textos repetidos ("ótimo", "recomendo"); o IC da AUC deve vir de bootstrap (2.000 réplicas) ou do método de DeLong.
- **Regressão "nota baixa ~ aspectos negativos previstos" (notebook 07):** os preditores são rótulos estimados com erro. Mesmo com 80–90% de acurácia, isso enviesa coeficientes e ICs a jusante ([Egami et al., 2023][egami23]). Com a amostra anotada da Olist (100 comentários), dá para estimar a taxa de erro e declarar a limitação. Se a amostra crescer, o estimador DSL do mesmo artigo é a correção formal. Trate essa regressão como **descritiva**.
- **Mudança de domínio:** RePro (Americanas, 2018) → Olist. Os 100 comentários anotados da Olist dão um F1 com IC enorme (~±0,08). Reporte o IC e trate o número como indicativo.

---

## 6. XAI

### 6.1 Escolha do método

| método | prós | contras | uso no projeto |
|---|---|---|---|
| **LIME** ([Ribeiro et al., 2016][lime16]) | Agnóstico; já implementado; serve também para o TF-IDF | Amostragem aleatória: explicações instáveis entre execuções; com 500 amostras e textos curtos, a variância é alta | Manter, mas medir estabilidade (2–3 `random_state`) |
| **SHAP Partition** com masker de texto ([Lundberg; Lee, 2017][shap17]; [docs SHAP][shap-docs]) | Valores de Shapley com hierarquia de tokens; aceita função customizada + tokenizer, o que serve para o modelo multicabeça | Mais caro; a documentação marca o Partition como "alpha" no benchmark de texto | Boa alternativa ao LIME, com a mesma `proba_aspecto` |
| **Integrated Gradients** ([Sundararajan et al., 2017][ig17]) via Captum `LayerIntegratedGradients` na camada de embeddings ([tutorial Captum][captum]) | Axiomas (sensibilidade, invariância de implementação); determinístico; barato (20–50 passos) | Depende do baseline (o tutorial usa [PAD] mantendo [CLS]/[SEP]); exige acesso ao modelo | **Recomendado** para o encoder: o `MultiAspectModel` é PyTorch puro |
| Atenção | Grátis | **Não use como explicação** ([Jain; Wallace, 2019][jain19]; o contraponto de [Wiegreffe; Pinter, 2019][wiegreffe19] mostra que depende da definição e exige testes adicionais) | Não usar |

Atanasova et al. (2020) compararam explicadores em classificação de texto e concluíram que **métodos baseados em gradiente** foram os melhores no conjunto de propriedades diagnósticas ([Atanasova et al., 2020][atanasova20]).

### 6.2 Avaliação de fidelidade (o que torna a XAI convincente)

- **Fidelidade ≠ plausibilidade** ([Jacovi; Goldberg, 2020][jacovi20]). Uma explicação que "parece certa" não prova que o modelo usa aquelas palavras.
- **ERASER** ([DeYoung et al., 2020][eraser20]):
  - *Comprehensiveness:* quanto P(classe) cai ao **remover** os k tokens mais importantes;
  - *Sufficiency:* quanto P(classe) cai ao manter **só** esses tokens;
  - AOPC: média das duas sobre k ∈ {1, 2, 3, 5, 10%…}.
- **ferret** ([Attanasio et al., 2023][ferret23]) implementa Gradient, IG, SHAP (Partition) e LIME, além de AOPC-comprehensiveness, AOPC-sufficiency e correlação de Kendall com leave-one-out. A integração direta é com `AutoModelForSequenceClassification`. Como o modelo do projeto tem 5 cabeças, o caminho prático é **implementar comprehensiveness/sufficiency à mão** (≈ 30 linhas, usando `predict_proba` com tokens removidos) ou embrulhar uma cabeça como modelo HF. [Suporte do ferret a modelos customizados não verificado.]

### 6.3 Protocolo viável (≈ 1 dia de trabalho, T4)

1. Escolha 100 avaliações do teste com `entrega` = negativo previsto e correto, mais 30 com erro.
2. Gere explicações com IG (Captum) e LIME para a classe "entrega/negativo".
3. Calcule comprehensiveness e sufficiency (top-k = 1, 3, 5 tokens) para IG, LIME e um **explicador aleatório** (baseline obrigatório) e reporte médias com IC.
4. Calcule a estabilidade do LIME: Jaccard dos top-5 entre 3 sementes.
5. Faça a análise qualitativa: 6 exemplos (3 acertos e 3 erros), incluindo um caso de **atalho** (ex.: "não" + "chegou" dispara "entrega negativa" mesmo quando o texto fala do produto).
6. Opcional e de alto valor: compare a explicação do **modelo com rótulos humanos com a do modelo com rótulos de LLM** no mesmo texto (sobreposição de top-k). Isso conecta a XAI à pergunta central.

---

## 7. Fairness com grupos pequenos

### 7.1 Tamanhos no teste (verificados)

- Gênero: M 805, F 729, NI 63.
- Região: Sudeste 1.022, Sul 220, Nordeste 169, Centro-Oeste 100, NI 60, **Norte 26**.
- Idade: 30–45 680, 45–60 368, <30 340, **60+ 126**, ausente 83.

No Norte, `anúncio` tem 3 negativos e 0 positivos ou neutros; o F1 macro por grupo nesse aspecto é essencialmente ruído.

### 7.2 Recomendações

1. **Métrica por grupo:** use a métrica principal (**F1 de tupla**, que agrega os 5 aspectos e tem mais suporte) e o macro com **`labels` fixos** iguais aos do teste global. Nunca use o macro "classes presentes" por grupo (2.1). O Fairlearn aceita qualquer `fn(y_true, y_pred)` ([MetricFrame][fl-mf]). As métricas prontas de paridade (`demographic_parity_difference` etc.) são para classificação binária ([guia Fairlearn][fl-common]) e não servem aqui sem binarizar. Uma binarização que faz sentido: "aspecto negativo detectado corretamente" (recall da classe negativa por grupo, isto é, igualdade de oportunidade).
2. **IC por grupo:** `MetricFrame(..., n_boot=1000, ci_quantiles=[0.025, 0.5, 0.975], random_state=0)` dá `by_group_ci` e `difference_ci()` ([Fairlearn: intervalos de confiança][fl-ci]). Para o F1 de tupla, que não é escalar sobre um vetor `y`, passe índices como `y_true` e feche a métrica sobre os DataFrames, ou faça o bootstrap à mão estratificado por grupo.
3. **Diferença máxima (`difference(method="between_groups")`) é enviesada para cima com grupos pequenos.** O máximo de estimativas ruidosas cresce com o ruído, e lacunas "significativas" podem sumir após correção ([Lum; Zhang; Bower, 2022][lum22]). Por isso:
   - reporte a diferença **entre grupos com n ≥ ~100** e trate Norte (26) e NI como "dados insuficientes";
   - prefira `difference(method="to_overall")` com IC;
   - uma alternativa robusta é agregar Norte com Nordeste ("N/NE"), declarando a decisão antes de ver o resultado.
4. **"NI"/ausentes:** mantenha como grupo explícito (não descarte; pode haver viés de quem não informa) e corrija a idade para `fillna("NI")` (2.8). Não tire conclusões sobre NI.
5. **Interpretação:** sem correção para múltiplos grupos e aspectos, alguma diferença "aparecerá" por acaso. A frase adequada é "nenhuma diferença entre grupos excedeu a incerteza amostral; o poder para detectar lacunas < X p.p. em grupos de ~100 é baixo". Também vale mencionar que o RePro tem demografia **autodeclarada do avaliador**, não do texto. Na Olist, região é a do **cliente** (`customer_state`).

---

## 8. Plano de mudanças priorizado

### P0: obrigatório antes de rodar os encoders (≈ 1 dia)

**(1) Conjunto de classes fixo em `metrics.py`** (corrige 2.1 e 2.2 e afeta a seleção de época do `encoder.py`):

```python
POLS = ["positivo", "negativo", "neutro"]

def labels_from_gold(y_true, include_absent=True):
    """Classes com suporte no gabarito — calcular UMA vez no teste inteiro e reutilizar."""
    out = {}
    for c in LABEL_COLUMNS:
        present = [l for l in POLARITIES if (y_true[c] == l).any()]
        out[c] = present if include_absent else [l for l in present if l != "ausente"]
    return out

def macro_f1(y_true, y_pred, labels=None, include_absent=False):
    labels = labels or labels_from_gold(y_true, include_absent)
    return float(np.mean([f1_score(y_true[c], y_pred[c], labels=labels[c],
                                   average="macro", zero_division=0)
                          for c in LABEL_COLUMNS]))

def _polarity_f1(t, p):
    mask = t != "ausente"
    labs = [l for l in POLS if (t[mask] == l).any()]
    return f1_score(t[mask], p[mask], labels=labs, average="macro", zero_division=0) if labs else np.nan
```

Em `bootstrap_ci` e `paired_bootstrap`, calcule `labels = labels_from_gold(y_true)` **antes** do laço e passe a cada chamada (via `functools.partial`). No `encoder.fit`, selecione a época pelo **F1 de tupla no dev** (ou pelo macro com labels fixos do dev).

**(2) Promover `tuple_f1` a métrica principal** na tabela do 06 (ordenação, IC e testes), mantendo o macro fixo como segunda coluna. Atualize o README explicando o motivo (2.1).

**(3) Família de hipóteses e Holm no 06:** substitua "melhor contra todos" pela lista 4.3 e acrescente:

```python
def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); adj[k] = run
    return adj
```

**(4) p-valor bilateral e 10.000 reamostras em `paired_bootstrap`:**

```python
p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
return {..., "p_valor": min(1.0, p_two), "ep": float(diffs.std(ddof=1))}
```

### P1: alto valor (≈ 2–3 dias, incluindo GPU)

**(5) 5 sementes por encoder** (4.2). No 05, envolva o treino num laço `for seed in [13, 42, 87, 2024, 31337]` e salve `pred_repro_teste_{modelo}_s{seed}.csv`. No 06, agregue média ± DP e faça o bootstrap hierárquico:

```python
def hier_bootstrap_diff(y, preds_a: list, preds_b: list, metric, n=2000, rng=...):
    out = []
    for _ in range(n):
        sa, sb = rng.integers(len(preds_a)), rng.integers(len(preds_b))   # reamostra sementes
        idx = rng.integers(0, len(y), len(y))                              # reamostra itens (pareado)
        out.append(metric(y.iloc[idx], preds_a[sa].iloc[idx]) - metric(y.iloc[idx], preds_b[sb].iloc[idx]))
    return np.mean(out), np.quantile(out, [.025, .975])
```

Opcional: `deepsig.aso(scores_a, scores_b)` para ε_min.

**(6) Validação extrínseca reforçada (07):** IC da AUC por bootstrap; AUC da nota e do placebo (produto); AUC por estrato de nota; ΔAUC (nota → nota + p_entrega) com IC; regressão logística sem penalização (`statsmodels.Logit` ou `LogisticRegression(C=np.inf)`) com controles; sensibilidade para atraso > 3 e > 7 dias; dose-resposta com IC de Wilson. Esboço:

```python
def delivery_validation(atrasado, p, nota=None, placebo=None, n_boot=2000, seed=SEED):
    res = {"auc": roc_auc_score(atrasado, p), "auc_ic": boot_auc(atrasado, p, n_boot, seed)}
    if nota is not None:
        res["auc_nota"] = roc_auc_score(atrasado, -np.asarray(nota))
        res["auc_por_nota"] = {k: roc_auc_score(atrasado[nota == k], p[nota == k]) for k in np.unique(nota)
                               if len(np.unique(atrasado[nota == k])) == 2}
    if placebo is not None:
        res["auc_placebo"] = roc_auc_score(atrasado, placebo)
    return res
```

**(7) Fairness (08):** métrica de tupla + macro com labels fixos por grupo, `n_boot`/`ci_quantiles` no MetricFrame, `difference(method="to_overall")` com IC, idade com `fillna("NI")`, sinalizar grupos com n < 100.

### P2: completa o trabalho (≈ 1–2 dias)

**(8) XAI com fidelidade (6.3):** IG via Captum + LIME, com comprehensiveness/sufficiency contra um explicador aleatório e estabilidade do LIME.

**(9) McNemar por aspecto com Holm** (secundário) e acerto exato reportado como métrica de negócio.

**(10) Calibração:** ECE por aspecto + temperature scaling no dev (4.5).

**(11) Testes unitários em `tests/test_core.py`:** (a) uma predição espúria de classe ausente no gabarito não muda `macro_f1` em mais de 1e-3; (b) `_polarity_f1` não depende de FNs de detecção; (c) `tuple_f1` dá 1 para predição perfeita e 0 para tudo "ausente".

---

## 9. Checklist de reporte para o relatório final

**Dados e protocolo**
- [ ] Origem e conversão do gabarito: polaridade do RePro é **por avaliação**, projetada nos aspectos; descartes por polaridade mista (com contagem) e o viés que isso induz (o teste exclui os casos mistos, que são os mais difíceis).
- [ ] Split (estratificado pela nota, semente), tamanhos e tabela de suporte por aspecto × classe (incluindo os zeros).
- [ ] Concordância humana do RePro (κ 0,68/0,71) como referência de teto.
- [ ] Hiperparâmetros, critério de seleção de época (no dev, nunca no teste), número de sementes e tempo de GPU.

**Métricas**
- [ ] Definição exata da métrica principal (F1 micro de tuplas) e das secundárias (macro com `labels` fixos, com ou sem "ausente", declarado).
- [ ] Justificativa da troca de métrica principal (defeito 2.1), com as duas colunas na tabela.
- [ ] Por aspecto: detecção F1, polaridade F1 (corrigida), suporte.
- [ ] "Neutro": F1 por aspecto com suporte e aviso de baixa confiabilidade.

**Incerteza e testes**
- [ ] IC95% bootstrap (método, nº de reamostras, unidade = avaliação) para toda métrica reportada.
- [ ] Encoders: média ± DP entre sementes (n de sementes) e IC hierárquico.
- [ ] Família de hipóteses declarada a priori; teste usado (bootstrap pareado ou aleatorização); p bilateral; correção de Holm; p ajustados.
- [ ] Tamanho de efeito com IC, não só p; frase de poder ("diferenças < ~2 p.p. de F1 de tupla não são detectáveis com n = 1.597").
- [ ] Nada de "equivalente" sem margem pré-definida.

**Validação extrínseca**
- [ ] AUC com IC; AUC da nota; ΔAUC incremental; AUC por nota; placebo; dose-resposta; sensibilidade da definição de atraso.
- [ ] Linguagem de validade de construto, não causal; limitações (duplicatas, mudança de domínio, rótulos previstos como covariáveis).

**XAI**
- [ ] Método, parâmetros (amostras do LIME, passos e baseline do IG), nº de exemplos e critério de seleção.
- [ ] Fidelidade (comprehensiveness/sufficiency) contra baseline aleatório; estabilidade.
- [ ] Exemplos qualitativos, incluindo erros, sem cherry-picking (sorteio com semente).

**Fairness**
- [ ] n por grupo; métrica por grupo com `labels` fixos; IC por grupo; diferença com IC; grupos pequenos sinalizados; tratamento de NI.
- [ ] Atributos autodeclarados e limitações (RePro: avaliador; Olist: região do cliente).

**Reprodutibilidade**
- [ ] Commit, versões (`scikit-learn`, `transformers`, `fairlearn`), sementes, predições salvas e notebooks executados de ponta a ponta.

---

## 10. Referências (ABNT)

ATANASOVA, P.; SIMONSEN, J. G.; LIOMA, C.; AUGENSTEIN, I. A diagnostic study of explainability techniques for text classification. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING (EMNLP), 2020. **Proceedings** [...]. [S. l.]: ACL, 2020. p. 3256–3274. DOI: 10.18653/v1/2020.emnlp-main.263. Disponível em: https://aclanthology.org/2020.emnlp-main.263. Acesso em: 30 out. 2026.

ATTANASIO, G.; PASTOR, E.; DI BONAVENTURA, C.; NOZZA, D. ferret: a framework for benchmarking explainers on transformers. In: CONFERENCE OF THE EUROPEAN CHAPTER OF THE ACL: SYSTEM DEMONSTRATIONS (EACL), 2023. **Proceedings** [...]. Dubrovnik: ACL, 2023. p. 256–266. DOI: 10.18653/v1/2023.eacl-demo.29. Disponível em: https://aclanthology.org/2023.eacl-demo.29. Acesso em: 30 out. 2026.

BERG-KIRKPATRICK, T.; BURKETT, D.; KLEIN, D. An empirical investigation of statistical significance in NLP. In: EMNLP-CoNLL, 2012. **Proceedings** [...]. Jeju Island: ACL, 2012. p. 995–1005. Disponível em: https://aclanthology.org/D12-1091. Acesso em: 30 out. 2026.

BOUTHILLIER, X. et al. Accounting for variance in machine learning benchmarks. In: **Proceedings of Machine Learning and Systems (MLSys)**, v. 3, 2021. Disponível em: https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html. Acesso em: 30 out. 2026.

CARD, D.; HENDERSON, P.; KHANDELWAL, U.; JIA, R.; MAHOWALD, K.; JURAFSKY, D. With little power comes great responsibility. In: EMNLP, 2020. **Proceedings** [...]. [S. l.]: ACL, 2020. p. 9263–9274. DOI: 10.18653/v1/2020.emnlp-main.745. Disponível em: https://aclanthology.org/2020.emnlp-main.745. Acesso em: 30 out. 2026.

DESAI, S.; DURRETT, G. Calibration of pre-trained transformers. In: EMNLP, 2020. **Proceedings** [...]. [S. l.]: ACL, 2020. p. 295–302. DOI: 10.18653/v1/2020.emnlp-main.21. Disponível em: https://aclanthology.org/2020.emnlp-main.21. Acesso em: 30 out. 2026.

DEYOUNG, J. et al. ERASER: a benchmark to evaluate rationalized NLP models. In: ANNUAL MEETING OF THE ACL, 58., 2020. **Proceedings** [...]. [S. l.]: ACL, 2020. p. 4443–4458. DOI: 10.18653/v1/2020.acl-main.408. Disponível em: https://aclanthology.org/2020.acl-main.408. Acesso em: 30 out. 2026.

DIETTERICH, T. G. Approximate statistical tests for comparing supervised classification learning algorithms. **Neural Computation**, v. 10, n. 7, p. 1895–1923, 1998. DOI: 10.1162/089976698300017197. Disponível em: https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf. Acesso em: 30 out. 2026.

DODGE, J. et al. **Fine-tuning pretrained language models**: weight initializations, data orders, and early stopping. arXiv:2002.06305, 2020. Disponível em: https://arxiv.org/abs/2002.06305. Acesso em: 30 out. 2026.

DROR, R.; BAUMER, G.; SHLOMOV, S.; REICHART, R. The hitchhiker's guide to testing statistical significance in natural language processing. In: ANNUAL MEETING OF THE ACL, 56., 2018. **Proceedings** [...]. Melbourne: ACL, 2018. p. 1383–1392. DOI: 10.18653/v1/P18-1128. Disponível em: https://aclanthology.org/P18-1128. Acesso em: 30 out. 2026.

DROR, R.; SHLOMOV, S.; REICHART, R. Deep dominance: how to properly compare deep neural models. In: ANNUAL MEETING OF THE ACL, 57., 2019. **Proceedings** [...]. Florence: ACL, 2019. p. 2773–2785. DOI: 10.18653/v1/P19-1266. Disponível em: https://aclanthology.org/P19-1266. Acesso em: 30 out. 2026.

EFRON, B. Better bootstrap confidence intervals. **Journal of the American Statistical Association**, v. 82, n. 397, p. 171–185, 1987. DOI: 10.1080/01621459.1987.10478410 [DOI não verificado].

EGAMI, N.; HINCK, M.; STEWART, B. M.; WEI, H. Using imperfect surrogates for downstream inference: design-based supervised learning for social science applications of large language models. In: **Advances in Neural Information Processing Systems 36 (NeurIPS 2023)**. Disponível em: https://proceedings.neurips.cc/paper_files/paper/2023/hash/d862f7f5445255090de13b825b880d59-Abstract.html. Acesso em: 30 out. 2026.

FAIRLEARN. **Confidence interval estimation**; **MetricFrame**; **Common fairness metrics**. Documentação oficial. Disponível em: https://fairlearn.org/main/user_guide/assessment/confidence_interval_estimation.html; https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html; https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html. Acesso em: 30 out. 2026.

FORMAN, G.; SCHOLZ, M. Apples-to-apples in cross-validation studies: pitfalls in classifier performance measurement. **ACM SIGKDD Explorations Newsletter**, v. 12, n. 1, p. 49–57, 2010. DOI: 10.1145/1882471.1882479. Disponível em: https://www.dcc.fc.up.pt/~ines/aulas/1516/DM1/Papers/v12-1-p49-forman-sigkdd.pdf. Acesso em: 30 out. 2026.

GUO, C.; PLEISS, G.; SUN, Y.; WEINBERGER, K. Q. On calibration of modern neural networks. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING (ICML), 34., 2017. **Proceedings** [...]. PMLR, v. 70, p. 1321–1330, 2017. Disponível em: https://proceedings.mlr.press/v70/guo17a.html. Acesso em: 30 out. 2026.

HOLM, S. A simple sequentially rejective multiple test procedure. **Scandinavian Journal of Statistics**, v. 6, n. 2, p. 65–70, 1979. Disponível em: https://www.jstor.org/stable/4615733. Acesso em: 30 out. 2026.

JACOVI, A.; GOLDBERG, Y. Towards faithfully interpretable NLP systems: how should we define and evaluate faithfulness? In: ANNUAL MEETING OF THE ACL, 58., 2020. **Proceedings** [...]. [S. l.]: ACL, 2020. p. 4198–4205. DOI: 10.18653/v1/2020.acl-main.386. Disponível em: https://aclanthology.org/2020.acl-main.386. Acesso em: 30 out. 2026.

JAIN, S.; WALLACE, B. C. Attention is not explanation. In: NAACL-HLT, 2019. **Proceedings** [...]. Minneapolis: ACL, 2019. p. 3543–3556. DOI: 10.18653/v1/N19-1357. Disponível em: https://aclanthology.org/N19-1357. Acesso em: 30 out. 2026.

KOEHN, P. Statistical significance tests for machine translation evaluation. In: EMNLP, 2004. **Proceedings** [...]. Barcelona: ACL, 2004. p. 388–395. Disponível em: https://aclanthology.org/W04-3250. Acesso em: 30 out. 2026.

LUM, K.; ZHANG, Y.; BOWER, A. De-biasing "bias" measurement. In: ACM CONFERENCE ON FAIRNESS, ACCOUNTABILITY, AND TRANSPARENCY (FAccT), 2022. **Proceedings** [...]. Seoul: ACM, 2022. p. 379–389. Disponível em: https://arxiv.org/abs/2205.05770. Acesso em: 30 out. 2026.

LUNDBERG, S. M.; LEE, S.-I. A unified approach to interpreting model predictions. In: **Advances in Neural Information Processing Systems 30 (NIPS 2017)**. p. 4765–4774. Disponível em: https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions. Acesso em: 30 out. 2026.

OPITZ, J.; BURST, S. **Macro F1 and macro F1**. arXiv:1911.03347, 2019. Disponível em: https://arxiv.org/abs/1911.03347. Acesso em: 30 out. 2026.

PONTIKI, M. et al. SemEval-2016 Task 5: aspect based sentiment analysis. In: INTERNATIONAL WORKSHOP ON SEMANTIC EVALUATION (SemEval-2016), 10., 2016. **Proceedings** [...]. San Diego: ACL, 2016. p. 19–30. DOI: 10.18653/v1/S16-1002. Disponível em: https://aclanthology.org/S16-1002. Acesso em: 30 out. 2026.

REIMERS, N.; GUREVYCH, I. Reporting score distributions makes a difference: performance study of LSTM-networks for sequence tagging. In: EMNLP, 2017. **Proceedings** [...]. Copenhagen: ACL, 2017. p. 338–348. DOI: 10.18653/v1/D17-1035. Disponível em: https://aclanthology.org/D17-1035. Acesso em: 30 out. 2026.

RIBEIRO, M. T.; SINGH, S.; GUESTRIN, C. "Why should I trust you?": explaining the predictions of any classifier. In: ACM SIGKDD INTERNATIONAL CONFERENCE ON KNOWLEDGE DISCOVERY AND DATA MINING, 22., 2016. **Proceedings** [...]. San Francisco: ACM, 2016. p. 1135–1144. DOI: 10.1145/2939672.2939778. Disponível em: https://arxiv.org/abs/1602.04938. Acesso em: 30 out. 2026.

SCHMITT, M.; STEINHEBER, S.; SCHREIBER, K.; ROTH, B. Joint aspect and polarity classification for aspect-based sentiment analysis with end-to-end neural networks. In: EMNLP, 2018. **Proceedings** [...]. Brussels: ACL, 2018. p. 1109–1114. DOI: 10.18653/v1/D18-1139. Disponível em: https://aclanthology.org/D18-1139. Acesso em: 30 out. 2026.

SCIKIT-LEARN. **sklearn.metrics.f1_score**. Documentação oficial. Disponível em: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html. Acesso em: 30 out. 2026.

SCIPY. **scipy.stats.bootstrap**. Documentação oficial. Disponível em: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html. Acesso em: 30 out. 2026.

SHAP. **Using custom functions and tokenizers**; **Emotion classification multiclass example**. Documentação oficial. Disponível em: https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/sentiment_analysis/Using%20custom%20functions%20and%20tokenizers.html. Acesso em: 30 out. 2026.

CAPTUM. **Interpreting BERT models (SQuAD)**. Tutorial oficial. Disponível em: https://captum.ai/tutorials/Bert_SQUAD_Interpret. Acesso em: 30 out. 2026.

SILVA, L. N. dos S. et al. RePro: a benchmark dataset for opinion mining in Brazilian Portuguese. In: INTERNATIONAL CONFERENCE ON COMPUTATIONAL PROCESSING OF PORTUGUESE (PROPOR), 16., 2024. **Proceedings** [...]. Santiago de Compostela: ACL, 2024. v. 1, p. 432–440. Disponível em: https://aclanthology.org/2024.propor-1.44. Acesso em: 30 out. 2026.

SØGAARD, A.; JOHANNSEN, A.; PLANK, B.; HOVY, D.; MARTÍNEZ ALONSO, H. What's in a p-value in NLP? In: CONFERENCE ON COMPUTATIONAL NATURAL LANGUAGE LEARNING (CoNLL), 18., 2014. **Proceedings** [...]. Ann Arbor: ACL, 2014. p. 1–10. DOI: 10.3115/v1/W14-1601. Disponível em: https://aclanthology.org/W14-1601. Acesso em: 30 out. 2026.

SUNDARARAJAN, M.; TALY, A.; YAN, Q. Axiomatic attribution for deep networks. In: ICML, 34., 2017. **Proceedings** [...]. PMLR, v. 70, p. 3319–3328, 2017. Disponível em: https://proceedings.mlr.press/v70/sundararajan17a.html. Acesso em: 30 out. 2026.

ULMER, D.; HARDMEIER, C.; FRELLSEN, J. **deep-significance**: easy and meaningful statistical significance testing in the age of neural networks. arXiv:2204.06815, 2022. Disponível em: https://arxiv.org/abs/2204.06815. Acesso em: 30 out. 2026. [autoria não verificada na página do arXiv]

WESTREICH, D.; GREENLAND, S. The Table 2 fallacy: presenting and interpreting confounder and modifier coefficients. **American Journal of Epidemiology**, v. 177, n. 4, p. 292–298, 2013. Disponível em: https://pmc.ncbi.nlm.nih.gov/articles/PMC3626058. Acesso em: 30 out. 2026.

WIEGREFFE, S.; PINTER, Y. Attention is not not explanation. In: EMNLP-IJCNLP, 2019. **Proceedings** [...]. Hong Kong: ACL, 2019. p. 11–20. DOI: 10.18653/v1/D19-1002. Disponível em: https://aclanthology.org/D19-1002. Acesso em: 30 out. 2026.

YEH, A. More accurate tests for the statistical significance of result differences. In: INTERNATIONAL CONFERENCE ON COMPUTATIONAL LINGUISTICS (COLING), 18., 2000. **Proceedings** [...]. Saarbrücken: ACL, 2000. v. 2, p. 947–953. Disponível em: https://aclanthology.org/C00-2137. Acesso em: 30 out. 2026.

ZHANG, W.; LI, X.; DENG, Y.; BING, L.; LAM, W. A survey on aspect-based sentiment analysis: tasks, methods, and challenges. **IEEE Transactions on Knowledge and Data Engineering**, v. 35, n. 11, p. 11019–11038, 2023. DOI: 10.1109/TKDE.2022.3230975. Disponível em: https://arxiv.org/abs/2203.01054. Acesso em: 30 out. 2026.

[sk-f1]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
[semeval16]: https://aclanthology.org/S16-1002
[schmitt18]: https://aclanthology.org/D18-1139
[zhang22]: https://arxiv.org/abs/2203.01054
[repro]: https://aclanthology.org/2024.propor-1.44
[efron87]: https://doi.org/10.1080/01621459.1987.10478410
[scipy-boot]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
[koehn04]: https://aclanthology.org/W04-3250
[bk12]: https://aclanthology.org/D12-1091
[yeh00]: https://aclanthology.org/C00-2137
[dror18]: https://aclanthology.org/P18-1128
[dror19]: https://aclanthology.org/P19-1266
[ulmer22]: https://arxiv.org/abs/2204.06815
[dietterich98]: https://web.engr.oregonstate.edu/~tgd/publications/dietterich-approximate-statistical-tests-nc1998.pdf
[rg17]: https://aclanthology.org/D17-1035
[dodge20]: https://arxiv.org/abs/2002.06305
[bouthillier21]: https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html
[holm79]: https://www.jstor.org/stable/4615733
[card20]: https://aclanthology.org/2020.emnlp-main.745
[sogaard14]: https://aclanthology.org/W14-1601
[guo17]: https://proceedings.mlr.press/v70/guo17a.html
[desai20]: https://aclanthology.org/2020.emnlp-main.21
[westreich13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3626058
[egami23]: https://proceedings.neurips.cc/paper_files/paper/2023/hash/d862f7f5445255090de13b825b880d59-Abstract.html
[lime16]: https://arxiv.org/abs/1602.04938
[shap17]: https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
[shap-docs]: https://shap.readthedocs.io/en/latest/example_notebooks/text_examples/sentiment_analysis/Using%20custom%20functions%20and%20tokenizers.html
[ig17]: https://proceedings.mlr.press/v70/sundararajan17a.html
[captum]: https://captum.ai/tutorials/Bert_SQUAD_Interpret
[jain19]: https://aclanthology.org/N19-1357
[wiegreffe19]: https://aclanthology.org/D19-1002
[atanasova20]: https://aclanthology.org/2020.emnlp-main.263
[jacovi20]: https://aclanthology.org/2020.acl-main.386
[eraser20]: https://aclanthology.org/2020.acl-main.408
[ferret23]: https://aclanthology.org/2023.eacl-demo.29
[fl-mf]: https://fairlearn.org/main/api_reference/generated/fairlearn.metrics.MetricFrame.html
[fl-common]: https://fairlearn.org/main/user_guide/assessment/common_fairness_metrics.html
[fl-ci]: https://fairlearn.org/main/user_guide/assessment/confidence_interval_estimation.html
[lum22]: https://arxiv.org/abs/2205.05770
