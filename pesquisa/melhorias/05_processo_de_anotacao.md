# Processo e ferramenta de anotação para T1, T2 e T3

*Pesquisa feita em 30/10/2026. Os números do RePro e da Olist foram calculados localmente com `src/absa/repro.py` (`convert`) sobre `data/raw/RePro.csv` e sobre `data/annotated/olist_para_anotar.csv`. Os links das afirmações factuais estão no texto. "[não verificado]" marca o que não consegui confirmar na fonte original durante esta pesquisa. "[estimativa própria]" marca contas ou estimativas minhas.*

---

## 1. Resumo executivo

1. **Ferramenta:** a melhor opção para este caso é uma **página HTML própria, de arquivo único e sem servidor**: interface sob medida para cada tarefa, atalhos mnemônicos (P/E/R/A/O + 1/2/3), salvamento automático no navegador e exportação de CSV já no formato `asp_*` do repositório. O plano B é o **Label Studio** (`pip install label-studio`), que é o melhor pronto para uso.
2. **Sem pré-anotação por LLM em T1 e T2.** Esses dados vão servir de teste para o próprio LLM do projeto, e sugestões de modelo puxam o anotador e inflam o desempenho medido ([Schroeder; Roy; Kabbara, 2025](https://aclanthology.org/2025.findings-acl.1323/); [Berzak et al., 2016](https://arxiv.org/abs/1605.04481)).
3. **T3 deve ser feita às cegas** (anotar do zero e só depois ver o rótulo convertido do RePro). Assim a auditoria também mede a concordância entre você e o anotador do RePro, e esse passa a ser o seu "segundo anotador" de graça.
4. **Controles de qualidade:** piloto de 15 itens com retorno, ordem aleatória, nota e metadados ocultos, sessões de no máximo 40 min, reteste de 20 itens de T1 depois de 7 dias ou mais e, se possível, um colega anotando 40 itens. Reporte κ/α com IC por bootstrap.
5. **Tempo:** cerca de **2 h 45 min** no total (faixa de 2 h 15 a 3 h 30), dentro do limite de 3–4 h [estimativa própria, a confirmar com os tempos que a própria página registra].

---

## 2. Comparação de ferramentas

### 2.1 O que as tarefas exigem

| Requisito | T1 (Olist do zero) | T2 (RePro misto) | T3 (auditoria RePro) |
|---|---|---|---|
| Itens | 100 (média de 16,4 palavras; máx. 36) | ~100, sorteados entre 976 avaliações "polaridade mista, vários tópicos" (média de 37,7 palavras; mediana 30) | 50, sorteados entre as 7.985 convertidas (média de 21,9 palavras) |
| Decisão | 5 aspectos × 4 classes | Só os tópicos já marcados pelo RePro, cada um recebendo positivo ou negativo. O conjunto precisa conter os dois, porque as 976 têm exatamente {NEGATIVO, POSITIVO}. 619 delas (63%) têm só 2 tópicos, ou seja, uma escolha binária | 5 × 4 às cegas e depois o veredito diante do rótulo convertido |
| O que esconder | Nota, atraso, UF, categoria | Nota | Rótulo convertido (até decidir) e nota |
| Saída | `olist_anotado.csv` compatível com `load_olist_gold` | CSV com `review_id` + `asp_*` | CSV com `asp_*` cego + `asp_*_repro` + veredito |

São três interfaces diferentes. Em especial, T2 precisa de **opções dinâmicas por item** (só os tópicos marcados) e de uma **validação de restrição** (aparecerem os dois sinais). T3 precisa **revelar** o rótulo depois da decisão.

### 2.2 Tabela comparativa

| Ferramenta | Instalação | 5 aspectos × 4 classes | Atalhos | Exportação | Curva de aprendizado | Ponto fraco para este caso |
|---|---|---|---|---|---|---|
| **Label Studio** (Community, código aberto) | `pip install label-studio` e depois `label-studio`; porta 8080 por padrão; também via Docker ([docs de instalação](https://labelstud.io/guide/install)) | Sim: 5 blocos `<Choices>` com `choice="single-radio"` e `required` ([tag Choices](https://labelstud.io/tags/choices)) | `hotkey` configurável em cada `<Choice>` ([tag Choice](https://labelstud.io/tags/choice)); atalhos globais remapeáveis ([hotkeys](https://labelstud.io/guide/hotkeys)) | JSON, JSON_MIN, CSV, TSV ([export](https://labelstud.io/guide/export)); o JSON traz `lead_time`, o tempo de rotulagem em segundos ([formato de tarefa](https://labelstud.io/guide/task_format)) | Baixa a média: configuração em XML | 20 opções exigem 20 teclas distintas. Para T2 é preciso montar opções dinâmicas (`value="$campo"` nos Choices; [tag Choices](https://labelstud.io/tags/choices)). Em T3, a "revelação depois de decidir" não sai pronta [não verificado]. O CSV sai com nomes de colunas da ferramenta e precisa ser convertido |
| **Argilla** (v2) | Hugging Face Spaces em um clique ou Docker; sem armazenamento persistente o Space **perde os dados** ao reiniciar ([quickstart](https://docs.argilla.io/latest/getting_started/quickstart/)) | Sim: vários `LabelQuestion` num mesmo `rg.Settings` (a lista de perguntas aceita várias; [guia de datasets](https://docs.argilla.io/latest/how_to_guides/dataset/)) | [não verificado] | Hub do Hugging Face via SDK ([quickstart](https://docs.argilla.io/latest/getting_started/quickstart/)); CSV só por código | Média: exige SDK Python e um servidor | Depende de servidor ou Space. Tem risco de perder dados no plano gratuito. É pensado para *suggestions* (pré-anotação), justamente o que queremos evitar em T1/T2 |
| **Doccano** | `pip install doccano`, depois `doccano init`, superusuário, servidor web e uma fila de tarefas num segundo terminal; ou Docker ([GitHub](https://github.com/doccano/doccano)) | Só de forma indireta: um conjunto único de rótulos no projeto, por exemplo 20 rótulos "produto:negativo", sem garantir uma classe por aspecto [inferência minha a partir dos tipos de projeto descritos no GitHub] | [não verificado] | Menu Dataset → Export ([docs](https://doccano.github.io/doccano/)) | Baixa | Não valida "exatamente um valor por aspecto", e a instalação tem mais passos |
| **Prodigy** (pago) | `pip` com chave de licença | Sim, com receitas/`choice` [não verificado neste caso] | Sim, é o ponto forte da ferramenta [não verificado] | JSONL | Média: receitas em Python | US$ 390 por licença pessoal vitalícia ([preço](https://prodi.gy/buy)). A licença acadêmica gratuita é para pesquisadores e funcionários. Alunos em disciplina, em geral, não se qualificam ([fórum oficial](https://support.prodi.gy/t/have-academic-licenses-for-students-been-discontinued/5188)) |
| **INCEpTION** | Instalador desktop ou servidor ([site](https://inception-project.github.io/)) | Sim, por camada de *document labels* ([site](https://inception-project.github.io/)) | [não verificado] | Vários formatos | Alta: feita para anotação multicamada, *entity linking* e curadoria ([Klie et al., 2018](https://aclanthology.org/C18-2002/)) | É exagero para classificar 250 textos curtos |
| **POTATO** | `pip install potato-annotation` e depois `potato start config.yaml` ([GitHub](https://github.com/davidjurgens/potato)) | Sim: esquemas radio/checkbox/likert ([GitHub](https://github.com/davidjurgens/potato)); vários esquemas por instância [não verificado] | Sim: atalhos de teclado, destaque de palavras-chave e *tooltips* ([Pei et al., 2022](https://aclanthology.org/2022.emnlp-demos.33/)) | Ver documentação de exportação [não verificado] | Baixa: configuração em YAML | Registra o tempo por item ([GitHub](https://github.com/davidjurgens/potato)). Licença GPLv3+ segundo o GitHub atual ([GitHub](https://github.com/davidjurgens/potato)). Restrição/validação em T2 [não verificado] |
| **Planilha** (Google Sheets/Excel com validação de dados) | Nenhuma | Sim: 5 colunas com lista suspensa | Fracos: digitar ou escolher na lista | CSV nativo | Nenhuma | Mostra a tabela inteira, o que facilita ver itens vizinhos e reordenar sem querer. É lenta no mouse e ruim para T2 e T3 |
| **Página HTML própria** | Abrir o arquivo no navegador | Sim, exatamente o esquema do projeto | Do jeito que for projetado | CSV no formato do repositório | Nenhuma para o anotador | Custa de 1 a 2 h para construir e testar, e o salvamento depende do navegador (ver mitigação em 6.2) |

### 2.3 Veredito: uma página própria é mesmo melhor?

**Sim, neste caso.** Os motivos:

1. **Escala minúscula, anotador único, nenhuma necessidade de servidor ou de vários usuários.** As vantagens do Label Studio e do Argilla (projetos, usuários, curadoria, API) não se aplicam aqui.
2. **As três tarefas pedem lógicas que as ferramentas prontas não cobrem sem gambiarra:** opções dinâmicas e validação "positivo e negativo presentes" em T2, e cego-depois-revela em T3.
3. **A saída já sai no formato do repositório** (`asp_*` com `ausente/positivo/negativo/neutro` e `obs`) e passa pelo validador que já existe (`load_olist_gold` rejeita valores inválidos).
4. **Viés sob controle por construção:** a página simplesmente não carrega nota, metadados nem rótulos de modelo.
5. **O tempo por item fica registrado** (início e fim), o que permite reportar o esforço com dados reais.

**Quando não vale a pena:** se construir a página passar de cerca de 2 h, ou se surgir um segundo anotador remoto. Nesse caso, use o **Label Studio**: 5 blocos `<Choices>` com `hotkey`, exportação JSON com `lead_time` e um script de conversão para `asp_*`.

---

## 3. Boas práticas (anotador único, pequena escala)

### 3.1 Ciclo de desenvolvimento do esquema
- Siga o ciclo **MATTER** (*Model, Annotate, Train, Test, Evaluate, Revise*) de [Pustejovsky e Stubbs (2012)](https://oreilly.com/library/view/natural-language-annotation/9781449332693/pr01.html): **guia → piloto → revisão do guia → anotação → medição**. O `anotacao/guia_de_anotacao.md` já cobre o "Model". Falta registrar as revisões feitas depois do piloto (com data) e **congelar o guia** antes da anotação valer.
- [Hovy e Lavid (2010)](https://www.jbe-platform.com/content/books/9789027249319-scl.113.intro) defendem tratar a anotação como procedimento metodológico explícito: guia, treinamento, medição de concordância e relato [conteúdo não verificado diretamente; citação bibliográfica confirmada].
- Uma meta-análise de 96 estudos mostra que a concordância depende, entre outros fatores, do **número de categorias** e do **treinamento** dos anotadores ([Bayerl; Paul, 2011](https://aclanthology.org/J11-4004/)). Isso justifica manter 4 classes fixas e fazer um piloto com retorno.

### 3.2 Rodada piloto
- Use **15 itens do RePro com polaridade única** (fora da amostra de T3) e compare **na hora** com o rótulo convertido. É calibração com retorno de um anotador humano externo, sem gastar itens do teste. Ajuste os casos difíceis do guia e só então comece T3/T1.
- Não use os 100 da Olist no piloto, para não contaminar o conjunto de teste de mudança de domínio.

### 3.3 Ordem e cegueira
- **Ordem aleatória**, com semente fixa e registrada. As anotações sofrem **efeitos de sequência**: o julgamento de um item é influenciado pelos anteriores ([Mathur; Baldwin; Cohn, 2017](https://aclanthology.org/D17-1306/); [Beck et al., 2024](https://preview.aclanthology.org/setup/2024.uncertainlp-1.8)). `sample_gold` já embaralha (`ordem_anotacao`). Em T2 e T3, embaralhe também e **intercale estratos** (notas, número de tópicos) para não haver blocos homogêneos.
- **Ocultar nota e metadados**: o guia já manda anotar só o que está escrito, e a nota está fora do CSV. Mantenha isso em T2 e T3. Em T3, os rótulos convertidos ficam ocultos até a decisão.

### 3.4 Fadiga
- Sessões de **no máximo 35–40 min**, com 5–10 min de pausa. Estratégia operacional: blocos de 25–40 itens, nunca uma tarefa inteira de uma vez [recomendação prática; não encontrei estudo em PLN com limite numérico verificado].
- Registre o tempo por item e confira se ele cai demais no fim das sessões (sinal de anotação apressada).

### 3.5 Variação legítima
- Nem toda divergência é erro: há casos genuinamente ambíguos ([Plank, 2022](https://aclanthology.org/2022.emnlp-main.731/)). Por isso:
  - mantenha a coluna `obs` e uma marca de **"dúvida"** por item;
  - em T2, ofereça a saída **"indecidível"** (por exemplo, quando o próprio tópico é misto), em vez de forçar uma escolha;
  - reporte quantos itens foram marcados como dúvida ou indecidíveis.

### 3.6 Consistência intra-anotador (teste-reteste)
- Reanote **20 itens de T1 (20%)**, sorteados, **7 dias ou mais depois**, em ordem nova e sem ver as respostas anteriores. Calcule κ de Cohen e α de Krippendorff entre as duas rodadas.

### 3.7 Concordância com outro anotador
1. **T3 às cegas × anotador do RePro (50 itens, 250 decisões aspecto-item):** é gratuito, porque já faz parte da auditoria.
2. **Colega em 40 itens de T1** (cerca de 25 min de trabalho dele, com o mesmo guia e a mesma página): mede a reprodutibilidade do esquema na Olist. É opcional, mas recomendado.

Use κ de Cohen (dois anotadores) e α de Krippendorff nominal, que serve para qualquer número de anotadores e lida com dados faltantes ([Hayes; Krippendorff, 2007](https://www.tandfonline.com/doi/10.1080/19312450709336664)). A revisão clássica da área sobre esses coeficientes é [Artstein e Poesio (2008)](https://aclanthology.org/J08-4004/).

### 3.8 Prevalência e o "paradoxo do kappa"
- `ausente` domina (e `anuncio`/`neutro` são raros). Isso pode gerar **alta concordância bruta com κ baixo** (Feinstein; Cicchetti, 1990: referência bibliográfica confirmada em índices, texto não lido [não verificado]). Reporte sempre:
  - concordância bruta;
  - κ/α **por aspecto** e agregado;
  - a matriz de confusão.

### 3.9 Tamanhos mínimos defensáveis
Pela aproximação de grandes amostras do erro padrão do κ [estimativa própria], com concordância observada de 0,90 e esperada de 0,60 (κ = 0,75), o IC95% fica em:

| n | IC95% do κ |
|---|---|
| 20 | ±0,33 |
| 30 | ±0,27 |
| 50 | ±0,21 |
| 100 | ±0,15 |

Ou seja:
- 20 itens de reteste dão IC largo **por aspecto**. Reporte o **agregado** (100 decisões, com bootstrap **por comentário**, porque os 5 aspectos de um mesmo texto não são independentes) e trate o valor por aspecto como descritivo.
- 40–50 itens é o mínimo razoável para uma estimativa entre anotadores com IC de cerca de ±0,2. [Sim e Wright (2005)](https://research.birmingham.ac.uk/en/publications/the-kappa-statistic-in-reliability-studies-use-interpretation-and/) tabulam tamanhos de amostra para estudos de κ [tabelas não verificadas].
- Referência de leitura: Krippendorff recomenda α ≥ 0,80, e 0,667 ≤ α < 0,80 só para conclusões provisórias [não verificado na obra original]. Artstein e Poesio (2008) discutem que esses limiares dependem da tarefa ([J08-4004](https://aclanthology.org/J08-4004/)).

### 3.10 Erros comuns ao relatar
A análise de 591 artigos de datasets mostra erros recorrentes justamente no **uso da concordância** e no **cálculo de taxas de erro** ([Klie; Eckart de Castilho; Gurevych, 2024](https://aclanthology.org/2024.cl-3.1/)). Deixe explícito o que foi comparado com o quê, n, unidade (item × aspecto) e método do IC.

---

## 4. Pré-anotação (aceitar/corrigir) × anotação do zero

| Evidência | Achado |
|---|---|
| [Fort e Sagot (2010)](https://aclanthology.org/W10-1807/), POS | Avaliam ganho de tempo e qualidade com atenção explícita a vieses [números não verificados: o PDF não pôde ser lido] |
| [Berzak et al. (2016)](https://arxiv.org/abs/1605.04481), sintaxe | **Ancoragem mensurável:** corrigir a saída do *parser* superestima o desempenho do *parser* e gera anotação de qualidade **menor** que a feita sem saída de máquina |
| [Mikulová et al. (2022)](https://aclanthology.org/2022.lrec-1.312/), sintaxe tcheca | Com um *parser* de alta acurácia, a pré-anotação **acelerou** e deixou a anotação **mais consistente** sem perda de qualidade |
| [Schroeder, Roy e Kabbara (2025)](https://aclanthology.org/2025.findings-acl.1323/), tarefas subjetivas com LLM | Pré-registrado, 7.000+ anotações. As sugestões de LLM **não aceleraram**, aumentaram a confiança declarada, foram muito aceitas e **mudaram a distribuição dos rótulos**. Usar esses rótulos para avaliar LLMs **inflou significativamente** o desempenho reportado |

**Conclusões para o projeto:**

1. **T1 e T2: anotar do zero, sem sugestão de modelo nenhum.** Esses rótulos vão medir regras, TF-IDF, BERTimbau, Albertina e **o próprio Qwen**. Pré-anotar com o LLM (ou com o TF-IDF) contaminaria exatamente a comparação central do trabalho (humano × LLM). O ganho de tempo também é incerto em tarefas subjetivas (Schroeder et al., 2025), e os textos são curtos (16 palavras em média em T1).
2. **T2 já é uma "pré-anotação humana" restrita:** os tópicos e o conjunto de polaridades vêm do anotador do RePro, e só falta atribuir. Isso é aceitável: a informação é humana, não do modelo avaliado, e a restrição reduz a decisão a cerca de 2 opções em 63% dos casos. **Reporte** que T2 parte dos tópicos do RePro e conte os itens em que você discordaria dos tópicos (anote em `obs`).
3. **T3 vira "às cegas + revelação"** em vez de "confirmar/corrigir". O modo confirmar seria mais rápido (cerca de 10 min contra cerca de 25 min), mas ficaria sujeito à ancoragem documentada por Berzak et al. (2016). Na prática, mediria a sua tendência a aceitar, não a qualidade da conversão.
4. **Quando a pré-anotação seria aceitável:** para gerar **dados de treino** extras (nunca de teste), com um modelo cuja acurácia já foi medida, e reportando no artigo a taxa de aceitação sem mudança e a comparação com uma subamostra feita do zero.

**Como reportar (se algum dia usar):** a fonte da sugestão, a taxa de aceitação sem alteração, o tempo com e sem sugestão, e a concordância entre a subamostra feita do zero e a pré-anotada.

---

## 5. Estimativa de tempo

Premissas [estimativa própria]: leitura de cerca de 200–250 palavras por minuto, mais o tempo de decisão. Com atalhos e "ausente" como padrão, a maioria dos comentários pede só 1 ou 2 marcações.

| Etapa | Itens | s/item | Tempo |
|---|---|---|---|
| Piloto com retorno (RePro polaridade única) + ajuste do guia | 15 | 40 | ~10 min + 10 min de revisão |
| T3 às cegas (5 × 4) | 50 | 30 | ~25 min |
| T3 revelação e veredito (só onde houver divergência; estimo 10–20 itens) | ~15 | 40 | ~10 min |
| T1 do zero (5 × 4) | 100 | 35 | ~60 min (2 sessões) |
| T2 atribuir polaridade (textos mais longos, decisão binária) | 100 | 20 | ~35 min |
| Reteste de T1 (depois de 7 dias ou mais) | 20 | 30 | ~10 min |
| Adjudicação final das dúvidas | ~15 | 40 | ~10 min |
| **Total do aluno** | | | **~2 h 50 min** (faixa de 2 h 15 a 3 h 30) |
| Colega (opcional) | 40 | 35 | ~25 min (tempo do colega) |

**Como reduzir sem perder validade:**
- Padrão "ausente" pré-marcado, para clicar só no que aparece (o padrão é a classe majoritária, um viés pequeno e conhecido; reporte).
- Em T2, mostrar só os tópicos marcados e preencher automaticamente o último tópico quando a restrição já define a resposta (por exemplo, com 2 tópicos, escolher um positivo faz o outro virar negativo).
- Em T3, revelar apenas as divergências.
- Se o tempo apertar, **reduza T2 para cerca de 60 itens** antes de cortar o reteste ou o piloto. Com 60 itens o IC de qualquer proporção ainda fica em cerca de ±0,12, e T2 serve principalmente para cobrir a categoria "mista".
- **Não** reduza T1 abaixo de 100 nem pule o reteste: são eles que sustentam as afirmações de domínio e de confiabilidade.

---

## 6. Fluxo recomendado

### 6.1 Ordem das tarefas e motivos
1. **Preparar os arquivos de entrada** (script, ~15 min com ajuda): sortear os 100 de T2 (semente fixa, estratificando por número de tópicos 2/3/4+), os 50 de T3 (do **teste** do RePro, estratificados por aspecto e polaridade) e os 15 do piloto (fora de T3). Guardar `review_id` **e um hash do texto**, porque `repro_{i}` depende da ordem das linhas do CSV.
2. **Piloto** (15) → revisar o guia → **congelar o guia** (anotar data e versão).
3. **T3 às cegas** (50): calibra você no esquema e já mede a concordância com o RePro.
4. **T1** (100) em 2–3 sessões.
5. **T2** (100): depois de T1, quando você já estiver calibrado nos limites entre os aspectos.
6. **T3 revelação**: veredito por divergência.
7. **Reteste** (20 de T1), 7 dias ou mais depois de T1.
8. **(Opcional) Colega** em 40 de T1, com o guia congelado e sem conversar sobre os itens.
9. **Adjudicação**: só para gerar a versão final de T1. As medidas de concordância usam as anotações **antes** da adjudicação.

### 6.2 Especificação mínima da página
- **Um arquivo** `anotacao/anotar.html`, sem dependências externas. Carrega um CSV escolhido pelo usuário (`<input type=file>`), o que evita embutir os textos do RePro no repositório.
- **Teclado:**

| Tecla | Ação |
|---|---|
| `P` `E` `R` `A` `O` | Seleciona o aspecto (produto, entrega, recebimento, anuncio, outros) |
| `1` / `2` / `3` / `0` | positivo / negativo / neutro / ausente |
| `Enter` | Salva e vai para o próximo |
| `Backspace` | Volta |
| `D` | Marca dúvida |
| `Tab` | Vai para o campo `obs` |

- **T2:** mostra só os tópicos marcados. Bloqueia o `Enter` se não houver um positivo e um negativo, a menos que `X` (indecidível) tenha sido marcado.
- **T3:** etapa 1 às cegas; depois do `Enter`, revela o rótulo convertido e pede veredito: `C` concordo comigo, `R` RePro estava certo, `A` ambíguo.
- **Salvamento:** `localStorage` a cada tecla (dentro de `try/catch`), botão **Exportar CSV**, e lembrete para exportar a cada 25 itens. Cada linha tem `t_inicio`/`t_fim`.
- **Validação final:** rodar `load_olist_gold("olist_anotado.csv")`, que já falha com valor inválido.

### 6.3 Formato de saída

| Arquivo | Colunas |
|---|---|
| `data/annotated/olist_anotado.csv` (T1) | `ordem_anotacao, review_id, texto, asp_produto, asp_entrega, asp_recebimento, asp_anuncio, asp_outros, obs`, que é o mesmo cabeçalho de `olist_para_anotar.csv` e é compatível com `load_olist_gold`. Os tempos e a marca de dúvida vão num arquivo separado, `olist_anotacao_log.csv` (`review_id, t_inicio, t_fim, duvida, sessao`), para não mexer no formato |
| `olist_reteste.csv` | Mesmo formato de T1, com os 20 itens |
| `olist_colega.csv` (opcional) | Mesmo formato de T1 |
| `repro_mistas_anotado.csv` (T2) | `review_id, hash_texto, asp_produto, …, asp_outros, indecidivel, obs`. Valores: `ausente` para os tópicos não marcados; `positivo`/`negativo` (e `neutro` só com justificativa em `obs`) para os marcados |
| `repro_auditoria.csv` (T3) | `review_id, hash_texto, asp_*` (cego), `asp_*_repro` (convertido), `veredito_*` (`concordo`/`repro_certo`/`ambiguo`), `obs` |

**Atenção à licença:** o README diz que o RePro **não é redistribuído**. Os arquivos de T2/T3 não devem trazer o texto (apenas `review_id` + hash), ou devem ficar fora do git [verificar a licença do RePro no Hugging Face; não verificado].

---

## 7. Controles de qualidade e o que reportar

### 7.1 Controles
1. Guia versionado e congelado depois do piloto, com o log das mudanças.
2. Ordem aleatória com semente registrada; nota, metadados e rótulos de modelo ocultos.
3. Sessões de no máximo 40 min, com tempo por item registrado. Sinalize itens resolvidos em menos de 5 s em T1.
4. Validação automática dos valores (`load_olist_gold`) e, em T2, da restrição {positivo, negativo}.
5. Reteste de 20% depois de 7 dias ou mais.
6. Concordância externa: T3 às cegas × RePro (50) e, opcionalmente, colega × você (40).
7. Medidas calculadas **antes** da adjudicação; a versão final adjudicada é usada só como gabarito.

### 7.2 Tabela para o relatório

| Medida | n | Como |
|---|---|---|
| Distribuição dos rótulos por aspecto (T1, T2) | 100 / ~100 | Contagens (`label_distribution`) |
| Intra-anotador (reteste) | 20 itens × 5 aspectos | Concordância bruta, κ de Cohen, α nominal; IC95% por **bootstrap por comentário** (1.000+ reamostragens); por aspecto (descritivo) e agregado |
| Você × RePro (T3 às cegas) | 50 × 5 | Idem + matriz de confusão |
| Taxa de erro da conversão do RePro | 50 | Proporção com veredito "conversão errada", com **IC de Wilson**. Exemplo [cálculo próprio]: 2/50 dá 1,1%–13,5%; 5/50 dá 4,3%–21,4%; 0/50 dá 0%–7,1%. Deixe explícito que, com n = 50, só se detectam erros sistemáticos grandes |
| Você × colega (opcional) | 40 × 5 | κ/α com IC |
| T2: proporção indecidível | ~100 | Contagem + exemplos |
| Esforço | todos | Mediana e intervalo interquartil de segundos por item, por tarefa |

### 7.3 O que escrever na seção de metodologia
- Quem anotou (um anotador, o autor) e o perfil de cada anotador.
- O guia: versão final no apêndice.
- O piloto: o que mudou depois dele.
- Que a nota e os metadados ficaram ocultos e que **não houve pré-anotação por modelo** em T1/T2.
- Como T2 usa os tópicos do RePro.
- A limitação de anotador único e o tamanho pequeno, com os IC explícitos.
- Que a divergência legítima foi registrada (`obs`, dúvida, indecidível) em vez de eliminada ([Plank, 2022](https://aclanthology.org/2022.emnlp-main.731/)).

---

## 8. Referências (ABNT)

ARTSTEIN, R.; POESIO, M. Survey article: inter-coder agreement for computational linguistics. **Computational Linguistics**, v. 34, n. 4, p. 555–596, 2008. DOI: 10.1162/coli.07-034-R2. Disponível em: https://aclanthology.org/J08-4004/. Acesso em: 30 out. 2026.

BAYERL, P. S.; PAUL, K. I. What determines inter-coder agreement in manual annotations? A meta-analytic investigation. **Computational Linguistics**, v. 37, n. 4, p. 699–725, 2011. DOI: 10.1162/COLI_a_00074. Disponível em: https://aclanthology.org/J11-4004/. Acesso em: 30 out. 2026.

BECK, J. et al. Order effects in annotation tasks: further evidence of annotation sensitivity. In: WORKSHOP ON UNCERTAINTY-AWARE NLP (UncertaiNLP), 1., 2024. **Proceedings** [...]. [S. l.]: ACL, 2024. p. 81–86. Disponível em: https://preview.aclanthology.org/setup/2024.uncertainlp-1.8. Acesso em: 30 out. 2026.

BERZAK, Y. et al. Anchoring and agreement in syntactic annotations. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING, 2016, Austin. **Proceedings** [...]. Austin: ACL, 2016. p. 2215–2224. DOI: 10.18653/v1/D16-1239. Disponível em: https://aclanthology.org/D16-1239/. Acesso em: 30 out. 2026.

DOCCANO. **doccano**: open source annotation tool for machine learning practitioners. [S. l.], [2018–]. Disponível em: https://github.com/doccano/doccano. Acesso em: 30 out. 2026.

EXPLOSION. **Prodigy**: buy. [S. l.], [2026]. Disponível em: https://prodi.gy/buy. Acesso em: 30 out. 2026.

FEINSTEIN, A. R.; CICCHETTI, D. V. High agreement but low kappa: I. The problems of two paradoxes. **Journal of Clinical Epidemiology**, v. 43, n. 6, p. 543–549, 1990.

FORT, K.; SAGOT, B. Influence of pre-annotation on POS-tagged corpus development. In: LINGUISTIC ANNOTATION WORKSHOP, 4., 2010, Uppsala. **Proceedings** [...]. Uppsala: ACL, 2010. p. 56–63. Disponível em: https://aclanthology.org/W10-1807/. Acesso em: 30 out. 2026.

HAYES, A. F.; KRIPPENDORFF, K. Answering the call for a standard reliability measure for coding data. **Communication Methods and Measures**, v. 1, n. 1, p. 77–89, 2007. DOI: 10.1080/19312450709336664.

HOVY, E.; LAVID, J. Towards a 'science' of corpus annotation: a new methodological challenge for corpus linguistics. **International Journal of Translation**, v. 22, n. 1, p. 13–36, 2010.

HUMANSIGNAL. **Label Studio documentation**: install, Choices, Choice, hotkeys, export, task format. [S. l.], [2026]. Disponível em: https://labelstud.io/guide/. Acesso em: 30 out. 2026.

ARGILLA. **Argilla documentation**: quickstart; dataset how-to. [S. l.], [2026]. Disponível em: https://docs.argilla.io/latest/. Acesso em: 30 out. 2026.

KLIE, J.-C. et al. The INCEpTION platform: machine-assisted and knowledge-oriented interactive annotation. In: INTERNATIONAL CONFERENCE ON COMPUTATIONAL LINGUISTICS: SYSTEM DEMONSTRATIONS, 27., 2018, Santa Fe. **Proceedings** [...]. Santa Fe: ACL, 2018. p. 5–9. Disponível em: https://aclanthology.org/C18-2002/. Acesso em: 30 out. 2026.

KLIE, J.-C.; ECKART DE CASTILHO, R.; GUREVYCH, I. Analyzing dataset annotation quality management in the wild. **Computational Linguistics**, v. 50, n. 3, p. 817–866, 2024. DOI: 10.1162/coli_a_00516. Disponível em: https://aclanthology.org/2024.cl-3.1/. Acesso em: 30 out. 2026.

MATHUR, N.; BALDWIN, T.; COHN, T. Sequence effects in crowdsourced annotations. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING, 2017, Copenhagen. **Proceedings** [...]. Copenhagen: ACL, 2017. p. 2860–2865. DOI: 10.18653/v1/D17-1306. Disponível em: https://aclanthology.org/D17-1306/. Acesso em: 30 out. 2026.

MIKULOVÁ, M. et al. Quality and efficiency of manual annotation: pre-annotation bias. In: LANGUAGE RESOURCES AND EVALUATION CONFERENCE, 13., 2022, Marseille. **Proceedings** [...]. Marseille: ELRA, 2022. p. 2909–2918. Disponível em: https://aclanthology.org/2022.lrec-1.312/. Acesso em: 30 out. 2026.

PEI, J. et al. POTATO: the portable text annotation tool. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING: SYSTEM DEMONSTRATIONS, 2022, Abu Dhabi. **Proceedings** [...]. Abu Dhabi: ACL, 2022. p. 327–337. DOI: 10.18653/v1/2022.emnlp-demos.33. Disponível em: https://aclanthology.org/2022.emnlp-demos.33/. Acesso em: 30 out. 2026.

PLANK, B. The "problem" of human label variation: on ground truth in data, modeling and evaluation. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING, 2022, Abu Dhabi. **Proceedings** [...]. Abu Dhabi: ACL, 2022. p. 10671–10682. DOI: 10.18653/v1/2022.emnlp-main.731. Disponível em: https://aclanthology.org/2022.emnlp-main.731/. Acesso em: 30 out. 2026.

PONTIKI, M. et al. SemEval-2016 Task 5: aspect based sentiment analysis. In: INTERNATIONAL WORKSHOP ON SEMANTIC EVALUATION, 10., 2016, San Diego. **Proceedings** [...]. San Diego: ACL, 2016. p. 19–30. DOI: 10.18653/v1/S16-1002. Disponível em: https://aclanthology.org/S16-1002/. Acesso em: 30 out. 2026.

PUSTEJOVSKY, J.; STUBBS, A. **Natural language annotation for machine learning**: a guide to corpus-building for applications. Sebastopol: O'Reilly, 2012. Disponível em: https://oreilly.com/library/view/natural-language-annotation/9781449332693/pr01.html. Acesso em: 30 out. 2026.

SCHROEDER, H.; ROY, D.; KABBARA, J. Just put a human in the loop? Investigating LLM-assisted annotation for subjective tasks. In: FINDINGS OF THE ASSOCIATION FOR COMPUTATIONAL LINGUISTICS: ACL 2025, Vienna. **Proceedings** [...]. Vienna: ACL, 2025. p. 25771–25795. DOI: 10.18653/v1/2025.findings-acl.1323. Disponível em: https://aclanthology.org/2025.findings-acl.1323/. Acesso em: 30 out. 2026.

SILVA, L. N. dos S. et al. RePro: a benchmark dataset for opinion mining in Brazilian Portuguese. In: INTERNATIONAL CONFERENCE ON COMPUTATIONAL PROCESSING OF PORTUGUESE (PROPOR), 16., 2024, Santiago de Compostela. **Proceedings** [...]. [S. l.]: ACL, 2024. v. 1, p. 432–440. Disponível em: https://aclanthology.org/2024.propor-1.44/. Acesso em: 30 out. 2026.

SIM, J.; WRIGHT, C. C. The kappa statistic in reliability studies: use, interpretation, and sample size requirements. **Physical Therapy**, v. 85, n. 3, p. 257–268, 2005. DOI: 10.1093/ptj/85.3.257.

UKP LAB. **INCEpTION**. Darmstadt, [2026]. Disponível em: https://inception-project.github.io/. Acesso em: 30 out. 2026.

JURGENS, D. et al. **POTATO** (repositório). [S. l.], [2026]. Disponível em: https://github.com/davidjurgens/potato. Acesso em: 30 out. 2026.
