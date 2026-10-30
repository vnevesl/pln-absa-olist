# Ponto fraco 2: casos de polaridade mista descartados e polaridade por aspecto inferida

*Levantamento feito em 30/10/2026 para o trabalho final de PLN (ABSA por categoria no RePro e na Olist).
Os números sobre o RePro foram recalculados a partir de `data/raw/RePro.csv` e de `src/absa/repro.py`.
Os números da literatura têm link para a fonte. Estimativas próprias aparecem como **[estimativa]** e o que não consegui conferir aparece como **[não verificado]**.*

---

## Resumo executivo

1. O descarte não é aleatório. Os 2.018 casos mistos são **37,5% das avaliações de 3★ e 31% das de 2★**, contra 6–8% nas de 1★ e 5★, e têm o dobro do tamanho (mediana de 30 palavras contra 16). Por isso, o teste atual de 1.597 avaliações superestima o desempenho exatamente na situação que justifica fazer ABSA.
2. A literatura não resolve isso com uma classe "conflito". Ela aparece no SemEval-2014, mas é rara e mal aprendida, e foi abandonada em 2015. A saída padrão é um **conjunto de teste separado, só com casos multiaspecto e multissentimento** (MAMS). Recomendo o mesmo: um "teste misto" com **150** casos desambiguados à mão, sem pré-anotação de LLM, com segundo anotador em 50.
3. Dá para usar os 976 + 1.042 casos mistos **no treino** com uma perda de rótulo parcial com restrição de conjunto. Ela tem forma fechada por inclusão-exclusão, ocupa cerca de 15 linhas de PyTorch e entra direto em `models/encoder.py`.
4. A conversão "polaridade única vale para todos os tópicos" precisa de **auditoria de cerca de 100 casos do teste**, estratificada por número de tópicos e reportada com intervalo de Wilson. Com 50 casos, o intervalo seria largo demais (para 10% de erro observado, IC95% de 4% a 21%).
5. Achado extra que pesa na decisão: o RePro anotou **título e corpo**, mas os modelos do projeto leem só `texto`. Em casos mistos, a parte positiva costuma estar no título ("Bom produto!" seguido de "Porém recebi um filtro velho…"). Incluir o título na entrada custa minutos e deve vir antes de tudo.

---

## 1. Diagnóstico com os dados do repositório

Valores recalculados com o `convert()` de `src/absa/repro.py`:

| Grupo | n | Observação |
|---|---|---|
| Polaridade única POSITIVO / NEGATIVO / NEUTRO | 4.127 / 3.449 / 409 | entram no treino/val/teste |
| Mista, 1 tópico | 1.042 | **1.026 só PRODUTO**, 7 ENTREGA, 7 RECEBIMENTO, 2 OUTROS |
| Mista, ≥ 2 tópicos | 976 | 619 com 2 tópicos, 305 com 3, 48 com 4, 4 com 5. Combinações mais comuns: produto+entrega (222), produto+recebimento (166), produto+anúncio (94), produto+outros (90) |
| Teste atual | 1.597 | **701 (44%) têm ≥ 2 tópicos** e receberam a mesma polaridade propagada a todos |

Fração de casos mistos por nota (o RePro é balanceado, com cerca de 2.000 por nota): 1★ 6,3% · 2★ 31,0% · **3★ 37,5%** · 4★ 18,3% · 5★ 7,7%. Os próprios autores do RePro registram que o rótulo "positivo/negativo" se concentra em 3★ (751) e 2★ (621) ([SILVA et al., 2024](https://aclanthology.org/2024.propor-1.44/)).

O que o artigo do RePro diz sobre a anotação ([SILVA et al., 2024](https://aclanthology.org/2024.propor-1.44/), seção 3):
- a polaridade é **uma classe por avaliação** entre quatro (positivo, negativo, neutro, positivo/negativo), atribuída considerando **título e corpo**;
- os tópicos são multirrótulo e, **"em caso de dúvida entre tópicos, marcar ambos"**. Isso infla o número de tópicos e é uma fonte direta de erro na propagação da polaridade;
- cada amostra teve pelo menos 2 anotadores, com curadoria por um terceiro; o κ de Cohen médio foi **0,68 para tópicos e 0,71 para polaridade**;
- um BERTimbau de referência obtém F1 de 0,88 na classe "Neg./Pos." no nível da avaliação, ou seja, detectar que a avaliação é mista é aprendível.

Inspecionando 10 casos mistos com vários tópicos (`random_state=2`, amostra indicativa), encontrei três padrões:
- **decomponível**: "não gostei do produto… quanto à entrega, foi muito rápido" → produto negativo, entrega positivo;
- **conflito dentro do próprio aspecto**: "chegou bem antes do prazo. Parece resistente… porém a lâmina não tritura bem" → entrega positivo, *produto* positivo e negativo ao mesmo tempo;
- **positivo só no título**: título "Bom", corpo "chegou todo amassado até arranhões". Pelo corpo, tudo é negativo.

Essas três situações definem o desenho a seguir.

---

## 2. Como a literatura trata o conflito

### 2.1 SemEval-2014 Task 4: a classe "conflict"
O SemEval-2014 definiu "conflict" para quando o mesmo aspecto recebe opinião positiva e negativa, tanto por termo quanto por categoria ([PONTIKI et al., 2014](https://aclanthology.org/S14-2004/)). Contagens da Tabela 3, por categoria em restaurantes: conflict tem **163 no treino e 52 no teste**, de 3.713 e 1.025 (cerca de 4–5%). Por termo, 45/16 em laptops e 91/14 em restaurantes (Tabela 2). A classe é rara, e trabalhos posteriores costumam **excluí-la** "para manter consistência com modelos anteriores" ([exemplo em estudo com LLMs, arXiv 2310.18025](https://arxiv.org/pdf/2310.18025)).

### 2.2 SemEval-2015 e 2016: conflito vira "sentimento dominante"
- **2015**: a classe "conflict" **deixou de ser usada**, porque, com o esquema de aspectos mais fino (Entidade#Atributo), a ocorrência de positivo e negativo para o mesmo atributo numa frase ficou rara. Nesses casos, "o sentimento dominante foi escolhido" ([PONTIKI et al., 2015](https://aclanthology.org/S15-2082/)).
- **2016**: na subtarefa **SB2, ABSA no nível do texto** (o análogo mais próximo do RePro, que é por avaliação), cada par {categoria, polaridade} resume a avaliação. Quando o aspecto aparece com sentimentos diferentes, anota-se o **dominante**, e "conflict" só **quando o dominante não é claro** ([PONTIKI et al., 2016](https://aclanthology.org/S16-1002/)).

**Lição para o projeto.** Conflito dentro de um aspecto é raro quando os aspectos são finos e comum quando são grossos. O aspecto `produto` do projeto é grosso (qualidade, tamanho, durabilidade, aparência), e é exatamente onde estão 1.026 dos 1.042 casos mistos com um tópico.

### 2.3 MAMS: teste separado e mais difícil, sem classe nova
O MAMS ([JIANG et al., 2019](https://aclanthology.org/D19-1654/)) parte da constatação de que, nos conjuntos existentes, a maioria das frases tem um aspecto ou vários com a mesma polaridade. Isso faz a ABSA **degenerar em classificação de sentimento da frase**: classificadores no nível da frase, que ignoram o aspecto, "ainda alcançam resultados competitivos". A Tabela 1 do artigo mostra que só **454 de 4.738** instâncias de ACSA do SemEval-2014 Restaurantes são multiaspecto e multissentimento. A solução foi um conjunto em que **toda frase tem pelo menos dois aspectos com polaridades diferentes**. A versão ACSA tem 8 categorias, 3 classes (sem conflito), 7.090 instâncias de treino e 901 de teste, anotadas por 3 pesquisadores. Os modelos do estado da arte caem bastante nele em relação ao SemEval-2014.

**Lição para o projeto.** O teste atual, sem casos mistos, é o cenário "degenerado" descrito pelo MAMS. Um teste misto à parte é a forma reconhecida de medir o que importa. Uma pesquisa sobre conjuntos de ABSA faz uma ressalva sobre a forma de construção do MAMS ([CHEBOLU et al., 2022, arXiv 2204.05232](https://arxiv.org/pdf/2204.05232)), o que reforça que um teste misto deve ser reportado **ao lado** do teste principal, não no lugar dele.

### 2.4 ACOS / ASQP: o conflito some com a granularidade
O ACOS ([CAI; XIA; YU, 2021](https://aclanthology.org/2021.acl-long.29/)) e o ASQP ([ZHANG et al., 2021](https://aclanthology.org/2021.emnlp-main.726/)) anotam quádruplas (categoria, termo de aspecto, termo de opinião, polaridade). Um aspecto com elogio e crítica vira **duas quádruplas** com opiniões diferentes, e não há classe de conflito. O ACOS ainda motiva a tarefa pela alta frequência de aspectos e opiniões **implícitos** em avaliações de produto. Para este trabalho isso fica fora de alcance, porque exigiria anotar trechos. Mas justifica dizer no relatório que o "misto dentro do aspecto" é um artefato da granularidade grossa.

### 2.5 Adicionar uma classe "misto/conflito"?

| | A favor | Contra |
|---|---|---|
| Classe "misto" para os 1.042 casos de um tópico | Recupera 10% do corpus. O rótulo é humano e direto (o RePro já tem "positivo/negativo", com F1 de 0,88 no nível da avaliação). Dá informação útil ao varejista ("gostou, mas…"). | Aparece quase só em `produto` (1.026 de 1.042), o que deixa a classe desbalanceada nos outros 4 aspectos. Rompe a comparação com o esquema de 4 classes e com a literatura, que descartou o conflito ([2015](https://aclanthology.org/S15-2082/), [MAMS](https://aclanthology.org/D19-1654/)). Obriga a mudar prompt, regras, métricas e o guia da Olist. Fica incoerente com os 976 casos de vários tópicos, que também têm conflito interno sem rótulo. O SemEval-2014 mostra que é a classe mais difícil de aprender. |

**Recomendação.** **Não** adicionar a 5ª classe ao esquema principal. Em vez disso:
(i) usar os 1.042 casos no treino como **rótulo parcial** `produto ∈ {positivo, negativo, neutro}`. Isso ensina pelo menos "produto foi mencionado e não é ausente" (ver seção 3);
(ii) no teste misto, permitir que o anotador marque `misto` como **observação** (coluna `obs`), sem que vire classe. Assim dá para reportar "x% dos aspectos em casos mistos têm conflito interno", seguindo a regra do SemEval-2016 SB2 (dominante quando claro, conflito quando não);
(iii) se sobrar tempo, rodar uma ablação com 5 classes só para `produto`. É opcional e de baixa prioridade.

---

## 3. Usar os casos mistos no treino

### 3.1 Enquadramento
O RePro dá, para cada avaliação mista: o conjunto de tópicos *T* e o fato de que a avaliação contém elogio e crítica. A atribuição aspecto → polaridade é desconhecida. Três linhas da literatura se encaixam:

- **Aprendizado com múltiplas instâncias (MIL)**: treinar um preditor de segmento a partir de rótulos do documento. O MILNET ([ANGELIDIS; LAPATA, 2018](https://aclanthology.org/Q18-1002/)) aprende sentimento por segmento só com o rótulo da avaliação. O MIL ponderado de [Pappas e Popescu-Belis (2017)](https://jair.org/index.php/jair/article/view/11051) faz o mesmo para notas por aspecto. [He et al. (2018)](https://aclanthology.org/P18-2092/) mostram ganhos ao transferir conhecimento de sentimento do documento para o nível do aspecto nos SemEval 2014–2016.
- **Aprendizado com rótulos parciais (PLL)**: cada exemplo traz um **conjunto de candidatos**, um dos quais é verdadeiro ([COUR; SAPP; TASKAR, 2011](https://jmlr.org/papers/v12/cour11a.html)). O método *classifier-consistent* (CC) de [Feng et al. (2020)](https://arxiv.org/abs/2007.08929) minimiza `−log Σ_{y∈Y} p(y|x)`, a probabilidade total do conjunto candidato (Eq. 12). O método *risk-consistent* repondera a entropia cruzada pela confiança do modelo normalizada dentro do conjunto, o que coincide com as atualizações do PRODEN ([LV et al., 2020](https://proceedings.mlr.press/v119/lv20a.html), com [código oficial em PyTorch](https://github.com/Lvcrezia77/PRODEN)).
- **Supervisão fraca**: o projeto já compara humano com LLM. Os casos mistos permitem um terceiro braço: o LLM desambigua **com a restrição dada** (tópicos e conjunto de polaridades fixos), o que é bem mais fácil que rotular do zero.

### 3.2 Particularidade do problema: o rótulo é de conjunto, não por aspecto
Tratar cada aspecto marcado como candidato `{pos, neg, neu}` de forma independente desperdiça informação. O que se sabe de fato é que **pelo menos um aspecto marcado é positivo e pelo menos um é negativo**. Isso é uma restrição conjunta, como no MIL, e evita a solução degenerada "tudo positivo".

Com probabilidades por aspecto *p_a(c)* (a cabeça softmax por aspecto que o `MultiAspectModel` já tem), e admitindo `neutro` como possível para um tópico marcado, a probabilidade do conjunto de atribuições válidas tem **forma fechada por inclusão-exclusão**, sem enumerar combinações:

```
P(válido) = Π_{a∉T} p_a(aus) · [ Π_{a∈T}(p+ + p− + p0) − Π_{a∈T}(p− + p0) − Π_{a∈T}(p+ + p0) + Π_{a∈T} p0 ]
            (todos fora de T ausentes) · (todos em T mencionados − sem positivo − sem negativo + sem nenhum dos dois)
```

Com um único tópico, o colchete dá zero: um aspecto não pode ser positivo e negativo ao mesmo tempo no esquema de 4 classes. Por isso, para os 1.042 casos de um tópico, usa-se só o rótulo parcial marginal `p+ + p− (+ p0)`. Esse é exatamente o argumento formal de que eles "não cabem" no esquema, e a perda o explicita.

```python
def perda_rotulo_parcial(logits, topicos, eps=1e-6):
    """Perda para avaliações com polaridade mista no RePro.
    logits: [B, 5, 4] na ordem POLARITIES (ausente, positivo, negativo, neutro).
    topicos: [B, 5] booleano com os tópicos marcados pelo humano no RePro."""
    p = logits.float().softmax(-1)
    aus, pos, neg, neu = p.unbind(-1)
    t, um = topicos.bool(), torch.ones_like(aus)
    fora = torch.where(t, um, aus).prod(-1)                     # não marcados -> ausente
    todos = torch.where(t, pos + neg + neu, um).prod(-1)
    sem_pos = torch.where(t, neg + neu, um).prod(-1)
    sem_neg = torch.where(t, pos + neu, um).prod(-1)
    so_neu = torch.where(t, neu, um).prod(-1)
    varios = t.sum(-1) > 1
    dentro = torch.where(varios, todos - sem_pos - sem_neg + so_neu,   # restrição de conjunto
                         (pos + neg + neu).mul(t).sum(-1))             # 1 tópico: só parcial
    return -torch.log((fora * dentro).clamp_min(eps)).mean()
```

*(Conferido numericamente: para 3 tópicos, a probabilidade calculada por inclusão-exclusão coincide com a soma por força bruta das 27 atribuições, 0,0066379 nos dois casos.)*

Perda total: `CE(convertidos) + λ · perda_rotulo_parcial(mistos)`, com λ ∈ {0,5; 1}. Nos casos mistos, a média da entropia cruzada atual (`_loss` em `encoder.py`) é substituída por esta. Para o TF-IDF + LR não há equivalente simples. O braço do rótulo parcial vale só para os encoders.

**Viabilidade.** O custo computacional é desprezível e não exige enumeração. O risco é o modelo resolver a restrição com atalhos posicionais ("primeira oração positiva"), uma preocupação geral de MIL e PLL **[estimativa]**. É exatamente por isso que o teste misto, anotado à mão, precisa existir: ele é o único lugar onde se mede se a atribuição aprendida está certa.

**Variante PRODEN (opcional).** Em vez da log-verossimilhança marginal, usar como alvo a distribuição *posterior* sobre as atribuições válidas, calculada com o próprio modelo e sem gradiente, e aplicar entropia cruzada. É mais estável no início se houver aquecimento com os convertidos. Recomendo a versão marginal acima por ser mais simples de explicar.

### 3.3 Experimentos a reportar (notebook 05)
| Braço | Treino | Pergunta |
|---|---|---|
| A (atual) | convertidos | linha de base |
| B | convertidos + mistos com perda de rótulo parcial | os mistos ajudam no teste misto sem prejudicar o principal? |
| C | convertidos + mistos desambiguados pelo LLM **com restrição** | supervisão fraca restrita contra rótulo parcial (conversa com a pergunta central do projeto) |

Avaliar todos nos dois testes, o principal e o misto, com IC por bootstrap.

---

## 4. Desenho do teste misto

### 4.1 População e amostra
- **População**: os 976 casos mistos com vários tópicos. Os de um tópico não precisam de desambiguação de atribuição.
- **Tamanho**: **150**, por amostragem aleatória estratificada por número de tópicos (2 / ≥ 3). Com acurácia de atribuição em torno de 70%, n = 150 dá IC95% de Wilson de cerca de **62% a 77%** (± 7–8 p.p.); com n = 100 seriam ± 9 p.p. (60%–78%). A correção para população finita (N = 976) reduz pouco: o n necessário para ± 7 p.p. com p = 0,25 cai de 147 para 128.
- **Poder estatístico**: com 150 itens, comparações *pareadas* entre modelos só detectam diferenças grandes. [Card et al. (2020)](https://aclanthology.org/2020.emnlp-main.745/) mostram que conjuntos pequenos deixam a maioria das comparações em NLP subdimensionadas. Por isso, no teste misto, reportar ICs e evitar alegar superioridade em diferenças de poucos pontos.
- **Exclusão**: os 150 saem do conjunto usado no braço B/C de treino, para não haver vazamento.

### 4.2 Protocolo de anotação para conter o viés
1. **Mostrar título e corpo**, porque foi isso que o RePro anotou. **Esconder** a nota em estrelas (o guia já diz isso), as predições de modelos e qualquer saída de LLM.
2. Os tópicos do RePro vêm **pré-preenchidos como "mencionados"**. O anotador escolhe, para cada um, entre `positivo / negativo / neutro`, e só pode mudar um tópico com justificativa na coluna `obs`.
3. Colunas extras: `conflito_interno` (aspecto com elogio e crítica, indicando o dominante quando houver, como no SemEval-2016 SB2), `so_no_titulo` (a polaridade só aparece no título) e `topico_duvidoso`.
4. **Anotar antes de olhar resultados de modelos no teste misto** e versionar o CSV com data, numa espécie de pré-registro. Isso evita ajustar o gabarito ao que o modelo prevê.
5. **Segundo anotador** (colega da disciplina) em **50** dos 150, sorteados, de forma independente. Reportar o **κ de Cohen** sobre a polaridade por aspecto marcado e comparar com o κ = 0,71 do RePro ([SILVA et al., 2024](https://aclanthology.org/2024.propor-1.44/)). Para a interpretação, seguir [Artstein e Poesio (2008)](https://aclanthology.org/J08-4004/), que discutem os limites dos rótulos verbais de κ. Discordâncias: resolver por discussão e registrar quantas foram.

### 4.3 LLM para propor a atribuição? Não no teste.
- [Schroeder, Roy e Kabbara (2025)](https://aclanthology.org/2025.findings-acl.1323/) testaram em experimento pré-registrado, com 410 anotadores segundo o [arXiv](https://arxiv.org/abs/2507.15821), sugestões de LLM em tarefas subjetivas. As sugestões **não aceleraram** a anotação, **aumentaram a confiança** dos anotadores, que **adotaram fortemente as sugestões** (mudando a distribuição de rótulos), e, quando esse gabarito é usado para avaliar LLMs, **o desempenho medido sobe significativamente**.
- Na anotação sintática com pré-anotação por *parser*, [Mikulová et al. (2022)](https://arxiv.org/abs/2306.09307) relatam ganho de velocidade e de consistência e discutem o viés de pré-anotação. [Fort e Sagot (2010)](https://aclanthology.org/W10-1807/) estudam o mesmo efeito em POS. Lá a tarefa é objetiva e o pré-anotador é muito preciso, ao contrário do caso aqui.
- **Consequência para este projeto**: a pergunta central é "rótulos de LLM chegam perto de rótulos humanos?". Um gabarito misto pré-anotado por LLM **favoreceria o próprio LLM e os modelos treinados com rótulos de LLM**, contaminando a comparação principal. O LLM só entra no **treino** (braço C). No teste, anotação humana cega.
- A validação humana das saídas de LLM, por tarefa, é a recomendação geral da literatura ([PANGAKIS; WOLKEN; FASCHING, 2023](https://arxiv.org/abs/2306.00176)), e o notebook 04 já faz isso no teste principal.

### 4.4 Métricas do teste misto
- F1 de tupla e F1 de polaridade só nos aspectos mencionados (`tuple_f1` e `_polarity_f1` de `metrics.py`, sem alteração);
- **acurácia de atribuição**: todos os tópicos marcados com a polaridade certa, com IC de Wilson;
- **coerência de conjunto, métrica gratuita com n grande**: em **todos** os 976 casos mistos com vários tópicos (menos os usados no treino), verificar se as polaridades previstas nos tópicos-ouro contêm positivo **e** negativo. É uma condição necessária, não suficiente, que não exige anotação nova e tem IC estreito. Pode ser calculada já para os modelos que existem (Regressão Logística) e mede diretamente a "degeneração para sentimento da frase" descrita no MAMS.

---

## 5. Auditoria da conversão

### 5.1 O que pode estar errado nos 7.985 convertidos
A conversão assume que, se a avaliação é só POSITIVO, todo tópico marcado é positivo. Ela falha quando:
(a) um tópico é mencionado sem opinião ("recebi ontem; produto ótimo" faz `entrega` virar positivo quando deveria ser neutro);
(b) o tópico foi marcado "na dúvida" pela regra do RePro de marcar ambos;
(c) a polaridade está só no título, que o modelo não lê.
Os erros (a) e (b) se concentram nos **701 casos de vários tópicos do teste**.

### 5.2 Tamanho e estratificação
- **Erro definido por avaliação**: pelo menos um aspecto cuja polaridade verdadeira difere da propagada. Registrar também o erro por aspecto e o tipo (a/b/c).
- **n = 100 avaliações do teste**, estratificadas: **70 com vários tópicos** e **30 com um tópico**. Estimar a taxa global ponderando pelos pesos dos estratos (701/1.597 e 896/1.597).
- Por que não 50: com 5 erros em 50 (10%), o IC95% de Wilson é **4,3%–21,4%** (Clopper-Pearson: 3,3%–21,8%), largo demais para concluir algo. Com 10 em 100: **5,5%–17,4%**. Para ± 5 p.p. com p ≈ 10% seriam cerca de 139 casos (122 com correção para N finito), um custo que não compensa.
- **Intervalo**: Wilson, recomendado por [Brown, Cai e DasGupta (2001)](https://projecteuclid.org/euclid.ss/1009213286) para n pequeno no lugar do Wald. Clopper-Pearson pode ir como versão conservadora. Se houver zero erros, usar o limite superior (com n = 50 e 0 erros, ele ainda é 7,1%).

### 5.3 Como reportar
Tabela com: estrato, n auditado, erros, taxa, IC95% de Wilson, tipos (a/b/c). Depois, uma frase de sensibilidade: "se x% dos rótulos do teste estão errados, diferenças entre modelos menores que ~x p.p. em F1 de tupla não são interpretáveis". Opcional: recalcular o F1 dos modelos só no subconjunto auditado, com o rótulo corrigido contra o convertido, para mostrar o efeito.

---

## 6. Plano concreto (ordem, esforço e arquivos)

| # | Ação | Arquivos | Esforço | Impacto esperado |
|---|---|---|---|---|
| 1 | **Entrada = título + corpo** no RePro (`f"{titulo}. {texto}"`), mantendo `texto` para a Olist ou usando o título dela quando existir (`data.py` já o expõe) | `repro.py` (nova coluna `entrada`), notebooks 03/04/05/07/08 | 0,5 h + rerodar | Elimina a incoerência entre rótulo e entrada, sobretudo nos mistos. É barato e deve ganhar alguns pontos **[estimativa]** |
| 2 | Manter os descartados com tópicos e conjunto de polaridades (colunas `topicos_ouro`, `pol_conjunto`) e exportar `data/annotated/repro_misto_para_anotar.csv` (150 casos estratificados, com `seed`) | `repro.py`, `sampling.py`, `data/annotated/LEIAME.md` | 1 h | Base para os passos 3–5 |
| 3 | **Métrica de coerência de conjunto** nos 976 (sem anotação) | `metrics.py` (nova função), notebook 06 | 1 h | Primeira evidência quantitativa do problema, com n grande, já para os baselines |
| 4 | **Teste misto: 150 casos** anotados pelo aluno de forma cega; **50 com segundo anotador**, κ | guia: nova seção "Desambiguação de casos mistos do RePro" em `anotacao/guia_de_anotacao.md` | **2,5–3 h** do aluno + **1 h** do colega + 0,5 h de κ | É a principal mudança nas conclusões: passa a haver um número para "bom, mas atrasou" |
| 5 | **Auditoria de 100 convertidos** do teste (70 com vários tópicos e 30 com um) | `data/annotated/auditoria_conversao.csv`, notebook 02 | **1,5 h** | Dá a margem de interpretação de todo o teste principal |
| 6 | **Perda de rótulo parcial** (braço B) e LLM restrito (braço C) | `models/encoder.py` (`_loss` com máscara de modo), notebooks 04/05 | 3–4 h de código + 2–3 execuções no T4 | Pergunta de pesquisa extra. Bom candidato a contribuição original |
| 7 | 100 comentários da Olist (já preparados) | `data/annotated/olist_para_anotar.csv` | 1,5–2 h | Mede a mudança de domínio. Útil, mas **não** substitui o teste misto, porque não garante casos mistos |

**Total de esforço humano**: cerca de **6–7 h do aluno** (passos 4, 5 e 7) mais **1 h de um colega**. Se faltar tempo, cortar nesta ordem: 7, depois o braço C, depois o braço B. Os passos 1, 3, 4 e 5 são o núcleo.

**Impacto esperado nas conclusões [estimativa]**. O teste principal continua sendo o número comparável com o RePro. O teste misto deve mostrar queda forte em todos os modelos, pelo padrão do MAMS, e provavelmente maior nos modelos de saco de palavras e de regras, que tendem a propagar o sentimento global. Se o braço B reduzir essa queda sem perder no teste principal, o trabalho ganha um resultado próprio. Se a auditoria mostrar erro de conversão acima de cerca de 10%, as diferenças pequenas entre modelos na tabela atual (por exemplo, entre BERTimbau e Albertina) devem ser apresentadas como empate técnico.

**Como reportar no artigo final**:
(i) a tabela atual passa a se chamar "Teste RePro (polaridade única)";
(ii) uma nova tabela "Teste misto (n = 150, desambiguado à mão, κ = …)";
(iii) uma linha de coerência de conjunto com n ≈ 826–976;
(iv) uma caixa de limitações com a taxa de erro da conversão e o IC;
(v) no guia e no README, trocar "A amostra da Olist anotada à mão cobre esses casos" por uma referência ao teste misto.

---

## 7. Riscos

| Risco | Mitigação |
|---|---|
| O aluno é anotador e autor: viés de confirmação | Anotar antes de rodar modelos no teste misto, versionar o CSV, segundo anotador em 50 e reportar o κ |
| κ baixo (abaixo de cerca de 0,6) na desambiguação | Reportar assim mesmo, como limite superior realista. Analisar discordâncias por tipo (conflito interno, título) e não "consertar" o gabarito até o κ subir |
| A desambiguação herda erros de tópico do RePro | Permitir mudar o tópico com justificativa na `obs` e reportar a taxa de tópicos questionados |
| Conflito interno frequente em `produto` torna a atribuição indecidível | Coluna `conflito_interno` mais regra do dominante (SemEval-2016). Itens indecidíveis saem da acurácia e são contados à parte |
| Perda de rótulo parcial aprende atalhos posicionais | Só reportar ganho se ele aparecer no teste misto humano. Fazer análise de erros com LIME (notebook 08) em 10 casos |
| Comparações no teste misto sem poder estatístico | Reportar IC e tamanho do efeito. Não usar McNemar com n = 150 para conclusões fortes ([CARD et al., 2020](https://aclanthology.org/2020.emnlp-main.745/)) |
| Mudar a entrada para título + corpo altera os resultados já publicados no README | Rerodar os baselines (CPU, minutos) e registrar a mudança |
| Vazamento entre teste misto e treino com mistos | Excluir os 150 dos braços B e C, com uma checagem por `review_id` em teste automatizado |

---

## Referências (ABNT)

ANGELIDIS, S.; LAPATA, M. Multiple instance learning networks for fine-grained sentiment analysis. **Transactions of the Association for Computational Linguistics**, v. 6, p. 17–31, 2018. Disponível em: https://aclanthology.org/Q18-1002/. Acesso em: 30 out. 2026.

ARTSTEIN, R.; POESIO, M. Inter-coder agreement for computational linguistics. **Computational Linguistics**, v. 34, n. 4, p. 555–596, 2008. Disponível em: https://aclanthology.org/J08-4004/. Acesso em: 30 out. 2026.

BROWN, L. D.; CAI, T. T.; DASGUPTA, A. Interval estimation for a binomial proportion. **Statistical Science**, v. 16, n. 2, p. 101–133, 2001. DOI: 10.1214/ss/1009213286. Disponível em: https://projecteuclid.org/euclid.ss/1009213286. Acesso em: 30 out. 2026.

CAI, H.; XIA, R.; YU, J. Aspect-category-opinion-sentiment quadruple extraction with implicit aspects and opinions. In: ANNUAL MEETING OF THE ACL AND IJCNLP, 59., 2021. **Proceedings** […]. [S. l.]: ACL, 2021. p. 340–350. Disponível em: https://aclanthology.org/2021.acl-long.29/. Acesso em: 30 out. 2026.

CARD, D. et al. With little power comes great responsibility. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING (EMNLP), 2020. **Proceedings** […]. [S. l.]: ACL, 2020. p. 9263–9274. Disponível em: https://aclanthology.org/2020.emnlp-main.745/. Acesso em: 30 out. 2026.

CHEBOLU, S. U. S. et al. Survey of aspect-based sentiment analysis datasets. **arXiv**, 2204.05232, 2022. Disponível em: https://arxiv.org/abs/2204.05232. Acesso em: 30 out. 2026. [autoria não conferida na página; não verificado]

COUR, T.; SAPP, B.; TASKAR, B. Learning from partial labels. **Journal of Machine Learning Research**, v. 12, p. 1501–1536, 2011. Disponível em: https://jmlr.org/papers/v12/cour11a.html. Acesso em: 30 out. 2026.

FENG, L. et al. Provably consistent partial-label learning. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS (NeurIPS), 33., 2020. **Proceedings** […]. 2020. p. 10948–10960. Disponível em: https://arxiv.org/abs/2007.08929. Acesso em: 30 out. 2026.

FORT, K.; SAGOT, B. Influence of pre-annotation on POS-tagged corpus development. In: LINGUISTIC ANNOTATION WORKSHOP, 4., 2010, Uppsala. **Proceedings** […]. [S. l.]: ACL, 2010. p. 56–63. Disponível em: https://aclanthology.org/W10-1807/. Acesso em: 30 out. 2026.

HE, R. et al. Exploiting document knowledge for aspect-level sentiment classification. In: ANNUAL MEETING OF THE ACL, 56., 2018, Melbourne. **Proceedings** […]. [S. l.]: ACL, 2018. p. 579–585. Disponível em: https://aclanthology.org/P18-2092/. Acesso em: 30 out. 2026.

JIANG, Q. et al. A challenge dataset and effective models for aspect-based sentiment analysis. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING AND IJCNLP (EMNLP-IJCNLP), 2019, Hong Kong. **Proceedings** […]. [S. l.]: ACL, 2019. p. 6280–6285. Disponível em: https://aclanthology.org/D19-1654/. Acesso em: 30 out. 2026.

LV, J. et al. Progressive identification of true labels for partial-label learning. In: INTERNATIONAL CONFERENCE ON MACHINE LEARNING (ICML), 37., 2020. **Proceedings of Machine Learning Research**, v. 119, p. 6500–6510, 2020. Disponível em: https://proceedings.mlr.press/v119/lv20a.html. Acesso em: 30 out. 2026.

MIKULOVÁ, M. et al. Quality and efficiency of manual annotation: pre-annotation bias. In: LANGUAGE RESOURCES AND EVALUATION CONFERENCE (LREC), 2022. **Proceedings** […]. 2022. Disponível em: https://arxiv.org/abs/2306.09307. Acesso em: 30 out. 2026.

PANGAKIS, N.; WOLKEN, S.; FASCHING, N. Automated annotation with generative AI requires validation. **arXiv**, 2306.00176, 2023. Disponível em: https://arxiv.org/abs/2306.00176. Acesso em: 30 out. 2026.

PAPPAS, N.; POPESCU-BELIS, A. Explicit document modeling through weighted multiple-instance learning. **Journal of Artificial Intelligence Research**, v. 58, 2017. Disponível em: https://jair.org/index.php/jair/article/view/11051. Acesso em: 30 out. 2026.

PONTIKI, M. et al. SemEval-2014 Task 4: aspect based sentiment analysis. In: INTERNATIONAL WORKSHOP ON SEMANTIC EVALUATION (SemEval), 8., 2014, Dublin. **Proceedings** […]. [S. l.]: ACL, 2014. p. 27–35. Disponível em: https://aclanthology.org/S14-2004/. Acesso em: 30 out. 2026.

PONTIKI, M. et al. SemEval-2015 Task 12: aspect based sentiment analysis. In: INTERNATIONAL WORKSHOP ON SEMANTIC EVALUATION (SemEval), 9., 2015, Denver. **Proceedings** […]. [S. l.]: ACL, 2015. Disponível em: https://aclanthology.org/S15-2082/. Acesso em: 30 out. 2026.

PONTIKI, M. et al. SemEval-2016 Task 5: aspect based sentiment analysis. In: INTERNATIONAL WORKSHOP ON SEMANTIC EVALUATION (SemEval), 10., 2016, San Diego. **Proceedings** […]. [S. l.]: ACL, 2016. p. 19–30. Disponível em: https://aclanthology.org/S16-1002/. Acesso em: 30 out. 2026.

SCHROEDER, H.; ROY, D.; KABBARA, J. Just put a human in the loop? Investigating LLM-assisted annotation for subjective tasks. In: **Findings of the Association for Computational Linguistics: ACL 2025**. [S. l.]: ACL, 2025. Disponível em: https://aclanthology.org/2025.findings-acl.1323/. Acesso em: 30 out. 2026.

SILVA, L. N. dos S. et al. RePro: a benchmark dataset for opinion mining in Brazilian Portuguese. In: INTERNATIONAL CONFERENCE ON COMPUTATIONAL PROCESSING OF PORTUGUESE (PROPOR), 16., 2024, Santiago de Compostela. **Proceedings** […]. [S. l.]: ACL, 2024. p. 432–440. Disponível em: https://aclanthology.org/2024.propor-1.44/. Acesso em: 30 out. 2026.

ZHANG, W. et al. Aspect sentiment quad prediction as paraphrase generation. In: CONFERENCE ON EMPIRICAL METHODS IN NATURAL LANGUAGE PROCESSING (EMNLP), 2021. **Proceedings** […]. [S. l.]: ACL, 2021. p. 9209–9219. Disponível em: https://aclanthology.org/2021.emnlp-main.726/. Acesso em: 30 out. 2026.
