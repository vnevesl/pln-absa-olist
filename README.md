# Do que o cliente reclama? ABSA em avaliações de e-commerce

Trabalho Prático Final da disciplina de **Processamento de Linguagem Natural (2026/2)**. O pré-projeto (Etapa 1) está em [`pre-projeto/`](pre-projeto/).

**Autor:** Vitor Neves de Lima

## Sobre o projeto

Quando um cliente dá uma nota baixa numa compra online, a loja sabe que ele ficou insatisfeito, mas não sabe **por quê**. Este projeto propõe usar **Análise de Sentimentos Baseada em Aspectos (ABSA)** para identificar, em cada comentário, **sobre o que** o cliente fala e com qual polaridade:

> *"O produto é ótimo, mas chegou dez dias atrasado e a caixa veio amassada."*
> → **Produto:** positivo · **Entrega:** negativo · **Condições de recebimento:** negativo

### Proposta em resumo

| Item | Escolha |
|---|---|
| Gabarito humano | [RePro](https://aclanthology.org/2024.propor-1.44/) (PROPOR 2024): avaliações da Americanas/B2W anotadas por humanos. 7.985 usáveis → treino 5.589 · validação 799 · **teste 1.597** |
| Domínio de aplicação | [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0): 40.641 comentários com datas de entrega |
| Paradigma | Aprendizado supervisionado (rótulos humanos) comparado a supervisão fraca (rótulos de LLM) |
| Tarefa | Classificação multirrótulo por categoria de aspecto (ACSA): 5 aspectos × {ausente, positivo, negativo, neutro} |
| Modelos | Regras → TF-IDF + Regressão Logística/Random Forest → BERTimbau; rótulos do Qwen2.5-7B (4 bits) para a comparação humano × LLM |
| Pergunta central | Treinar com rótulos de LLM, nos mesmos textos, chega perto de treinar com rótulos humanos? |
| Avaliação | F1 macro por aspecto e F1 de tupla com IC por bootstrap, bootstrap pareado e McNemar, sempre no teste humano |
| Validação extrínseca | Na Olist: "entrega negativa" prevista × atraso real registrado nos pedidos |
| IA responsável | LIME (explicabilidade) e Fairlearn (gênero, região e idade no RePro; região na Olist) |
| Ambiente | Google Colab gratuito (T4) |

> **Mudança em relação ao pré-projeto:** a anotação própria de 800 comentários foi substituída pelo gabarito humano publicado do RePro (mais escala e comparabilidade com a literatura), e o aspecto "atendimento" virou "outros" para seguir a taxonomia do RePro. Uma amostra opcional de 100 comentários da Olist, anotada à mão, mede a mudança de domínio.

### Resultados (teste humano do RePro, n = 1.597)

Métrica principal: F1 de pares (aspecto, polaridade). Entrada dos modelos: título + corpo do comentário.

| Modelo | F1 de pares (IC95%) | F1 macro |
|---|---|---|
| Regras (palavras-chave) | 0,654 (0,636–0,672) | 0,326 |
| TF-IDF + Random Forest | 0,815 (0,801–0,829) | 0,518 |
| TF-IDF + Regressão Logística | 0,867 (0,856–0,878) | 0,764 |
| **BERTimbau, rótulos humanos (3 sementes)** | **0,918 ± 0,002** (0,908–0,927) | **0,848** |
| BERTimbau, só a validação humana (controle) | 0,779 (0,766–0,792) | 0,429 |
| BERTimbau, rótulos do Qwen2.5-7B (3 sementes) | 0,794 ± 0,011 (0,779–0,802) | 0,594 |

H2: os rótulos do Qwen concordam com os humanos em F1 de pares 0,772 (validação), e o BERTimbau treinado com eles
recupera só 10,8% da distância entre o controle de 599 rótulos humanos e o modelo humano completo — a H2 não se confirma.

Na Olist (metade "relato", 19.239 pedidos entregues), a probabilidade prevista de "entrega negativa" acrescenta
informação à nota para detectar atraso real: AUC de 0,817 (só a nota) para 0,859 (nota + BERTimbau).

### Números verificados nos dados

Reproduzíveis com [`scripts/estatisticas_olist.py`](scripts/estatisticas_olist.py):

- 99.224 avaliações, das quais **40.950 têm comentário** (36.155 textos distintos)
- Comentários curtos: média de 11,7 palavras, mediana 9, p95 33
- Notas entre os comentários: 50,1% de 5★, 14,6% de 4★, 8,7% de 3★, 5,2% de 2★, 21,4% de 1★
- **73,0%** dos comentários de pedidos atrasados têm nota ≤ 2, contra **18,5%** nos pedidos entregues no prazo

## Estrutura do repositório

```
src/absa/                     pacote Python usado pelos notebooks
  config.py                   caminhos, taxonomia dos 5 aspectos, rótulos
  repro.py                    gabarito humano: download, conversão e divisão do RePro
  data.py                     download da Olist e junção das tabelas
  preprocess.py               limpeza leve (Transformers/LLM) e clássica (TF-IDF)
  heuristics.py               regras por palavra-chave (linha de base)
  llm_labeler.py              rotulagem por LLM: prompt few-shot, JSON, cache
  metrics.py                  F1 por aspecto e de tupla, bootstrap, McNemar, AUC de atraso
  sampling.py                 amostra opcional da Olist para anotação
  models/baseline.py          TF-IDF + Regressão Logística / Random Forest
  models/encoder.py           BERTimbau / Albertina com uma cabeça por aspecto
notebooks/                    roteiro de execução (Colab, GPU T4 gratuita)
  experimentos.ipynb          H1, H2 e H3 num único notebook (BERTimbau, Qwen2.5-7B e validação na Olist)
artigo/                       artigo em LaTeX (Overleaf) e referências
results/                      predições e resumos gerados pelos notebooks (base das tabelas do artigo)
anotacao/guia_de_anotacao.md  definição dos aspectos e conversão do RePro
data/annotated/               amostra opcional da Olist para anotação manual
tests/                        testes do pacote (pytest)
pre-projeto/                  PDF entregue e fonte HTML
pesquisa/                     levantamento do estado da arte
scripts/                      números do artigo (numeros_artigo.py, resultados_bertimbau.py)
docs/                         roteiro e decisões de projeto
```

## Como executar

**No Colab:** abra `notebooks/experimentos.ipynb` pelo GitHub (Arquivo → Abrir notebook → GitHub → `vnevesl/pln-absa-olist`), escolha
**GPU T4** e clique em **Executar tudo**. A primeira célula clona o repositório, instala o pacote e salva os
resultados no Google Drive; se a sessão cair, rodar de novo continua de onde parou.

O notebook roda H1 e H3 primeiro (~20 min) e depois a H2 (~2–3 h, quase tudo na rotulagem pelo Qwen).

As tabelas do artigo saem de `scripts/numeros_artigo.py` (linhas de base, CPU) e `scripts/resultados_bertimbau.py`
(lê `results/encoders/` e `results/h2/`).

**Localmente** (testes e linhas de base, sem GPU):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Principais referências

- SILVA, L. N. dos S. et al. RePro: a benchmark dataset for opinion mining in Brazilian Portuguese. PROPOR, 2024.
- SILVA, L. N. dos S.; REAL, L. Automated topic annotation in Brazilian product reviews with Sabiá-3. STIL, 2024.
- SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: pretrained BERT models for Brazilian Portuguese. BRACIS, 2020.
- GILARDI, F.; ALIZADEH, M.; KUBLI, M. ChatGPT outperforms crowd workers for text-annotation tasks. PNAS, 2023.
- ZHANG, W. et al. A survey on aspect-based sentiment analysis: tasks, methods, and challenges. 2022.

A lista completa está no PDF e em [`pesquisa/estado_da_arte_absa_olist.md`](pesquisa/estado_da_arte_absa_olist.md).

## Licença dos dados

Os dados da Olist (CC BY-NC-SA 4.0) e do RePro não são redistribuídos aqui; o código baixa os arquivos das fontes públicas. A exceção são os 100 comentários da Olist em `data/annotated/`, mantidos sob a mesma licença não comercial.
