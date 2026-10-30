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
| Modelos | Regras → TF-IDF + Regressão Logística/Random Forest → BERTimbau → Albertina PT-BR → LLM few-shot (Qwen2.5-7B, 4 bits) |
| Pergunta central | Treinar com rótulos de LLM, nos mesmos textos, chega perto de treinar com rótulos humanos? |
| Avaliação | F1 macro por aspecto e F1 de tupla com IC por bootstrap, bootstrap pareado e McNemar, sempre no teste humano |
| Validação extrínseca | Na Olist: "entrega negativa" prevista × atraso real registrado nos pedidos |
| IA responsável | LIME (explicabilidade) e Fairlearn (gênero, região e idade no RePro; região na Olist) |
| Ambiente | Google Colab gratuito (T4) |

> **Mudança em relação ao pré-projeto:** a anotação própria de 800 comentários foi substituída pelo gabarito humano publicado do RePro (mais escala e comparabilidade com a literatura), e o aspecto "atendimento" virou "outros" para seguir a taxonomia do RePro. Uma amostra opcional de 100 comentários da Olist, anotada à mão, mede a mudança de domínio.

### Resultados até agora (teste humano do RePro, n = 1.597)

| Modelo | F1 macro (IC95%) | F1 tupla |
|---|---|---|
| Regras (palavras-chave) | 0,442 (0,432–0,453) | 0,586 |
| TF-IDF + Random Forest | 0,612 (0,585–0,639) | 0,787 |
| TF-IDF + Regressão Logística | 0,759 (0,730–0,783) | 0,835 |
| BERTimbau, Albertina, LLM | *a rodar no Colab (notebooks 04 e 05)* | |

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
notebooks/                    roteiro de execução (Colab)
  01_eda_olist                exploração da Olist e números do pré-projeto
  02_gabarito_repro           conversão do RePro e distribuição dos rótulos
  03_baselines                regras e TF-IDF (CPU)
  04_rotulagem_llm            Qwen2.5-7B: concordância com humanos e rótulos fracos (GPU)
  05_encoders                 BERTimbau e Albertina com rótulos humanos × LLM (GPU)
  06_comparacao               tabela final, IC, bootstrap pareado, McNemar
  07_validacao_olist          aplicação na Olist e validação com o atraso real (GPU)
  08_xai_fairness             LIME e Fairlearn (GPU)
anotacao/guia_de_anotacao.md  definição dos aspectos e conversão do RePro
data/annotated/               amostra opcional da Olist para anotação manual
tests/                        testes do pacote (pytest)
pre-projeto/                  PDF entregue e fonte HTML
pesquisa/                     levantamento do estado da arte
scripts/                      estatísticas do pré-projeto
```

## Como executar

**No Colab:** abra um notebook pelo GitHub (Arquivo → Abrir notebook → GitHub → `vnevesl/pln-absa-olist`). A primeira célula clona o repositório, instala o pacote e liga cache, modelos e predições ao Google Drive. Rode na ordem: 02 → 03 → 04 → 05 → 06 → 07 → 08.

**Localmente** (só os notebooks sem GPU: 01, 02, 03 e 06):

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
