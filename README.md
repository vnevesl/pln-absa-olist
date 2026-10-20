# Do que o cliente reclama? ABSA em avaliações de e-commerce

Pré-projeto da disciplina de **Processamento de Linguagem Natural (2026/2)**, Trabalho Prático Final, Etapa 1.

**Autor:** Vitor Neves de Lima

## Sobre o projeto

Quando um cliente dá uma nota baixa numa compra online, a loja sabe que ele ficou insatisfeito, mas não sabe **por quê**. Este projeto propõe usar **Análise de Sentimentos Baseada em Aspectos (ABSA)** para identificar, em cada comentário, **sobre o que** o cliente fala e com qual polaridade:

> *"O produto é ótimo, mas chegou dez dias atrasado e a caixa veio amassada."*
> → **Produto:** positivo · **Entrega:** negativo · **Condições de recebimento:** negativo

### Proposta em resumo

| Item | Escolha |
|---|---|
| Dados | [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (CC BY-NC-SA 4.0) + [RePro](https://aclanthology.org/2024.propor-1.44/) (PROPOR 2024) |
| Paradigma | Aprendizado supervisionado, com supervisão fraca |
| Tarefa | Classificação multirrótulo por categoria de aspecto (ACSA): 5 aspectos × {positivo, negativo, neutro, não mencionado} |
| Rótulos | 800 comentários anotados à mão (kappa em 200) + ~12 mil rotulados por LLM aberto (Qwen2.5-7B, 4 bits) |
| Modelos | TF-IDF + Regressão Logística/Random Forest → BERTimbau Base → Albertina PT-BR → transferência RePro→Olist → LLM few-shot como teto |
| Avaliação | F1 macro por aspecto, bootstrap, McNemar, custo de inferência |
| Diferencial | Validação extrínseca: "entrega negativa" prevista × atraso real registrado nos pedidos |
| IA responsável | LIME (explicabilidade), Fairlearn (desempenho por região e categoria), *confident learning* (ruído nota×texto) |
| Ambiente | Google Colab gratuito (T4), 3 meses |

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
  data.py                     download da Olist e junção das tabelas
  preprocess.py               limpeza leve (Transformers/LLM) e clássica (TF-IDF)
  sampling.py                 sorteio do gabarito (800) e divisão validação/teste
  heuristics.py               regras de rotulagem (baseline e sinal fraco)
  llm_labeler.py              rotulagem por LLM: prompt few-shot, JSON, cache
  metrics.py                  F1 por aspecto, F1 de tupla, bootstrap, McNemar, AUC de atraso
  models/baseline.py          TF-IDF + Regressão Logística / Random Forest
  models/encoder.py           BERTimbau / Albertina com uma cabeça por aspecto
notebooks/                    roteiro de execução (Colab)
  01_eda_e_juncao             exploração e números do pré-projeto
  02_amostragem_gabarito      gera os arquivos para anotação manual
  03_baseline_tfidf           regras e TF-IDF
  04_rotulagem_llm            Qwen2.5-7B em 4 bits: concordância e rótulos fracos
  05_encoders_...             ajuste fino de BERTimbau e Albertina
anotacao/guia_de_anotacao.md  definição dos aspectos e casos difíceis
data/annotated/               gabarito anotado à mão (versionado)
tests/                        testes do pacote (pytest)
pre-projeto/                  PDF entregue e fonte HTML
pesquisa/                     levantamento do estado da arte
scripts/                      estatísticas do pré-projeto
```

## Como executar

**No Colab:** abra um notebook pelo GitHub (Arquivo → Abrir notebook → GitHub → `vnevesl/pln-absa-olist`). A primeira célula clona o repositório, instala o pacote e liga o cache ao Google Drive.

**Localmente:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Ordem de trabalho

1. `01` e `02`: explorar os dados e gerar `data/annotated/gabarito_para_anotar.csv`
2. Anotar os 800 comentários seguindo o guia; um colega anota `segundo_anotador.csv`
3. `04`: medir a concordância do LLM na validação e rotular ~12 mil comentários
4. `03` e `05`: treinar e avaliar as linhas de base e os encoders
5. Próximos: transferência RePro→Olist, validação com atraso, LIME, Fairlearn

## Principais referências

- SILVA, L. N. dos S. et al. RePro: a benchmark dataset for opinion mining in Brazilian Portuguese. PROPOR, 2024.
- SILVA, L. N. dos S.; REAL, L. Automated topic annotation in Brazilian product reviews with Sabiá-3. STIL, 2024.
- SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: pretrained BERT models for Brazilian Portuguese. BRACIS, 2020.
- GILARDI, F.; ALIZADEH, M.; KUBLI, M. ChatGPT outperforms crowd workers for text-annotation tasks. PNAS, 2023.
- ZHANG, W. et al. A survey on aspect-based sentiment analysis: tasks, methods, and challenges. 2022.

A lista completa está no PDF e em [`pesquisa/estado_da_arte_absa_olist.md`](pesquisa/estado_da_arte_absa_olist.md).

## Licença dos dados

Os dados da Olist e da B2W/RePro estão sob licenças **não comerciais** (CC BY-NC-SA 4.0) e não são redistribuídos aqui; o script baixa os arquivos diretamente da fonte pública.
