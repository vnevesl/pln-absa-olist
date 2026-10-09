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
pre-projeto/
  Pre_Projeto_PLN_Vitor_Neves_de_Lima.pdf   texto entregue
  pre_projeto_pln.html                      fonte do PDF
pesquisa/
  estado_da_arte_absa_olist.md              levantamento de datasets, estado da arte,
                                            pipeline, desafios e referências (ABNT)
scripts/
  estatisticas_olist.py                     reproduz as estatísticas citadas no texto
```

## Principais referências

- SILVA, L. N. dos S. et al. RePro: a benchmark dataset for opinion mining in Brazilian Portuguese. PROPOR, 2024.
- SILVA, L. N. dos S.; REAL, L. Automated topic annotation in Brazilian product reviews with Sabiá-3. STIL, 2024.
- SOUZA, F.; NOGUEIRA, R.; LOTUFO, R. BERTimbau: pretrained BERT models for Brazilian Portuguese. BRACIS, 2020.
- GILARDI, F.; ALIZADEH, M.; KUBLI, M. ChatGPT outperforms crowd workers for text-annotation tasks. PNAS, 2023.
- ZHANG, W. et al. A survey on aspect-based sentiment analysis: tasks, methods, and challenges. 2022.

A lista completa está no PDF e em [`pesquisa/estado_da_arte_absa_olist.md`](pesquisa/estado_da_arte_absa_olist.md).

## Licença dos dados

Os dados da Olist e da B2W/RePro estão sob licenças **não comerciais** (CC BY-NC-SA 4.0) e não são redistribuídos aqui; o script baixa os arquivos diretamente da fonte pública.
