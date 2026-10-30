# Amostra da Olist para anotação manual (opcional)

O gabarito principal é o RePro (baixado por `src/absa/repro.py`). Esta pasta guarda uma amostra de 100
comentários da Olist para medir a mudança de domínio e cobrir avaliações mistas.

| Arquivo | Conteúdo |
|---|---|
| `olist_para_anotar.csv` | 100 comentários (estratificados por nota, 30% de pedidos atrasados), colunas `asp_*` a preencher |
| `olist_anotado.csv` | o mesmo arquivo depois de anotado (criar ao terminar) |
| `olist_metadados.csv` | nota, atraso, UF e categoria de cada comentário (não usar durante a anotação) |

Valores aceitos em cada coluna `asp_*`: `ausente`, `positivo`, `negativo`, `neutro`.
Regras em [`anotacao/guia_de_anotacao.md`](../../anotacao/guia_de_anotacao.md).

Os textos vêm do *Brazilian E-Commerce Public Dataset by Olist* (CC BY-NC-SA 4.0) e mantêm essa licença:
uso não comercial, com atribuição.
