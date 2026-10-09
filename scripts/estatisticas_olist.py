"""Estatísticas descritivas das avaliações da Olist usadas no pré-projeto.

Baixa os CSVs públicos de avaliações e pedidos e reproduz os números citados
no texto: volume de avaliações, comentários com texto, distribuição de notas,
tamanho dos comentários e relação entre atraso na entrega e nota baixa.

Uso:
    pip install pandas
    python scripts/estatisticas_olist.py
"""
import pandas as pd

BASE = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"

reviews = pd.read_csv(BASE + "olist_order_reviews_dataset.csv")
orders = pd.read_csv(BASE + "olist_orders_dataset.csv")

print("Avaliações:", len(reviews), "| review_id únicos:", reviews.review_id.nunique())
print("Notas (todas):", reviews.review_score.value_counts().sort_index().to_dict())

texto = reviews.review_comment_message.fillna("").str.strip()
com_texto = reviews[texto != ""]
print("Com comentário:", len(com_texto), "| textos distintos:", com_texto.review_comment_message.nunique())
dist = com_texto.review_score.value_counts(normalize=True).sort_index() * 100
print("Notas (com comentário, %):", dist.round(1).to_dict())

palavras = com_texto.review_comment_message.str.split().str.len()
print(f"Palavras por comentário: média {palavras.mean():.1f}, mediana {palavras.median():.0f}, p95 {palavras.quantile(.95):.0f}")

# Atraso em dias = entrega real - data estimada (somente pedidos entregues).
# A data estimada vem sem horário (00:00), então só conta como atrasado o pedido
# entregue a partir do dia seguinte ao prometido; entregas no próprio dia são pontuais.
m = com_texto.merge(orders, on="order_id")
m = m[m.order_delivered_customer_date.notna()]
atraso_dias = (pd.to_datetime(m.order_delivered_customer_date) - pd.to_datetime(m.order_estimated_delivery_date)).dt.days
atrasado = atraso_dias > 0
print("Comentários de pedidos entregues:", len(m), f"| atrasados: {atrasado.sum()} ({100 * atrasado.mean():.1f}%)")
print(f"Nota <= 2: {100 * (m[atrasado].review_score <= 2).mean():.1f}% nos atrasados vs "
      f"{100 * (m[~atrasado].review_score <= 2).mean():.1f}% nos pontuais")
