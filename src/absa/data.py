"""Download dos CSVs da Olist e montagem da tabela de comentários.

A tabela final tem uma linha por avaliação com comentário escrito, já cruzada
com o pedido (datas de entrega e atraso), o primeiro item (categoria do
produto) e o cliente (UF).
"""
import urllib.request

import pandas as pd

from .config import INTERIM, RAW

# Espelho oficial da Olist no GitHub (mesmos arquivos publicados no Kaggle)
OLIST_URL = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"
OLIST_FILES = [
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_customers_dataset.csv",
    "product_category_name_translation.csv",
]

REGIOES = {
    "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MT", "MS"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}
UF2REGIAO = {uf: reg for reg, ufs in REGIOES.items() for uf in ufs}


def download_olist(dest=RAW, force=False):
    """Baixa os CSVs necessários para `dest` (pula os que já existem)."""
    dest.mkdir(parents=True, exist_ok=True)
    for name in OLIST_FILES:
        path = dest / name
        if path.exists() and not force:
            continue
        print(f"baixando {name}...")
        tmp = path.with_suffix(".part")  # evita deixar arquivo pela metade se cair
        with urllib.request.urlopen(OLIST_URL + name, timeout=60) as resp, tmp.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        tmp.rename(path)
    return dest


def load_raw(src=RAW):
    return {name.removesuffix(".csv"): pd.read_csv(src / name) for name in OLIST_FILES}


def build_reviews_table(raw):
    """Junta as tabelas e devolve uma linha por comentário com texto."""
    reviews = raw["olist_order_reviews_dataset"].copy()
    reviews["texto"] = reviews.review_comment_message.fillna("").str.strip()
    reviews = reviews[reviews.texto != ""]
    # Há review_id repetidos ligados a pedidos diferentes; o texto é o mesmo
    reviews = reviews.drop_duplicates("review_id")

    orders = raw["olist_orders_dataset"]
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col])

    items = raw["olist_order_items_dataset"]
    first_item = (
        items.sort_values("order_item_id")
        .groupby("order_id")
        .agg(product_id=("product_id", "first"), n_itens=("order_item_id", "count"),
             n_vendedores=("seller_id", "nunique"), preco_total=("price", "sum"))
        .reset_index()
    )
    products = raw["olist_products_dataset"][["product_id", "product_category_name"]]
    customers = raw["olist_customers_dataset"][["customer_id", "customer_state"]]

    df = (
        reviews.merge(orders, on="order_id", how="left")
        .merge(first_item, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
        .merge(customers, on="customer_id", how="left")
    )

    entregue = df.order_delivered_customer_date.notna()
    # A data estimada vem sem horário: só é atraso a partir do dia seguinte
    df["atraso_dias"] = (df.order_delivered_customer_date - df.order_estimated_delivery_date).dt.days
    df["entregue"] = entregue
    df["atrasado"] = entregue & (df.atraso_dias > 0)
    df["regiao"] = df.customer_state.map(UF2REGIAO)
    df["n_palavras"] = df.texto.str.split().str.len()

    cols = [
        "review_id", "order_id", "texto", "review_comment_title", "review_score",
        "review_creation_date", "order_status", "entregue", "atrasado", "atraso_dias",
        "product_category_name", "n_itens", "n_vendedores", "preco_total",
        "customer_state", "regiao", "n_palavras",
    ]
    return df[cols].rename(columns={
        "review_score": "nota",
        "product_category_name": "categoria",
        "customer_state": "uf",
        "review_comment_title": "titulo",
    }).reset_index(drop=True)


def load_reviews(rebuild=False):
    """Atalho: baixa (se preciso), monta e guarda em data/interim/reviews.csv.gz."""
    out = INTERIM / "reviews.csv.gz"
    if out.exists() and not rebuild:
        return pd.read_csv(out, parse_dates=["review_creation_date"])
    download_olist()
    df = build_reviews_table(load_raw())
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
