"""Constantes do projeto: caminhos, taxonomia de aspectos e codificação dos rótulos."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
ANNOTATED = DATA / "annotated"
LLM_CACHE = DATA / "llm_cache"

SEED = 42

# Taxonomia (adaptada do RePro, Silva et al., 2024). A definição de cada
# aspecto, com exemplos, está em anotacao/guia_de_anotacao.md.
ASPECTS = ["produto", "entrega", "recebimento", "anuncio", "atendimento"]

ASPECT_DESCRIPTIONS = {
    "produto": "qualidade, funcionamento, durabilidade, tamanho ou aparência do item",
    "entrega": "prazo, atraso, rapidez, rastreio ou transportadora",
    "recebimento": "estado em que o pedido chegou: avaria, embalagem, item faltando ou errado",
    "anuncio": "fidelidade do produto à descrição, às fotos ou ao preço anunciado",
    "atendimento": "contato com vendedor ou loja, suporte, troca, devolução ou reembolso",
}

# Cada aspecto recebe exatamente uma destas classes
POLARITIES = ["ausente", "positivo", "negativo", "neutro"]
POL2ID = {p: i for i, p in enumerate(POLARITIES)}
ID2POL = dict(enumerate(POLARITIES))

LABEL_COLUMNS = [f"asp_{a}" for a in ASPECTS]
