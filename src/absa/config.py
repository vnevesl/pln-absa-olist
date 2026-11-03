"""Constantes do projeto: caminhos, taxonomia de aspectos e codificação dos rótulos."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
ANNOTATED = DATA / "annotated"
LLM_CACHE = DATA / "llm_cache"
MODELS = ROOT / "models"

SEED = 42

# Taxonomia do RePro (Silva et al., PROPOR 2024), cujo gabarito humano é a
# régua de avaliação do projeto. Mapeamento: ENTREGA -> entrega, PRODUTO ->
# produto, CONDICOESDERECEBIMENTO -> recebimento, ANUNCIO -> anuncio,
# OUTROS -> outros. Detalhes em anotacao/guia_de_anotacao.md.
ASPECTS = ["produto", "entrega", "recebimento", "anuncio", "outros"]

ASPECT_DESCRIPTIONS = {
    "produto": "qualidade, funcionamento, durabilidade, aparência, custo-benefício e elogios gerais ao item",
    "entrega": "prazo, atraso, rapidez, não entrega, frete, rastreio, transportadora ou retirada",
    "recebimento": "estado em que o pedido chegou: avaria, embalagem, item faltando, trocado ou fora do esperado",
    "anuncio": "fidelidade do produto à descrição, às fotos ou às especificações anunciadas",
    "outros": "loja ou vendedor, atendimento, troca, estoque, pagamento, site e demais assuntos da compra",
}

# Versão longa, para o prompt do LLM, seguindo o guia de anotação do RePro
# (Silva; Real, STIL 2024, §3.1). Ver pesquisa/melhorias/03 §7.1.
ASPECT_GUIDE = {
    "produto": "qualidade, funcionamento, durabilidade, tamanho, aparência, originalidade, experiência de uso, "
               "custo-benefício e elogios ou críticas gerais ao item comprado",
    "entrega": "prazo, atraso, rapidez, não entrega, frete, rastreio, transportadora, retirada na loja ou "
               "entrega de produto virtual (vale-presente, código)",
    "recebimento": "condições em que o pedido chegou: avaria, embalagem, item faltando, item trocado ou "
                   "produto que não atendeu ao que o cliente esperava receber",
    "anuncio": "fidelidade à descrição, às imagens ou às especificações técnicas do anúncio",
    "outros": "loja ou vendedor, atendimento, dúvidas ao vendedor, troca e devolução, estoque, forma de "
              "pagamento, experiência de compra no site e textos sem relação com a compra",
}

# Cada aspecto recebe exatamente uma destas classes
POLARITIES = ["ausente", "positivo", "negativo", "neutro"]
POL2ID = {p: i for i, p in enumerate(POLARITIES)}
ID2POL = dict(enumerate(POLARITIES))

LABEL_COLUMNS = [f"asp_{a}" for a in ASPECTS]
