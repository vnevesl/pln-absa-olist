"""Pré-processamento em duas versões.

- `clean_light`: para Transformers e LLM. Mantém maiúsculas, pontuação e
  stopwords; só remove ruído que o tokenizador não precisa ver.
- `clean_classic`: para a linha de base TF-IDF. Minúsculas, sem acentos
  repetidos, stopwords removidas preservando negações, lematização opcional.
"""
import re

from unidecode import unidecode

# A Olist trocou nomes de lojas/parceiros nos textos por casas de Game of Thrones
GOT_HOUSES = [
    "stark", "targaryen", "lannister", "baratheon", "tyrell", "martell",
    "greyjoy", "arryn", "tully", "bolton", "frey", "mormont",
]
_GOT_RE = re.compile(r"\b(" + "|".join(GOT_HOUSES) + r")\b", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_REPEAT_RE = re.compile(r"(\w)\1{2,}")  # "muitoooo" -> "muitoo"
_SPACES_RE = re.compile(r"\s+")

# Palavras que mudam a polaridade e NÃO podem sair junto com as stopwords
NEGATIONS = {"nao", "nunca", "jamais", "nem", "nenhum", "nenhuma", "nada", "sem"}

# Lista curta e explícita (sem acento, após unidecode), para não depender de download
STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "pelo", "pela", "pelos", "pelas", "para",
    "pra", "com", "e", "ou", "que", "se", "ao", "aos", "me", "te", "lhe", "eu", "ele",
    "ela", "eles", "elas", "voce", "voces", "meu", "minha", "meus", "minhas", "seu",
    "sua", "seus", "suas", "isso", "isto", "esse", "essa", "este", "esta", "aquele",
    "aquela", "foi", "ser", "sao", "era", "estao", "estava", "tem", "ter",
    "tinha", "ja", "mais", "muito", "tambem", "so", "como", "mas", "quando", "ate",
} - NEGATIONS


def clean_light(text: str) -> str:
    text = _URL_RE.sub(" ", text)
    text = _GOT_RE.sub("<LOJA>", text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = text.replace("\r", " ").replace("\n", " ")
    return _SPACES_RE.sub(" ", text).strip()


def clean_classic(text: str, lemmatizer=None) -> str:
    """`lemmatizer`: função opcional str -> list[str] (ex.: spaCy pt_core_news_sm)."""
    text = clean_light(text).replace("<LOJA>", " loja ")
    text = unidecode(text.lower())
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = lemmatizer(text) if lemmatizer else text.split()
    tokens = [unidecode(t.lower()) for t in tokens]
    return " ".join(t for t in tokens if t and t not in STOPWORDS)


def spacy_lemmatizer(model="pt_core_news_sm"):
    """Cria um lematizador spaCy (requer `python -m spacy download pt_core_news_sm`)."""
    import spacy

    nlp = spacy.load(model, disable=["parser", "ner"])
    return lambda text: [tok.lemma_ for tok in nlp(text)]
