"""Funções de rotulagem heurísticas (estilo Snorkel).

Não pretendem acertar sozinhas: servem para (1) combinar com o LLM na
rotulagem fraca e (2) servir de baseline "regras" na comparação. Cada função
devolve, por aspecto, a polaridade ou None quando não tem opinião.
"""
import re

from unidecode import unidecode

from .config import ASPECTS


def _norm(text):
    return unidecode(text.lower())


def _rx(*words):
    return re.compile(r"\b(" + "|".join(words) + r")", re.IGNORECASE)


# Raízes sem acento (o texto é normalizado com unidecode)
ENTREGA = _rx("cheg", "entreg", "prazo", "atras", "demor", "correio", "transportad", "rastre", "envi", "receb")
ENTREGA_NEG = _rx("atras", "demor", "nao cheg", "ainda nao receb", "nao receb", "fora do prazo", "nao foi entregue", "extravi")
ENTREGA_POS = _rx("antes do prazo", "chegou rapido", "entrega rapida", "chegou antes", "super rapid", "muito rapid", "no prazo", "dentro do prazo")
AVARIA = _rx("quebrad", "amassad", "danificad", "avariad", "rasgad", "veio errad", "produto errado", "faltando", "falt", "incomplet", "violad", "riscad")
ANUNCIO = _rx("diferente d", "descricao", "anunci", "foto", "nao corresponde", "nao e o mesmo", "propaganda")
ATENDIMENTO = _rx("atendimento", "vendedor", "loja", "contato", "sac", "suporte", "reembols", "estorn", "devolu", "troca", "resposta", "respond")
NEGATIVO = _rx("pessim", "horrivel", "ruim", "nao recomendo", "decepcion", "absurd", "lixo", "nunca mais", "nao gostei", "insatisfeit", "problema")
POSITIVO = _rx("otim", "excelente", "perfeit", "adorei", "amei", "recomendo", "bom", "boa", "lindo", "satisfeit", "maravilh", "gostei", "parabens")
PRODUTO = _rx("produto", "qualidade", "material", "tamanho", "funciona", "bom", "otim", "excelente", "lindo", "perfeit", "ruim", "pessim")


def label_entrega(text, atrasado=None):
    t = _norm(text)
    if ENTREGA_NEG.search(t):
        return "negativo"
    if ENTREGA_POS.search(t):
        return "positivo"
    if atrasado and ENTREGA.search(t):
        return "negativo"
    return None


def label_recebimento(text, atrasado=None):
    return "negativo" if AVARIA.search(_norm(text)) else None


def label_anuncio(text, atrasado=None):
    return "negativo" if ANUNCIO.search(_norm(text)) else None


def label_atendimento(text, atrasado=None):
    t = _norm(text)
    if not ATENDIMENTO.search(t):
        return None
    if NEGATIVO.search(t):
        return "negativo"
    if POSITIVO.search(t):
        return "positivo"
    return None


def label_produto(text, atrasado=None):
    t = _norm(text)
    if not PRODUTO.search(t):
        return None
    if NEGATIVO.search(t):
        return "negativo"
    if POSITIVO.search(t):
        return "positivo"
    return None


LABELERS = {
    "produto": label_produto,
    "entrega": label_entrega,
    "recebimento": label_recebimento,
    "anuncio": label_anuncio,
    "atendimento": label_atendimento,
}


def heuristic_labels(text, atrasado=None, default="ausente"):
    """Rótulo completo por regras; aspectos sem opinião ficam como `default`."""
    return {a: (LABELERS[a](text, atrasado) or default) for a in ASPECTS}
