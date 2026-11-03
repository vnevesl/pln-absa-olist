"""Rotulagem por LLM com saída JSON, exemplos fixos e cache retomável.

O LLM é tratado como uma função `generate(prompts: list[str]) -> list[str]`,
então o mesmo código funciona com a API da Maritaca (Sabiá) ou com um modelo
local quantizado no Colab.

Todas as respostas brutas ficam em cache (JSONL), com chave que inclui o
modelo e a versão do prompt: se a execução cair, nada é refeito, e trocar de
modelo nunca reaproveita respostas de outro.
"""
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .config import ASPECT_GUIDE, ASPECTS, LLM_CACHE, POLARITIES, SEED

PROMPT_VERSION = "v2"  # mude ao alterar o texto do prompt: invalida o cache automaticamente

SYSTEM = (
    "Você é um anotador de avaliações de compras online em português. "
    "Para cada aspecto, diga se o cliente fala dele e qual é a opinião do cliente sobre ele."
)


def build_prompt(text: str, examples: list[dict] | None = None) -> str:
    aspectos = "\n".join(f'- "{a}": {d}' for a, d in ASPECT_GUIDE.items())
    valores = ", ".join(f'"{p}"' for p in POLARITIES)
    partes = [
        f"Aspectos:\n{aspectos}\n",
        f"Para cada aspecto, responda um destes valores: {valores}.\n"
        '- "ausente": o aspecto não é mencionado, nem de forma implícita '
        '("chegou antes do previsto" fala de entrega).\n'
        '- "positivo" ou "negativo": a opinião do cliente sobre aquele aspecto.\n'
        '- "neutro": o aspecto é mencionado sem opinião ("recebi hoje", "quando chega?").\n'
        'Avalie cada aspecto separadamente, mesmo que o comentário tenha um "mas". '
        'Se o texto fala da compra mas não de produto, entrega, recebimento ou anúncio, use "outros".\n',
        "Responda apenas com um JSON com exatamente estas chaves: " + ", ".join(ASPECTS) + ".\n",
    ]
    for ex in examples or []:
        rotulo = json.dumps({a: ex[a] for a in ASPECTS}, ensure_ascii=False)
        partes.append(f"Comentário: {ex['texto']}\nJSON: {rotulo}\n")
    partes.append(f"Comentário: {text}\nJSON:")
    return "\n".join(partes)


_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_response(raw: str) -> dict | None:
    """Extrai o primeiro JSON válido e normaliza; None se a resposta for inválida."""
    match = _JSON_RE.search(raw or "")
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for a in ASPECTS:
        v = str(obj.get(a, "ausente")).strip().lower()
        v = {"positiva": "positivo", "negativa": "negativo", "neutra": "neutro", "nenhum": "ausente"}.get(v, v)
        if v not in POLARITIES:
            return None
        out[a] = v
    return out


def select_fixed_examples(pool: pd.DataFrame, k=10, seed=SEED, text_col="texto_modelo") -> list[dict]:
    """Exemplos fixos que cobrem cada aspecto com opinião ao menos uma vez (e incluem um caso neutro)."""
    pool = pool.sample(frac=1, random_state=seed)
    escolhidos = []
    for a in ASPECTS:
        for pol in ["negativo", "positivo"]:
            cand = pool[(pool[f"asp_{a}"] == pol) & ~pool.index.isin([e.name for e in escolhidos])]
            if len(cand) and len(escolhidos) < k:
                escolhidos.append(cand.iloc[0])
    neutros = pool[(pool.filter(like="asp_") == "neutro").any(axis=1) & ~pool.index.isin([e.name for e in escolhidos])]
    if len(neutros):
        escolhidos[-1] = neutros.iloc[0]
    return [{"texto": r[text_col], **{a: r[f"asp_{a}"] for a in ASPECTS}} for r in escolhidos]


class JsonlCache:
    def __init__(self, name: str, folder: Path = LLM_CACHE):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{name}.jsonl"
        self.data = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["raw"]

    def get(self, key):
        return self.data.get(key)

    def put(self, key, raw, **meta):
        self.data[key] = raw
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "raw": raw, **meta}, ensure_ascii=False) + "\n")


def cache_key(prompt: str, model_id: str) -> str:
    return hashlib.sha1(f"{model_id}|{PROMPT_VERSION}|{prompt}".encode()).hexdigest()


def label_texts(texts, generate, model_id: str, examples=None, cache_name="llm", batch_size=16, progress=True):
    """Rotula uma lista de textos.

    Devolve (rótulos, respostas_brutas). Rótulo None = resposta inválida: quem chama
    deve DESCARTAR a linha (nunca tratá-la como "ausente").
    """
    cache = JsonlCache(cache_name)
    prompts = [build_prompt(t, examples) for t in texts]
    pendentes = [p for p in dict.fromkeys(prompts) if cache.get(cache_key(p, model_id)) is None]
    pendentes.sort(key=len)  # lotes com textos de tamanho parecido: menos padding, geração mais rápida
    t0 = time.time()
    for i in range(0, len(pendentes), batch_size):
        lote = pendentes[i:i + batch_size]
        for prompt, raw in zip(lote, generate(lote), strict=True):
            cache.put(cache_key(prompt, model_id), raw, model=model_id, prompt_version=PROMPT_VERSION)
        if progress:
            feitos = min(i + batch_size, len(pendentes))
            resta = (time.time() - t0) / feitos * (len(pendentes) - feitos)
            print(f"\r{feitos}/{len(pendentes)} rotulados · faltam ~{resta / 60:.0f} min", end="", flush=True)
    if progress and pendentes:
        print()
    raws = [cache.get(cache_key(p, model_id)) for p in prompts]
    return [parse_response(r) for r in raws], raws


# ---------------------------------------------------------------- backends

def openai_compatible_generator(model="sabiazinho-4", base_url="https://chat.maritaca.ai/api",
                                api_key_env="MARITACA_API_KEY", max_tokens=80, workers=4, retries=3):
    """API compatível com OpenAI (padrão: Maritaca). A chave vem só de variável de ambiente.

    Faz até `retries` tentativas por texto quando a chamada falha ou o JSON vem inválido.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)

    def one(prompt):
        raw = None
        for tentativa in range(retries):
            try:
                r = client.chat.completions.create(
                    model=model, temperature=0, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                )
                raw = r.choices[0].message.content
                if parse_response(raw) is not None:
                    return raw
            except Exception as erro:  # noqa: BLE001 — limite de taxa, rede: espera e tenta de novo
                raw = f"ERRO: {type(erro).__name__}"
                time.sleep(2 ** tentativa * 3)
        return raw

    def generate(prompts):
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, prompts))

    return generate


def hf_generator(model_id="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True, max_new_tokens=60):
    """LLM aberto via transformers, quantizado em 4 bits (cabe na T4 gratuita do Colab)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4") if load_in_4bit else None
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"quantization_config": quant, "device_map": "auto"}
    try:  # transformers >= 4.56 usa `dtype`; versões antigas, `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, **kwargs)
    model.eval()

    def generate(prompts):
        chats = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = tok(chats, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate
