"""Rotulagem fraca por LLM com saída JSON, few-shot e cache.

O LLM é tratado como uma função `generate(prompts: list[str]) -> list[str]`,
então o mesmo código funciona com um modelo local quantizado (Colab T4) ou
com uma API compatível com OpenAI (ex.: Maritaca/Sabiá).

Todas as respostas brutas ficam em cache (JSONL) — se a sessão do Colab cair,
nada é refeito.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from .config import ASPECT_DESCRIPTIONS, ASPECTS, LLM_CACHE, POLARITIES

SYSTEM = (
    "Você é um anotador de avaliações de compras online em português. "
    "Para cada aspecto, diga se o cliente fala dele e com qual polaridade."
)


def build_prompt(text: str, examples: list[dict] | None = None) -> str:
    aspectos = "\n".join(f'- "{a}": {d}' for a, d in ASPECT_DESCRIPTIONS.items())
    valores = ", ".join(f'"{p}"' for p in POLARITIES)
    partes = [
        f"Aspectos:\n{aspectos}\n",
        f"Para cada aspecto, responda um destes valores: {valores}. "
        'Use "ausente" quando o aspecto não for mencionado, nem de forma implícita '
        '(ex.: "chegou antes do previsto" fala de entrega). Use "neutro" quando o '
        "aspecto é mencionado sem opinião. Avalie cada aspecto separadamente, "
        'mesmo que o comentário tenha um "mas".\n',
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

    def put(self, key, raw):
        self.data[key] = raw
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")


def _key(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def label_texts(texts, generate, examples=None, cache_name="llm", batch_size=8, progress=True):
    """Rotula uma lista de textos. Devolve (rótulos, respostas_brutas); rótulo None = resposta inválida."""
    cache = JsonlCache(cache_name)
    prompts = [build_prompt(t, examples) for t in texts]
    pendentes = [p for p in dict.fromkeys(prompts) if cache.get(_key(p)) is None]
    for i in range(0, len(pendentes), batch_size):
        lote = pendentes[i:i + batch_size]
        for prompt, raw in zip(lote, generate(lote)):
            cache.put(_key(prompt), raw)
        if progress:
            print(f"\r{min(i + batch_size, len(pendentes))}/{len(pendentes)} rotulados", end="")
    raws = [cache.get(_key(p)) for p in prompts]
    return [parse_response(r) for r in raws], raws


# ---------------------------------------------------------------- backends

def hf_generator(model_id="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True, max_new_tokens=80):
    """LLM local via transformers (4 bits cabe na T4 do Colab)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4") if load_in_4bit else None
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant,
                                                 device_map="auto", torch_dtype=torch.float16)

    def generate(prompts):
        chats = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = tok(chats, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def openai_compatible_generator(model="sabiazinho-3", base_url="https://chat.maritaca.ai/api",
                                api_key_env="MARITACA_API_KEY", max_tokens=80):
    """API compatível com OpenAI. A chave vem de variável de ambiente, nunca do código."""
    from openai import OpenAI

    client = OpenAI(api_key=os.environ[api_key_env], base_url=base_url)

    def generate(prompts):
        outs = []
        for p in prompts:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=max_tokens,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
            )
            outs.append(r.choices[0].message.content)
        return outs

    return generate
