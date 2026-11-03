# Detalhamento do que roda no Colab — pln-absa-olist

Repositório: https://github.com/vnevesl/pln-absa-olist
Objetivo deste documento: permitir que outra pessoa (ou o assistente dela) revise a metodologia e o código e aponte erros ou escolhas discutíveis.

---

## 1. Tarefa

ABSA por categoria de aspecto (ACSA) em avaliações de e-commerce em português.

- **5 aspectos:** produto, entrega, recebimento, anuncio, outros
- **4 classes por aspecto:** ausente, positivo, negativo, neutro
- Cada comentário vira um vetor de 5 rótulos (um por aspecto).

## 2. Dados

### RePro (Hugging Face `lucasnil/repro`, B2W, PROPOR 2024) — treino e teste
- Rotulado por humanos: tópicos mencionados + polaridade.
- **Atenção:** no RePro a polaridade é por avaliação, não por tópico. Então:
  - comentário com 1 polaridade → essa polaridade vai para todos os tópicos marcados;
  - comentário com polaridades mistas → **descartado** (não dá para saber qual polaridade é de qual tópico).
- Splits oficiais de treino e teste. A validação foi dividida em:
  - `demo` (200): só serve de fonte para os exemplos do prompt do LLM;
  - `val_dev` (599): escolha de época do BERTimbau e medida de concordância LLM × humano.
  - `train_test_split(val, train_size=200, stratify=nota, random_state=42)`
- Texto de entrada: `"título. corpo"`, quando o título tem letras e não é prefixo do corpo.

### Olist (Kaggle) — validação externa (H3)
- 99.224 avaliações, das quais 40.641 têm comentário. Inclui datas de entrega real e estimada.
- `atrasado = (data entregue − data estimada) > 0`.
- Pedidos divididos por hash do `order_id` em `dev` / `relato`. Os números do artigo usam só `relato` (n = 19.239, sendo 1.876 atrasados).
- **Não tem rótulo de aspecto:** o modelo é treinado só no RePro e aplicado à Olist.

## 3. Hipóteses

- **H1** — BERTimbau supera as linhas de base clássicas no teste humano do RePro.
- **H2** — BERTimbau treinado com rótulos de um LLM (mesmos textos) fica próximo do treinado com rótulos humanos.
- **H3** — a probabilidade prevista de "entrega negativa" acompanha o atraso real na Olist, mesmo controlando pela nota (estrelas).

## 4. Métricas e estatística

- **Principal:** F1 micro de pares (aspecto, polaridade), ignorando "ausente".
- **Secundárias:**
  - F1 macro, com as classes fixadas pelo gabarito, sem "ausente";
  - F1 de detecção (aspecto mencionado ou não).
- IC 95% por bootstrap.
- Comparações por bootstrap pareado bilateral (10.000 reamostragens), com correção de Holm. McNemar como secundário.
- H3:
  - AUC com IC por bootstrap;
  - regressão logística com validação cruzada 5×, comparando nota contra nota + P(entrega neg);
  - AUC dentro de cada nota;
  - placebo: P(produto neg).

---

## 5. Notebook 1 — `notebooks/experimentos_colab.ipynb` (H1 e H3) — JÁ RODOU

1. Clona o repositório, instala o pacote e monta o Drive.
2. Carrega o RePro e cria `demo` e `val_dev`.
3. Treina o BERTimbau Base (`neuralmind/bert-base-portuguese-cased`):
   - **arquitetura:** encoder + 1 cabeça softmax de 4 classes por aspecto (5 cabeças);
   - **hiperparâmetros:** lr 3e-5, batch 32, 4 épocas, max_len 128, fp16 na T4;
   - **escolha da época:** a de maior F1 de pares em `val_dev`;
   - **condições:**
     - **H:** rótulos humanos, sementes 13, 42 e 87;
     - **V (controle):** treina só com `val_dev` (599 textos) e escolhe a época no `demo`.
4. Prediz o teste e salva CSVs no Drive. Pula o que já existe, então é retomável.
5. Aplica o modelo H (semente 42) à Olist `relato` e salva as probabilidades por aspecto e polaridade.
6. Zipa e baixa os resultados.

Linhas de base, rodadas localmente:
- regras (regex);
- TF-IDF (1–2-gramas) + Regressão Logística;
- TF-IDF (1–2-gramas) + Random Forest.

Em ambos os modelos TF-IDF, há um classificador de 4 classes por aspecto.

### Resultados obtidos (teste humano do RePro)

| Modelo | F1 pares | F1 macro |
|---|---|---|
| Regras | 0,654 | — |
| TF-IDF + Random Forest | 0,815 | — |
| TF-IDF + Reg. Logística | 0,867 (IC 0,856–0,878) | 0,764 |
| BERTimbau H (3 sementes) | **0,918 ± 0,002** | 0,848 |
| BERTimbau V (controle, só 599 textos) | 0,779 | 0,429 |

- BERTimbau H − TF-IDF: Δ = +0,048 (IC +0,038 a +0,059), p < 0,001 após Holm.
- **H3** (Olist relato):
  - AUC da nota sozinha = 0,821;
  - AUC de P(entrega neg) do BERTimbau = 0,830; do TF-IDF = 0,843;
  - placebo P(produto neg) = 0,461;
  - regressão com nota = 0,817, com nota + BERTimbau = 0,859 e com nota + TF-IDF = 0,864.

---

## 6. Notebook 2 — `notebooks/h2_qwen_colab.ipynb` (H2) — RODANDO AGORA (é o que demora)

Restrição do projeto: **custo zero**. Por isso não usamos API paga, e sim um LLM aberto no Colab gratuito (GPU T4, 15 GB).

### Passo a passo
1. Clona, instala `.[llm]` (transformers, bitsandbytes, accelerate) e monta o Drive. O cache do LLM fica no Drive.
2. Carrega o RePro. Escolhe **6 exemplos fixos** do `demo` para o prompt (few-shot), cobrindo cada aspecto com opinião e incluindo 1 caso neutro.
3. Carrega o **Qwen2.5-7B-Instruct quantizado em 4 bits** (NF4, bitsandbytes). O download baixa cerca de 15 GB de pesos fp16, que são quantizados ao carregar.
4. **Rotula `val_dev` (599)** e mede a concordância LLM × humano (F1 de pares com IC).
5. **Rotula o treino (5.589)**. ⬅️ **ESTA É A CÉLULA QUE DEMORA (~2,5 h)**
6. Treina o BERTimbau com os rótulos do LLM (condição **L**, sementes 13, 42 e 87), com os mesmos hiperparâmetros de H, e avalia no **mesmo teste humano**.
7. Zipa e baixa os resultados.

### Como a rotulagem funciona

- **Prompt:**
  - sistema: *"Você é um anotador de avaliações de compras online…"*;
  - corpo: guia de aspectos + definições das 4 classes + 6 exemplos `Comentário: … JSON: {...}` + o comentário alvo;
  - o modelo responde só com um JSON de 5 chaves.
- **Decodificação:** gulosa (`do_sample=False`), `max_new_tokens=60`, lotes de 16, prompts ordenados por tamanho para reduzir o padding.
- **Resposta inválida:** se o JSON não for válido, a linha é descartada do treino e conta como erro na avaliação de concordância.
- **Cache retomável:** cada resposta é gravada num JSONL no Drive, com chave `sha1(modelo|versão_do_prompt|prompt)`. Se o Colab cair, ao rodar de novo só faz o que falta.

### Código da célula lenta (células 4 e 5 do notebook)

```python
def rotular(df, nome, lote=16):
    arq = SAIDA / f"repro_{nome}_qwen7b.csv"
    if arq.exists():
        return pd.read_csv(arq)
    rot, _ = label_texts(df.texto_modelo.tolist(), generate, MODELO, exemplos,
                         cache_name=f"qwen7b_{nome}", batch_size=lote)
    out = df[["review_id"]].copy()
    for a in ASPECTS:
        out[f"asp_{a}"] = [r[a] if r else None for r in rot]
    out["valido"] = [r is not None for r in rot]
    out.to_csv(arq, index=False)
    return out

rot_val = rotular(val_dev, "valdev")      # 599 textos  -> concordância com humano
rot_treino = rotular(treino, "treino")    # 5.589 textos -> ~2,5 h na T4  <-- GARGALO
```

### Funções chamadas (`src/absa/llm_labeler.py`)

```python
def label_texts(texts, generate, model_id, examples=None, cache_name="llm", batch_size=16):
    cache = JsonlCache(cache_name)                       # JSONL no Drive
    prompts = [build_prompt(t, examples) for t in texts]
    pendentes = [p for p in dict.fromkeys(prompts) if cache.get(cache_key(p, model_id)) is None]
    pendentes.sort(key=len)                              # menos padding
    for i in range(0, len(pendentes), batch_size):
        lote = pendentes[i:i + batch_size]
        for prompt, raw in zip(lote, generate(lote)):
            cache.put(cache_key(prompt, model_id), raw)  # salva a cada lote
    raws = [cache.get(cache_key(p, model_id)) for p in prompts]
    return [parse_response(r) for r in raws], raws

def hf_generator(model_id="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=True, max_new_tokens=60):
    tok = AutoTokenizer.from_pretrained(model_id); tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant,
                                                 device_map="auto", dtype=torch.float16).eval()
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
```

### Por que demora

- São 5.589 textos com um prompt longo (guia + 6 exemplos, centenas de tokens), num modelo de 7B em 4 bits numa T4.
- O bitsandbytes em 4 bits é lento na geração. Mediu-se ~1,7 s por texto, ou seja, ~2,5 h.
- O Colab gratuito desconecta após algumas horas ou por inatividade. Por isso existe o cache no Drive: fecha, reabre, clica em "Executar tudo" e ele continua.

### Saídas esperadas

- Concordância LLM × humano no `val_dev` e no treino.
- F1 do BERTimbau L no teste humano, comparado com H (bootstrap pareado + Holm).
- **Recuperação do gap** = (L − V) / (H − V): quanto do ganho de ter 5,6 mil rótulos humanos o LLM recupera de graça.

---

## 7. Pontos que gostaríamos que fossem revisados

1. **Descartar comentários com polaridade mista** do RePro: isso deixa o teste mais fácil do que o mundo real? A limitação está declarada no artigo.
2. **Escolha da época de L em `val_dev` humano:** usa 599 rótulos humanos para selecionar o modelo treinado com rótulos do LLM. É justo, já que H usa o mesmo, mas o L não é 100% livre de humano.
3. **Resposta inválida do LLM** é descartada do treino e conta como erro na concordância. É o tratamento certo?
4. **H3:** o TF-IDF teve AUC um pouco maior que o BERTimbau, e dentro das notas 4 e 5 o TF-IDF também vai melhor. Nossa leitura é que a H3 é sustentada pelos dois modelos, não que o BERTimbau seja melhor nela.
5. **Velocidade:** há forma gratuita mais rápida na T4? Opções: vLLM com AWQ, menos exemplos no prompt, `max_new_tokens` menor ou um Qwen2.5-3B. Trocar agora invalidaria o cache já feito.
6. Qualquer vazamento entre treino e teste, ou entre RePro e Olist, que não tenhamos visto.
