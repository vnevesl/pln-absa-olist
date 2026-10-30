# LLM como anotador e o experimento humano × LLM: como deixar sólido

*Pesquisa de melhorias nº 03 · 30/10/2026 · escopo: `src/absa/llm_labeler.py`, `src/absa/config.py`, `notebooks/04_rotulagem_llm.ipynb`, `notebooks/05_encoders.ipynb`*

Convenções: **[não verificado]** marca algo que não confirmei numa fonte primária. **[estimativa]** marca números calculados por mim, que precisam ser medidos num piloto. Todo o resto tem link.

---

## 1. Resumo executivo

1. **O maior risco do experimento não é o modelo, é o desalinhamento entre o prompt e o gabarito.** No RePro, a polaridade é da avaliação inteira e foi propagada para os tópicos (`repro.py`). Os anotadores viram título + corpo e trabalharam em lotes separados por nota ([Silva et al., 2024](https://aclanthology.org/2024.propor-1.44/)). Já o prompt atual vê só o corpo e usa definições de aspecto que divergem do guia do RePro (ex.: "custo-benefício" é PRODUTO no RePro, mas no prompt "preço" está em "outros"). Corrigir isso deve render mais que trocar de LLM.
2. **Modelo:** dentro da T4, o melhor custo-benefício verificado é **Qwen2.5-14B-Instruct-AWQ** (PoETa v2 = 71,0, contra 72,2 do Sabiá-3 e 63,7 do Qwen2.5-7B; [Almeida et al., 2025](https://arxiv.org/html/2511.17808v2)). Ele roda em **vLLM** com **decodificação restrita por JSON Schema**. O Qwen2.5-7B fica como plano B. O **Sabiazinho-4 / Sabiá-4** entra como segundo anotador barato via API (cerca de R$ 15 a R$ 75 para todos os textos [estimativa]).
3. **Formato:** para classificação, decodificação restrita (enum por aspecto) + resposta direta, sem CoT. Isso elimina respostas inválidas e não prejudica tarefas que não exigem raciocínio ([Tam et al., 2024](https://arxiv.org/abs/2408.02442); [Sprague et al., 2025](https://arxiv.org/abs/2409.12183)). Também permite extrair **probabilidades por aspecto**, que servem de confiança e de rótulo suave.
4. **Evidência direta no próprio RePro:** com BERTimbau, rótulos do Sabiá-3 deram F1 macro de 0,62, contra 0,68 com rótulos humanos (tópicos, 1.400 textos). A perda se concentra em OUTROS (F1 0,35 × 0,73) e CONDIÇÕES DE RECEBIMENTO ([Silva; Real, 2024](https://aclanthology.org/2024.stil-1.61)). Esse é o número de referência que o projeto deve superar ou explicar.
5. **Desenho:** curva de aprendizado humana (10/25/50/100%), condições LLM (bruto, filtrado por confiança, suave, robusto a ruído), mistura humano+LLM e o controle "treinar só na validação humana" ([Zhu et al., 2023](https://aclanthology.org/2023.acl-long.796/)), com 3 sementes. A métrica de destaque é o **"equivalente-humano"**: quantos rótulos humanos o conjunto de 5.589 rótulos do LLM vale.

---

## 2. Escolha do modelo

### 2.1 O que se sabe sobre desempenho em português

- **PoETa v2** (44 tarefas, 12 nativas em PT; [Almeida et al., 2025](https://arxiv.org/html/2511.17808v2)), Tabela 2: Qwen2.5-14B **71,0**; Qwen3-14B 70,5; Qwen3-8B 64,7; Qwen2.5-7B 63,7; Falcon3-10B 63,5; Llama-3.1-8B 53,5; Qwen2.5-3B 53,1; Sabiá-7B 39,0. Comerciais: GPT-4.1 76,2; GPT-4o 75,2; Sabiá-3 72,2. Gemma 3 não aparece na tabela. Um artigo do STIL 2026 alerta que médias agregadas escondem diferenças grandes por tarefa ([STIL 2026](https://aclanthology.org/2026.stil-1.18.pdf)).
- **Sentimento em PT-BR** ([Schuck et al., 2025, JBCS](https://journals-sol.sbc.org.br/index.php/jbcs/article/view/5793)): 23 LLMs, 12 datasets, ICL com 6 exemplos (3 por classe) e saída em JSON. Entre modelos < 13B, a ordem por acurácia média foi Gemma-2-9B 0,934 (F1 0,785), Qwen2-7B 0,923 (F1 0,700) e Llama-3-8B 0,919 (F1 0,647). O Llama-3.1-8B ficou em 0,759 (F1 0,490), com 8,3% de respostas fora do formato. No topo, Claude-3.5-Sonnet 0,948 e Sabiá-3 0,946. Os pequenos foram estatisticamente equivalentes a GPT-4o e DeepSeek-V3 em acurácia, mas tendem a **"sobregerar"** (texto além do rótulo). O ajuste fino em PT reduziu alucinação, mas não melhorou o desempenho de forma consistente.
- **PROPOR 2026** ([Lima; Pinheiro; Caminha, 2026](https://preview.aclanthology.org/ingest-dnd/2026.propor-1.21.pdf)): 29 LLMs abertos. CoT + few-shot ajuda mais os modelos grandes; vários modelos compactos chegam a F1 competitivo (resumo; números por modelo não verificados por mim).
- **Open PT LLM Leaderboard** ([Garcia, HF Space](https://huggingface.co/spaces/eduagarcia/open_pt_llm_leaderboard)): existe e inclui ENEM, BLUEX, ASSIN2, hate speech etc. Não consegui ler o ranking atual (Space em Gradio), por isso não citei posições **[não verificado]**. Para a escolha, usei o PoETa v2, que é publicado e versionado.

### 2.2 Restrições da T4 que mudam a escolha

- Sem bf16. O vLLM exige capacidade de computação ≥ 7,5 e lista a T4 explicitamente ([docs vLLM, GPU](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)). AWQ, GPTQ e bitsandbytes têm suporte em Turing (SM 7.5) ([docs vLLM, quantização](https://docs.vllm.ai/en/stable/features/quantization/)). É preciso forçar `dtype="float16"`.
- **Gemma 3 em fp16 estoura** (ativações > 65.504 viram infinito em T4/V100). A Unsloth documenta o problema e uma correção própria ([Unsloth, Gemma 3](https://docs.unsloth.ai/basics/gemma-3-how-to-run-and-fine-tune)). Não sei se o Gemma 4 tem o mesmo problema **[não verificado]**, mas é um risco que não vale correr num prazo de 2 meses.
- **Qwen3.5-9B** (fev/2026): multimodal, arquitetura híbrida Gated DeltaNet + atenção, *thinking* ligado por padrão, e o card pede vLLM/Transformers **do branch main/nightly** ([card HF](https://huggingface.co/Qwen/Qwen3.5-9B)). O MMMLU é forte (81,2), mas a pilha é imatura para T4 + quantização **[compatibilidade não verificada]**.

### 2.3 Tabela de decisão

| Modelo | Tam. (4 bits) | Evidência PT | Cabe na T4? | Prós | Contras | Veredito |
|---|---|---|---|---|---|---|
| **Qwen2.5-14B-Instruct-AWQ** | ~10,0 GB de pesos ([HF](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ/tree/main)) | PoETa v2 71,0 | Sim, com folga curta (~4 GB p/ KV) [testar] | Melhor aberto verificado que cabe; checkpoint AWQ oficial; PT na lista de idiomas | ~2× mais lento que o 7B | **Principal** |
| Qwen2.5-7B-Instruct (AWQ ou nf4) | ~5,5 GB | PoETa v2 63,7 | Folgado | É o plano atual; rápido | 7 pts abaixo do 14B no PoETa | **Plano B** / ablação de escala |
| Qwen3-8B-AWQ (modo sem *thinking*) | ~6 GB | PoETa v2 64,7 | Sim | ~igual ao 7B; `enable_thinking=False` ([card](https://huggingface.co/Qwen/Qwen3-8B-AWQ)) | O card mostra perda do AWQ no modo sem *thinking* (LiveBench 48,9 × 53,5) | Opcional |
| Qwen3-14B-AWQ | ~10 GB | PoETa v2 70,5 | Sim [testar] | Equivalente ao 2.5-14B | Não supera o 2.5-14B no PoETa | Alternativa |
| Gemma 3 12B / Gemma 4 E4B | 7–8 GB | Sem dado no PoETa v2; Gemma-2-9B foi o melhor pequeno em Schuck | Sim | Gemma 4 é Apache 2.0, 140+ idiomas ([card](https://huggingface.co/google/gemma-4-E4B)) | Estouro fp16 na T4 (Gemma 3) | Evitar na T4 |
| Llama 3.1 8B | ~5,5 GB | PoETa 53,5; Schuck F1 0,49 | Sim | — | Fraco em PT, sobregera | Descartar |
| Qwen3.5-9B | ~6 GB | MMMLU 81,2 (card) | Talvez | Mais novo | Exige *nightly*; arquitetura híbrida; multimodal | Só se sobrar tempo |
| Tucano (≤ 2,4B) ([Corrêa et al., 2025](https://arxiv.org/abs/2411.07854)) / Gervásio 7B ([Santos et al., 2024](https://arxiv.org/abs/2402.18766)) | — | Gervásio é Llama-2; Sabiá-7B (Llama-1) tem PoETa 39,0 | Sim | Nativos PT | Base antiga, instruções fracas | Descartar como anotador |
| **Sabiazinho-4 / Sabiá-4 (API)** | — | Sabiá-3: PoETa 72,2, acc 0,946 em Schuck | API | Nativo PT-BR; barato; batch com 50% de desconto ([preços](https://docs.maritaca.ai/pt/precos)) | Fechado; possível contaminação; não vi doc de JSON Schema **[não verificado]** | **Segundo anotador** e "professor forte" |

Preços da Maritaca (R$/1M tokens, entrada/saída): Sabiazinho-4 1,00/4,00; Sabiá-4 5,00/20,00; Batch API pela metade ([docs](https://docs.maritaca.ai/pt/precos)). Modelos atuais: `sabia-4` e `sabiazinho-4` ([docs](https://docs.maritaca.ai/pt/modelos)). **O padrão `sabiazinho-3` em `openai_compatible_generator` está descontinuado** e precisa mudar.

### 2.4 vLLM × transformers no Colab

| | transformers + bitsandbytes nf4 (atual) | vLLM + AWQ |
|---|---|---|
| Batching | Estático, com *padding* à esquerda | Contínuo (*continuous batching*) |
| Prefixo comum (instruções + 6 exemplos ≈ 80% do prompt) | Recalculado a cada texto | `enable_prefix_caching=True` reaproveita o KV do prefixo |
| Saída estruturada | Regex + parse depois | `StructuredOutputsParams(json=schema)` com xgrammar/guidance ([docs](https://docs.vllm.ai/en/latest/features/structured_outputs.html)) |
| Logprobs por token | Possível, mas manual | `SamplingParams(logprobs=k)` |
| Risco no Colab | Baixo | Instalação pesada; versão muda rápido. Os campos `guided_*` antigos foram removidos na v0.12 (mesma doc) |

Recomendação: **vLLM como backend principal**, mantendo o `hf_generator` como fallback, já que a interface `generate(prompts)` do projeto permite trocar sem mexer no resto.

---

## 3. Decodificação e formato

1. **Saída restrita por JSON Schema**, com `enum` dos 4 valores em cada aspecto e as 5 chaves obrigatórias, na ordem fixa de `ASPECTS`. Isso zera as respostas inválidas. Hoje, no notebook 04, uma resposta inválida vira "ausente" em tudo, o que **injeta ruído sistemático no rótulo de treino** e penaliza o LLM no teste por uma falha de formato, e não de compreensão.
2. **O que Tam et al. (2024) dizem de fato:** formatos estritos prejudicam tarefas de **raciocínio**, mas em **classificação** formatos estruturados podem até melhorar a acurácia. Erros de parse não são a principal causa das quedas ([EMNLP Industry 2024](https://arxiv.org/abs/2408.02442)). Como ACSA aqui é classificação, a restrição é segura. Ainda assim, vale uma ablação barata na validação: **restrito × livre + parse**.
3. **Sem CoT.** A meta-análise de mais de 100 artigos mostra ganho de CoT concentrado em matemática e lógica simbólica ([Sprague et al., ICLR 2025](https://arxiv.org/abs/2409.12183)). Para o Qwen3, usar `enable_thinking=False`. Uma variante "justificativa curta antes do JSON" pode ser testada só na validação. Para modelos grandes há evidência de ganho de CoT + few-shot em sentimento PT ([Lima et al., 2026](https://preview.aclanthology.org/ingest-dnd/2026.propor-1.21.pdf)), mas ela multiplica o custo por cerca de 5.
4. **Temperatura 0** (guloso) para o rótulo principal. *Self-consistency* por amostragem ([Wang et al., 2023](https://arxiv.org/abs/2203.11171)) serve para raciocínio. Para classificação, a diversidade útil vem de **variar o prompt ou o modelo** (seção 4).
5. **Probabilidades por aspecto:** com `logprobs=20`, ler no token em que começa cada valor a distribuição sobre {ausente, positivo, negativo, neutro} e renormalizar sobre as 4 opções. Isso dá (a) confiança para filtrar, (b) rótulo suave para destilação e (c) curva de calibração. **Conferir no tokenizador** se os 4 rótulos começam por tokens distintos depois de `"` (no Qwen, provavelmente `aus`/`pos`/`neg`/`neut` **[não verificado]**). Se não começarem, usar códigos de uma letra só para o *scoring* ou pontuar cada sequência completa.
6. **Metadados que o anotador humano viu:** título, corpo e nota (os lotes do RePro eram por estrela; [Silva et al., 2024](https://aclanthology.org/2024.propor-1.44/)). Dar título e nota ao LLM é justo e aproxima as condições. Fazer ablação "com/sem nota" na validação, porque a nota também cria atalho de polaridade.

---

## 4. Boas práticas de anotação por LLM (2021–2026)

| Prática | Evidência | Aplicação aqui |
|---|---|---|
| **Validar contra humanos, tarefa a tarefa** | GPT-4 em 27 tarefas: desempenho "altamente contingente" à tarefa ([Pangakis et al., 2023](https://arxiv.org/abs/2306.00176)) | Concordância **por aspecto** na validação, κ de Cohen comparado ao IAA humano do RePro (κ = 0,68 tópico, 0,71 sentimento) |
| LLMs podem superar *crowd* em tarefas simples | ChatGPT zero-shot superou MTurk em 4 de 5 tarefas, com custo < US$ 0,003/rótulo, cerca de 20× mais barato ([Gilardi et al., 2023](https://arxiv.org/abs/2303.15056)) | Justifica a pergunta, mas o RePro tem anotadores especialistas + curadoria, um padrão mais alto que *crowd* |
| Rótulo LLM + humano > cada um sozinho | GPT-3 reduziu de 50% a 96% o custo para o mesmo desempenho; a mistura foi melhor que cada fonte ([Wang et al., 2021](https://aclanthology.org/2021.findings-emnlp.354/)) | Condição "mistura" (seção 6) |
| Padronizar o processo: escolha de modelo, prompt estruturado, **estabilidade do prompt**, validação, ética | [Törnberg, 2024](https://arxiv.org/abs/2402.05129); survey em [Tan et al., EMNLP 2024](https://arxiv.org/abs/2402.13446) | Versionar o prompt (hash), relatar variação entre ≥ 3 formulações |
| **Sensibilidade a formato** | Até 76 pontos de diferença entre formatos equivalentes (Llama-2-13B), em média ~10, e não some com mais exemplos nem com modelos maiores ([Sclar et al., ICLR 2024](https://arxiv.org/abs/2310.11324)) | Relatar faixa (mín–máx) de F1 entre 3 variantes de prompt, não só a melhor |
| Ordem dos exemplos importa | [Lu et al., ACL 2022](https://aclanthology.org/2022.acl-long.556/) | Fixar ordem; exemplo mais parecido por último no kNN |
| **Exemplos por similaridade (kNN)** > aleatórios | [Liu et al., DeeLIO 2022](https://aclanthology.org/2022.deelio-1.10) | Pool de demonstrações rotulado, recuperação por embedding |
| Rótulo de demonstração importa menos que formato e distribuição | [Min et al., EMNLP 2022](https://arxiv.org/abs/2202.12837) | Garantir **cobertura** de cada aspecto e polaridade nos exemplos |
| Confiança por concordância entre prompts | Concordância entre múltiplos prompts estima melhor o acerto que logprob de um prompt só ([Portillo Wightman et al., TrustNLP 2023](https://aclanthology.org/2023.trustnlp-1.28/)) | 3 variantes → voto majoritário; concordância vira confiança |
| Filtrar pode não ajudar o aluno | Filtro de autoverificação subiu a acurácia dos rótulos em ~5 pts, mas não melhorou o modelo treinado ([Gekhman et al., EMNLP 2023](https://arxiv.org/abs/2305.11171)) | Filtragem é **hipótese a testar**, não melhoria garantida |
| Verificar e reanotar só o que tem baixa confiança | Verificador de rótulos LLM supera logit-baseline; humanos reanotam os piores ([Wang et al., CHI 2024](https://doi.org/10.1145/3613904.3641960)) | Condição "LLM + humano nos k% menos confiantes" (simulada com o gabarito) |
| Rótulos candidatos/suaves para o aluno | CanDist (ACL 2025): destilar conjuntos de candidatos do LLM ([Xia et al., 2025](https://arxiv.org/abs/2506.03857)); LLKD (NAACL Findings 2025) seleciona amostras confiantes do professor ([2025.findings-naacl.142](https://preview.aclanthology.org/dashboard/2025.findings-naacl.142)) | Condição "rótulo suave" (KL com as probabilidades do LLM) |

**Riscos específicos:**

- **Contaminação:** o RePro é público desde 2024 ([HF](https://huggingface.co/datasets/lucasnil/repro)) e o LLM pode tê-lo visto. Para testar, use o método de completar trechos ([Golchin; Surdeanu, ICLR 2024](https://arxiv.org/abs/2308.08493)): dar a primeira metade de 100 textos do teste e medir a sobreposição da continuação com o original. A amostra própria da Olist (100 textos) funciona como teste não contaminado.
- **Viés:** o LLM pode tender a "ausente" em OUTROS. O Sabiá-3 teve recall 0,41 em OUTROS no RePro ([Silva; Real, 2024](https://aclanthology.org/2024.stil-1.61)). Comparar distribuições de rótulo (LLM × humano) por aspecto e por nota, e cruzar com o Fairlearn (gênero, UF) já previsto no notebook 08.

---

## 5. Evidências: ABSA com LLM, em PT e em geral

| Trabalho | Cenário | Números | Lição |
|---|---|---|---|
| **[Silva; Real, STIL 2024](https://aclanthology.org/2024.stil-1.61)** | **RePro**, tópicos (6, multirrótulo), Sabiá-3 com o guia do RePro no prompt | LLM × humano (amostra aleatória, 1.400): F1 macro 0,69. ENTREGA 0,96, PRODUTO 0,95, ANÚNCIO 0,71, RECEBIMENTO 0,65, OUTROS 0,56 (recall 0,41). BERTimbau treinado: **humano 0,68 × LLM 0,62**; OUTROS 0,73 × 0,35; RECEBIMENTO 0,79 × 0,66; ANÚNCIO 0,67 × 0,72. 69 alucinações em 3.854 | **Referência direta**: a perda está nos aspectos difusos. Não houve polaridade, few-shot nem filtragem, e é aí que o projeto pode contribuir |
| [Silva et al., PROPOR 2024 (RePro)](https://aclanthology.org/2024.propor-1.44/) | Gabarito | ≥ 2 anotadores + curador; κ tópico 0,68, sentimento 0,71 | Teto humano: um LLM com κ ≈ 0,7 já está "no nível de um anotador" |
| [Eiras; Brito, IEEE Access 2026](https://ipt.br/?p=66196) | ABSA não supervisionado, cervejas PT (1.712 rotulados), Sabiá-3 × GPT-4o mini, 0/1/few-shot | F1 extração 0,857; categoria 0,826; **sentimento 0,552**. Sentimento foi o mais sensível ao prompt | Polaridade por aspecto é o elo fraco; precisa de ablação de prompt |
| [Schuck et al., JBCS 2025](https://journals-sol.sbc.org.br/index.php/jbcs/article/view/5793) | Polaridade (nível documento), 12 datasets PT | Ver §2.1 | Modelos de 7–9B ≈ grandes em acurácia; sobregeração |
| [Zhang et al., Findings NAACL 2024](https://arxiv.org/abs/2305.15005) | 13 tarefas, 26 datasets | LLMs ficam atrás de modelos pequenos treinados em ABSA estruturado; vencem em poucos dados | Esperar que o LLM direto no teste perca para o BERTimbau humano, e que o ganho do LLM apareça quando há poucos rótulos |
| [Šmíd et al., WASSA 2024](https://aclanthology.org/2024.wassa-1.6/) | LLaMA/Orca-2 em 4 tarefas ABSA, 8 datasets EN | Ajuste fino superou o SOTA; zero/few-shot bem abaixo do ajustado | Mesma lição |
| [Simmering; Huoviala, 2023](https://arxiv.org/abs/2310.18025) | SemEval-14 (aspecto + polaridade) | GPT-3.5 ajustado F1 83,8 (SOTA na época); prompts detalhados ajudam em zero/few-shot | Investir nas **definições** dos aspectos no prompt |

---

## 6. Desenho do experimento humano × LLM

### 6.1 Princípios

- **Mesmos textos, mesma arquitetura, mesmo teste humano** (1.597). Só muda a fonte do rótulo.
- **Contabilidade de humanos:** todo rótulo humano usado pelo pipeline do LLM (exemplos few-shot, pool kNN, ajuste do prompt na validação, seleção de época) entra no "orçamento humano" da condição LLM. Hoje o notebook 05 seleciona a época na **validação humana (799)** para os dois casos. Isso é aceitável, mas precisa ser declarado. E, como mostram [Zhu et al. (ACL 2023)](https://aclanthology.org/2023.acl-long.796/), um conjunto limpo desse tamanho pode, sozinho, bater métodos de supervisão fraca. Por isso o controle "treinar só nos 799 da validação" é **obrigatório**.
- **Sementes:** ≥ 3 por condição (5 nas principais). Reportar média ± dp, e IC por bootstrap sobre o teste para a média de sementes ou para a semente mediana.
- **Nunca olhar o teste** para escolher prompt, modelo, limiar de confiança ou hiperparâmetro.

### 6.2 Divisão da validação (799)

- `val_demo` (≈ 200): pool de exemplos few-shot / kNN, estratificado para cobrir os 5 aspectos × 4 polaridades.
- `val_dev` (≈ 599): ajuste de prompt, limiar de confiança e seleção de época dos encoders.

### 6.3 Condições (BERTimbau em todas; Albertina só nas marcadas com ★)

| ID | Rótulos de treino (5.589 textos) | Humanos usados | Pergunta |
|---|---|---|---|
| H-10/25/50/100 ★(100) | Humanos, subamostras estratificadas por nota de 559 / 1.397 / 2.795 / 5.589 | n + 799 | Curva de aprendizado humana |
| V-only | Só `val_dev`, sem treino (época escolhida por validação cruzada interna) | 599 | Controle de [Zhu et al., 2023](https://aclanthology.org/2023.acl-long.796/) |
| L-7B | Qwen2.5-7B, hard | 799 | Ablação de escala |
| **L-14B** ★ | Qwen2.5-14B, hard | 799 | **Condição principal** |
| L-ens | Voto de 3 prompts × (14B + Sabiá-4) | 799 | O conjunto ajuda? |
| L-filt | L-14B só com confiança ≥ τ (τ escolhido em `val_dev`) | 799 | Filtragem ajuda o aluno? (cf. Gekhman 2023) |
| L-soft | Alvo = probabilidades do LLM (KL), sem *argmax* | 799 | Destilação suave |
| L-robust | L-14B + *label smoothing* 0,1 ([Lukasik et al., ICML 2020](https://arxiv.org/abs/2003.02819)) ou GCE q = 0,7 ([Zhang; Sabuncu, NeurIPS 2018](https://arxiv.org/abs/1805.07836)) | 799 | Perda robusta compensa o ruído? |
| M-10/25 | 10% / 25% humanos + LLM no restante (humano tem prioridade) | 559 / 1.397 + 799 | Mistura ([Wang et al., 2021](https://aclanthology.org/2021.findings-emnlp.354/)) |
| R-10 | LLM, com os 10% de **menor confiança** trocados pelo rótulo humano (reanotação simulada) | 559 + 799 | Verificação dirigida ([Wang et al., CHI 2024](https://doi.org/10.1145/3613904.3641960)) |

Co-teaching ([Han et al., NeurIPS 2018](https://arxiv.org/abs/1804.06872)) e *confident learning* ([Northcutt et al., JAIR 2021](https://arxiv.org/abs/1911.00068); o `cleanlab` já está nas dependências) ficam como extras: são mais caros de implementar e, segundo Zhu et al., tendem a perder para usar bem os dados limpos.

### 6.4 Métricas de comparação

- F1 macro por aspecto e F1 de tupla (as que já existem), com bootstrap pareado e McNemar contra H-100.
- **Recuperação da lacuna** = (F1_L − F1_piso) / (F1_H100 − F1_piso), com piso = V-only ou TF-IDF+LR (0,759).
- **Equivalente-humano** = n humano em que a curva H-n (interpolação log-linear) atinge o F1 de L-14B. Frase de resultado: "5.589 rótulos do LLM valem ≈ X rótulos humanos".
- **Qualidade do rótulo em si:** κ por aspecto LLM × humano no treino, comparado a κ = 0,68/0,71 do RePro. Matriz de confusão por aspecto. Curva de confiabilidade (calibração) das probabilidades.
- **Ruído × desempenho:** correlação entre o F1 do rótulo LLM por aspecto e a queda do aluno por aspecto.

### 6.5 Como reportar custo

Uma tabela por fonte de rótulo, **por mil rótulos**: tempo de parede, GPU-hora (tipo de GPU), tokens de entrada e saída, R$ (API), e humanos-hora. Para humanos, não invente salário: use uma fórmula explícita, custo = n × (anotadores por texto × s/texto + fração curada × s/curadoria) × R$/h, com os parâmetros declarados. O RePro usou ≥ 2 anotadores + curador ([Silva et al., 2024](https://aclanthology.org/2024.propor-1.44/)). Medir os s/texto com a anotação própria dos 100 comentários da Olist. Como referência externa, US$ 0,003 por rótulo de LLM contra MTurk em [Gilardi et al. (2023)](https://arxiv.org/abs/2303.15056).

---

## 7. Plano concreto (mudanças no código)

### 7.1 `src/absa/config.py`: alinhar definições ao guia do RePro

As descrições atuais divergem do guia publicado (descrições em [Silva; Real, 2024](https://aclanthology.org/2024.stil-1.61), §3.1):

| Aspecto | Atual | RePro |
|---|---|---|
| produto | qualidade, funcionamento, durabilidade, tamanho, aparência | + **originalidade, custo-benefício**, experiência de uso, **elogios gerais** |
| entrega | prazo, atraso, rapidez, rastreio, transportadora | + **não entrega, retirada na loja, entrega virtual (vale-presente, código), frete** |
| recebimento | avaria, embalagem, item faltando/errado | + **se o produto atendeu à expectativa** |
| anuncio | fidelidade à descrição, fotos, preço anunciado | descrição, imagens, **especificações técnicas** |
| outros | loja/vendedor, atendimento, troca, **preço**, site | dúvidas ao vendedor, atendimento, **estoque, experiência de compra, forma de pagamento, texto sem sentido** |

Sugestão: criar `ASPECT_GUIDE` (texto longo, para o prompt), mantendo `ASPECT_DESCRIPTIONS` (curto, para tabelas). Tirar "preço" de outros, ou dizer explicitamente "custo-benefício → produto".

### 7.2 `src/absa/llm_labeler.py`

1. **Versão do prompt e modelo na chave do cache.** Hoje `_key = sha1(prompt)`, então trocar o modelo com o mesmo `cache_name` reaproveita respostas do modelo antigo. Usar `sha1(model_id | PROMPT_VERSION | decoding | prompt)`. Gravar no JSONL também `model`, `prompt_version`, `probs` e `latency_s`.
2. **`build_prompt(texto, titulo=None, nota=None, examples=...)`:** incluir título e (opcional) nota. Regras de polaridade explícitas:
   - "polaridade é a opinião do cliente sobre aquele aspecto";
   - "neutro = mencionado sem opinião";
   - "na dúvida entre ausente e outros, prefira outros se o texto fala da compra mas não de produto/entrega/recebimento/anúncio" (ataca o recall baixo de OUTROS).
   - Manter "Responda só o JSON".
3. **`JSON_SCHEMA`:** `{"type":"object","properties":{a:{"enum":POLARITIES} for a in ASPECTS},"required":ASPECTS,"additionalProperties":false}`.
4. **Novo backend `vllm_generator`:**
   ```python
   def vllm_generator(model_id="Qwen/Qwen2.5-14B-Instruct-AWQ", max_model_len=2048,
                      gpu_mem=0.90, logprobs=20, enable_thinking=None):
       from vllm import LLM, SamplingParams
       from vllm.sampling_params import StructuredOutputsParams
       llm = LLM(model=model_id, dtype="float16", quantization="awq",
                 max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
                 enable_prefix_caching=True, seed=SEED)
       sp = SamplingParams(temperature=0, max_tokens=60, logprobs=logprobs,
                           structured_outputs=StructuredOutputsParams(json=JSON_SCHEMA))
       def generate(prompts):  # prompts = mensagens já formatadas no chat template
           outs = llm.chat([[{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": p}] for p in prompts], sp,
                           chat_template_kwargs=({"enable_thinking": enable_thinking}
                                                 if enable_thinking is not None else None))
           return [{"raw": o.outputs[0].text, "probs": _aspect_probs(o.outputs[0])} for o in outs]
       return generate
   ```
   A API exata de `llm.chat`/`chat_template_kwargs` varia entre versões **[conferir na versão instalada]**. Mandar lotes grandes (500–1.000 prompts) de uma vez, para o vLLM agendar.
5. **`_aspect_probs(output)`:** percorrer os tokens gerados, achar a posição de cada valor (logo após `"<aspecto>": "`) e, dos `logprobs` daquela posição, somar a massa dos tokens que iniciam cada um dos 4 rótulos e renormalizar.
6. **`label_texts` retorna `(rotulos, probs, raws)`.** Remover o fallback "inválido → ausente" no notebook. Com decodificação restrita isso não acontece, e no `hf_generator` o certo é fazer *retry* ou marcar NaN e excluir do treino.
7. **Few-shot kNN:** `select_examples(texto, pool, k=6, embedder)`. Embedding com um encoder de sentenças multilíngue (ou mean-pool do BERTimbau), top-k por cosseno, **forçando ao menos 1 exemplo com cada aspecto presente** e ordem crescente de similaridade. Atenção: com kNN o prefixo deixa de ser comum e perde-se o ganho do prefix caching. Medir na validação se o ganho de qualidade compensa. Alternativa barata: 12 exemplos fixos com cobertura garantida.
8. **Variantes de prompt** (`PROMPT_VARIANTS = {"v1":..., "v2":..., "v3":...}`) para a faixa de sensibilidade (Sclar) e para o voto de 3.
9. **`openai_compatible_generator`:** padrão `model="sabiazinho-4"`; concorrência (ThreadPool de 4–8) ou Batch API (−50%); `response_format` só se a Maritaca suportar JSON Schema **[não verificado]**, senão parse + *retry*.
10. **`hf_generator` (fallback):** ordenar prompts por comprimento antes de montar lotes, `attn_implementation="sdpa"`, `batch_size` 16–32.

### 7.3 `notebooks/04_rotulagem_llm.ipynb`

1. Dividir a validação em `val_demo` e `val_dev` (§6.2). Hoje são 6 exemplos aleatórios, que podem nem cobrir todos os aspectos.
2. **Seção "ablação na val_dev"**, barata (~600 textos por variante): {7B, 14B, Sabiazinho-4} × {fixo, kNN} × {com/sem nota} × {restrito/livre}. Escolher **uma** configuração e congelar (registrar `PROMPT_VERSION`).
3. Reportar κ por aspecto contra a referência do RePro (0,68/0,71), a matriz de confusão e a curva de calibração.
4. Teste de contaminação (completar trechos de 100 textos do teste).
5. Rotular o treino com a configuração congelada **+ 3 variantes** (para voto e confiança). Salvar `repro_treino_rotulos_llm.parquet` com rótulo hard, probabilidades (5×4), concordância e modelo.
6. Registrar tempo por 1.000 textos e tokens (para a §8).

### 7.4 `notebooks/05_encoders.ipynb` (e `models/encoder.py`)

1. Loop `for cond in CONDICOES for seed in SEEDS` com retomada: pular se `models/{cond}_{seed}.pt` e a predição já existirem no Drive.
2. `encoder.py`: aceitar alvo **suave** (tensor n×5×4) com perda KL/CE suave, `label_smoothing` (o `nn.CrossEntropyLoss` já aceita o parâmetro) e uma opção de GCE. Aceitar `sample_weight` (para ponderar pela confiança como alternativa a filtrar).
3. Subamostras humanas estratificadas por nota e **aninhadas** (10% ⊂ 25% ⊂ 50%), com a mesma semente em todas as condições.
4. Célula final: curva de aprendizado com faixa ± dp, linhas horizontais para as condições LLM e o cálculo do **equivalente-humano**.
5. Albertina só em H-100, L-14B e na melhor condição LLM, para caber no tempo.

---

## 8. Custos e tempo

**Volume:** treino 5.589 + teste 1.597 + `val_dev` ≈ 600 + Olist ≈ 6.000 ≈ **13.800 textos**. O comentário típico é curto (mediana de 9 palavras na Olist, README).

**Tokens por chamada [estimativa]:** instruções + definições ≈ 350 tokens; 6 exemplos ≈ 300; texto ≈ 40; saída JSON ≈ 40. Total ≈ 730 de entrada e 40 de saída. Com 3 variantes de prompt, a entrada triplica.

| Configuração | Vazão | 13.800 textos × 1 prompt | Observação |
|---|---|---|---|
| transformers + nf4, Qwen2.5-7B, lote 8 (atual) | ~1–3 s/texto **[estimativa]** | ~4–11 h | Várias sessões do Colab; o cache torna isso viável, mas é lento |
| vLLM + AWQ, Qwen2.5-7B, prefix caching | ~0,1–0,3 s/texto **[estimativa]** | ~25–70 min | Prefixo comum ≈ 90% do prompt |
| vLLM + AWQ, Qwen2.5-14B | ~0,2–0,6 s/texto **[estimativa]** | ~45 min–2,5 h | KV curto: ~4 GB livres |
| Sabiazinho-4 (API) | Limitada por *rate limit* | — | ≈ 10,1M tokens de entrada × R$ 1 + 0,55M de saída × R$ 4 ≈ **R$ 12** (≈ R$ 6 em batch) |
| Sabiá-4 (API) | idem | — | ≈ 10,1 × 5 + 0,55 × 20 ≈ **R$ 61** (≈ R$ 31 em batch) |

Preços: [Maritaca](https://docs.maritaca.ai/pt/precos). A contagem de tokens depende do tokenizador **[estimativa]**. **Ação:** piloto com 200 textos da `val_dev` em cada backend, medindo s/texto e tokens reais, antes de rodar tudo.

**Encoders [estimativa]:** o notebook 05 diz "~10 min por treino" (BERTimbau, 4 épocas, 5.589 textos). Grade da §6.3 ≈ 13 condições × 3 sementes ≈ 39 treinos. Como as subamostras são menores, isso dá ~4–6 h de T4, divididas em 3–4 sessões.

---

## 9. Riscos e mitigação

| Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|
| Desalinhamento entre o esquema do prompt e o do gabarito (definições, título, nota, polaridade por avaliação) | Alta | Alto: mede "diferença de esquema", não "qualidade do LLM" | §7.1 e §7.2-2; relatar o κ antes e depois do alinhamento |
| 14B AWQ não cabe ou dá OOM na T4 | Média | Médio | `max_model_len=1536`, `gpu_memory_utilization=0.92`, `enforce_eager=True`; senão usar o 7B |
| Instalação do vLLM quebra no Colab | Média | Médio | Fixar a versão no `pyproject` (extra `[vllm]`); manter o `hf_generator` |
| Contaminação do RePro no pré-treino | Média | Médio (infla o LLM direto no teste) | Teste de completar trechos; reportar também na amostra Olist |
| Sensibilidade a prompt | Alta | Médio | 3 variantes; relatar faixa; congelar antes do teste |
| *Overfitting* do prompt na validação | Média | Médio | Ajustar só em `val_dev`; poucas iterações, todas registradas |
| Classes raras (anuncio, neutro) com poucos exemplos no teste | Alta | IC largo | IC por bootstrap; não tirar conclusões por classe com n < 30 |
| OUTROS subdetectado pelo LLM (visto com o Sabiá-3) | Alta | Alto na tupla | Regra explícita no prompt; exemplos de OUTROS; análise por aspecto |
| Fallback "inválido → ausente" | Certa (hoje) | Médio | Decodificação restrita; *retry* |
| Cache reaproveitado entre modelos | Média | Alto (silencioso) | Modelo e versão na chave |
| Seleção de época na validação humana mascara o custo humano | Certa | Médio | Contabilizar no orçamento; controle V-only; variante com seleção na validação LLM |

---

## 10. Referências (ABNT)

ALMEIDA, T. S. et al. **PoETa v2**: toward more robust evaluation of large language models in Portuguese. arXiv:2511.17808, 2025. Disponível em: https://arxiv.org/abs/2511.17808.

CORRÊA, N. K. et al. Tucano: advancing neural text generation for Portuguese. **Patterns**, 2025. arXiv:2411.07854. Disponível em: https://arxiv.org/abs/2411.07854.

EIRAS, [?]; BRITO, [?]. Unsupervised aspect-based sentiment analysis through LLM: a case study of an unlabeled Portuguese beer database. **IEEE Access**, 2026. Disponível em: https://ipt.br/?p=66196. [nomes completos e paginação não verificados]

GEKHMAN, Z. et al. TrueTeacher: learning factual consistency evaluation with large language models. In: EMNLP, 2023. Disponível em: https://arxiv.org/abs/2305.11171.

GILARDI, F.; ALIZADEH, M.; KUBLI, M. ChatGPT outperforms crowd workers for text-annotation tasks. **Proceedings of the National Academy of Sciences**, v. 120, n. 30, e2305016120, 2023. Disponível em: https://arxiv.org/abs/2303.15056.

GOLCHIN, S.; SURDEANU, M. Time travel in LLMs: tracing data contamination in large language models. In: ICLR, 2024. Disponível em: https://arxiv.org/abs/2308.08493.

HAN, B. et al. Co-teaching: robust training of deep neural networks with extremely noisy labels. In: NeurIPS, 2018. Disponível em: https://arxiv.org/abs/1804.06872.

LIMA, J. V. R. J.; PINHEIRO, V.; CAMINHA, C. Portuguese sentiment analysis with open-source LLMs: models, prompts, and efficient deployment. In: PROPOR, 2026. p. 212-221. Disponível em: https://preview.aclanthology.org/ingest-dnd/2026.propor-1.21.pdf.

LIU, J. et al. What makes good in-context examples for GPT-3? In: DeeLIO, 2022. p. 100-114. Disponível em: https://aclanthology.org/2022.deelio-1.10.

LU, Y. et al. Fantastically ordered prompts and where to find them. In: ACL, 2022. Disponível em: https://aclanthology.org/2022.acl-long.556/.

LUKASIK, M. et al. Does label smoothing mitigate label noise? In: ICML, 2020. Disponível em: https://arxiv.org/abs/2003.02819.

MARITACA AI. **Modelos** e **Preços**. Documentação, 2026. Disponível em: https://docs.maritaca.ai/pt/modelos e https://docs.maritaca.ai/pt/precos. Acesso em: 30 out. 2026.

MIN, S. et al. Rethinking the role of demonstrations: what makes in-context learning work? In: EMNLP, 2022. Disponível em: https://arxiv.org/abs/2202.12837.

NORTHCUTT, C.; JIANG, L.; CHUANG, I. Confident learning: estimating uncertainty in dataset labels. **Journal of Artificial Intelligence Research**, v. 70, 2021. Disponível em: https://arxiv.org/abs/1911.00068.

PANGAKIS, N.; WOLKEN, S.; FASCHING, N. Automated annotation with generative AI requires validation. arXiv:2306.00176, 2023. Disponível em: https://arxiv.org/abs/2306.00176.

PORTILLO WIGHTMAN, G.; DELUCIA, A.; DREDZE, M. Strength in numbers: estimating confidence of large language models by prompt agreement. In: TrustNLP Workshop, 2023. Disponível em: https://aclanthology.org/2023.trustnlp-1.28/.

QWEN TEAM. **Qwen2.5-14B-Instruct-AWQ**; **Qwen3-8B-AWQ**; **Qwen3.5-9B**. Hugging Face, 2024-2026. Disponível em: https://huggingface.co/Qwen. Acesso em: 30 out. 2026.

GOOGLE DEEPMIND. **Gemma 4 E4B** (model card). Hugging Face, 2026. Disponível em: https://huggingface.co/google/gemma-4-E4B. Acesso em: 30 out. 2026.

SANTOS, R. et al. Advancing generative AI for Portuguese with open decoder Gervásio PT*. In: SIGUL, 2024. Disponível em: https://arxiv.org/abs/2402.18766.

SCHUCK, A. F. et al. Evaluating large language models for Brazilian Portuguese sentiment analysis: a comparative study of multilingual state-of-the-art vs. Brazilian Portuguese fine-tuned LLMs. **Journal of the Brazilian Computer Society**, v. 31, n. 1, 2025. DOI: 10.5753/jbcs.2025.5793.

SCLAR, M. et al. Quantifying language models' sensitivity to spurious features in prompt design. In: ICLR, 2024. Disponível em: https://arxiv.org/abs/2310.11324.

SILVA, L. N. dos S. et al. RePro: a benchmark dataset for opinion mining in Brazilian Portuguese. In: PROPOR, 2024. p. 432-440. Disponível em: https://aclanthology.org/2024.propor-1.44/.

SILVA, L. N. dos S.; REAL, L. Automated topic annotation in Brazilian product reviews: a case study of adversarial examples with Sabiá-3. In: STIL, 2024. Disponível em: https://aclanthology.org/2024.stil-1.61.

SIMMERING, P. F.; HUOVIALA, P. Large language models for aspect-based sentiment analysis. arXiv:2310.18025, 2023. Disponível em: https://arxiv.org/abs/2310.18025.

ŠMÍD, J.; PŘIBÁŇ, P.; KRÁL, P. LLaMA-based models for aspect-based sentiment analysis. In: WASSA, 2024. p. 63-70. Disponível em: https://aclanthology.org/2024.wassa-1.6/.

SPRAGUE, Z. et al. To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning. In: ICLR, 2025. Disponível em: https://arxiv.org/abs/2409.12183.

TAM, Z. R. et al. Let me speak freely? A study on the impact of format restrictions on performance of large language models. In: EMNLP (Industry Track), 2024. p. 1218-1236. Disponível em: https://arxiv.org/abs/2408.02442.

TAN, Z. et al. Large language models for data annotation and synthesis: a survey. In: EMNLP, 2024. Disponível em: https://arxiv.org/abs/2402.13446.

TÖRNBERG, P. Best practices for text annotation with large language models. arXiv:2402.05129, 2024. Disponível em: https://arxiv.org/abs/2402.05129.

UNSLOTH. Gemma 3: how to run and fine-tune. Documentação, 2025. Disponível em: https://docs.unsloth.ai/basics/gemma-3-how-to-run-and-fine-tune.

VLLM PROJECT. GPU installation; Quantization; Structured outputs. Documentação, 2026. Disponível em: https://docs.vllm.ai/. Acesso em: 30 out. 2026.

WANG, S. et al. Want to reduce labeling cost? GPT-3 can help. In: Findings of EMNLP, 2021. p. 4195-4205. Disponível em: https://aclanthology.org/2021.findings-emnlp.354/.

WANG, X. et al. Self-consistency improves chain of thought reasoning in language models. In: ICLR, 2023. Disponível em: https://arxiv.org/abs/2203.11171.

WANG, X. et al. Human-LLM collaborative annotation through effective verification of LLM labels. In: CHI, 2024. DOI: 10.1145/3613904.3641960.

XIA, M. et al. Prompt candidates, then distill: a teacher-student framework for LLM-driven data annotation. In: ACL, 2025. Disponível em: https://arxiv.org/abs/2506.03857.

ZHANG, W. et al. Sentiment analysis in the era of large language models: a reality check. In: Findings of NAACL, 2024. Disponível em: https://arxiv.org/abs/2305.15005.

ZHANG, Z.; SABUNCU, M. Generalized cross entropy loss for training deep neural networks with noisy labels. In: NeurIPS, 2018. Disponível em: https://arxiv.org/abs/1805.07836.

ZHU, D. et al. Weaker than you think: a critical look at weakly supervised learning. In: ACL, 2023. p. 14229-14253. Disponível em: https://aclanthology.org/2023.acl-long.796/.
