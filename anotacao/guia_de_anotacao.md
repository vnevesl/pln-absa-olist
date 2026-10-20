# Guia de anotação: aspectos em avaliações da Olist

Para cada comentário, marque **uma** opção em **cada** um dos 5 aspectos:

| Valor | Quando usar |
|---|---|
| `ausente` | O aspecto não é mencionado, nem de forma implícita |
| `positivo` | O cliente fala bem do aspecto |
| `negativo` | O cliente fala mal do aspecto |
| `neutro` | O aspecto é mencionado, mas sem opinião ("recebi hoje", "chegou dia 10") |

Leia o comentário inteiro antes de marcar. **A nota em estrelas não entra na decisão**: anote só o que está escrito.

## Os 5 aspectos

### 1. `produto`
Qualidade, funcionamento, durabilidade, tamanho, cor, aparência do item.
- "Produto de ótima qualidade" → **positivo**
- "Parou de funcionar em uma semana" → **negativo**
- "Tudo certo, gostei muito" → **positivo** (elogio genérico sem outro alvo vai para produto)

### 2. `entrega`
Prazo, atraso, rapidez, rastreio, transportadora, Correios.
- "Chegou antes do prazo" → **positivo**
- "Até hoje não recebi" / "atrasou 10 dias" → **negativo**
- "Recebi ontem" → **neutro**

### 3. `recebimento`
Estado em que o pedido chegou: avaria, embalagem, item faltando, item trocado, quantidade errada.
- "Veio bem embalado" → **positivo**
- "Caixa amassada", "veio só 1 de 2", "mandaram a cor errada" → **negativo**

### 4. `anuncio`
Fidelidade à descrição, às fotos ou ao preço anunciado.
- "Igualzinho à foto" → **positivo**
- "Bem diferente do anunciado", "menor do que dizia a descrição" → **negativo**

### 5. `atendimento`
Contato com a loja ou vendedor, suporte, troca, devolução, reembolso.
- "A loja resolveu rápido" → **positivo**
- "Mandei e-mail e ninguém responde" → **negativo**

## Casos difíceis

| Situação | Como anotar |
|---|---|
| **"Mas" no meio da frase** ("produto bom, mas demorou") | Anote cada parte no seu aspecto: produto **positivo**, entrega **negativo** |
| **Implícito** ("chegou rapidinho") | Conta como mencionado: entrega **positivo** |
| **Recomendação** ("recomendo a loja") | atendimento **positivo** apenas se o elogio for claramente à loja; senão, produto **positivo** |
| **Item faltando por envio separado** ("veio só um, o outro ainda não chegou") | recebimento **negativo** e entrega **negativo** |
| **Ironia** ("ótimo, esperei um mês") | Anote o sentido real: entrega **negativo** |
| **Pergunta ou texto sem opinião** ("quando chega?") | entrega **neutro** |
| **Comentário vazio de conteúdo** ("ok", "aaa") | Tudo **ausente** |
| **Na dúvida entre dois aspectos** | Escolha o mais específico (avaria → recebimento, não produto) e anote a dúvida na coluna `obs` |

## Processo

1. Anote os 800 comentários do arquivo `data/annotated/gabarito_para_anotar.csv`, preenchendo as colunas `asp_*` e, se quiser, `obs`.
2. O segundo anotador recebe `data/annotated/segundo_anotador.csv` (200 comentários) **sem ver as suas respostas**.
3. Depois de ~50 comentários, revise este guia: se surgir um caso que ele não cobre, acrescente a regra aqui e reanote os afetados.
