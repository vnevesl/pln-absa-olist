# Guia de aspectos

O gabarito principal do projeto é o **RePro** (Silva et al., PROPOR 2024), anotado por humanos. Este guia
documenta como os tópicos do RePro foram convertidos para o esquema do projeto e serve de regra para a
**anotação opcional de 100 comentários da Olist** (`data/annotated/olist_para_anotar.csv`).

## Esquema

Cada comentário recebe **uma** opção em **cada** um dos 5 aspectos:

| Valor | Quando usar |
|---|---|
| `ausente` | O aspecto não é mencionado, nem de forma implícita |
| `positivo` | O cliente fala bem do aspecto |
| `negativo` | O cliente fala mal do aspecto |
| `neutro` | O aspecto é mencionado sem opinião ("recebi hoje", "quando chega?") |

**A nota em estrelas não entra na decisão**: anote só o que está escrito (por isso ela não aparece no arquivo).

## Os 5 aspectos e a correspondência com o RePro

| Aspecto | Tópico no RePro | O que cobre | Exemplos |
|---|---|---|---|
| `produto` | PRODUTO | Qualidade, funcionamento, durabilidade, tamanho, aparência | "Ótima qualidade" (positivo) · "Parou de funcionar" (negativo) |
| `entrega` | ENTREGA | Prazo, atraso, rapidez, rastreio, transportadora | "Chegou antes do prazo" (positivo) · "Até hoje não recebi" (negativo) |
| `recebimento` | CONDICOESDERECEBIMENTO | Estado em que chegou: avaria, embalagem, item faltando ou trocado | "Bem embalado" (positivo) · "Caixa amassada", "veio a cor errada" (negativo) |
| `anuncio` | ANUNCIO | Fidelidade à descrição, às fotos ou ao preço anunciado | "Igual à foto" (positivo) · "Diferente do anunciado" (negativo) |
| `outros` | OUTROS | Loja ou vendedor, atendimento, troca, devolução, preço, site | "A loja resolveu rápido" (positivo) · "Ninguém responde" (negativo) |

O tópico INADEQUADA do RePro (texto que não é avaliação) não vira aspecto: os outros tópicos marcados na
mesma avaliação são mantidos.

### Conversão da polaridade do RePro

O RePro anota a polaridade da **avaliação inteira**, não de cada tópico. Por isso:

- polaridade única → vale para todos os tópicos marcados;
- polaridade mista (positivo e negativo) → a avaliação é descartada, porque não dá para saber qual aspecto é positivo e qual é negativo.

Isso remove 2.018 das 10.003 avaliações, justamente os casos do tipo "bom, mas atrasou". A amostra da Olist
anotada à mão cobre esses casos.

## Casos difíceis (para a anotação da Olist)

| Situação | Como anotar |
|---|---|
| **"Mas" no meio da frase** ("produto bom, mas demorou") | Cada parte no seu aspecto: produto **positivo**, entrega **negativo** |
| **Implícito** ("chegou rapidinho") | Conta como mencionado: entrega **positivo** |
| **Elogio genérico** ("tudo certo, gostei muito") | produto **positivo** |
| **Recomendação da loja** ("recomendo a loja") | outros **positivo** |
| **Item faltando por envio separado** | recebimento **negativo** e entrega **negativo** |
| **Ironia** ("ótimo, esperei um mês") | O sentido real: entrega **negativo** |
| **Comentário sem conteúdo** ("ok", "aaa") | Tudo **ausente** |
| **Dúvida entre dois aspectos** | O mais específico (avaria → recebimento) e registre na coluna `obs` |
