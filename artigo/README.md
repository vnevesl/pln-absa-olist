# Artigo (LaTeX / Overleaf)

- `main.tex`: o artigo.
- `referencias.bib`: as referências (BibTeX).

## Importar no Overleaf

1. Overleaf → **New Project → Upload Project** → escolha `artigo_overleaf.zip`.
2. **Menu → Compiler: pdfLaTeX** (o padrão).
3. Compile. As partes em **vermelho entre colchetes** são pendências (`\pendente{...}`): resultados que dependem
   de experimento ou de decisão sua.

Antes da entrega, troque a linha `\newcommand{\pendente}[1]{\textcolor{red}{[#1]}}` por
`\newcommand{\pendente}[1]{}` só depois de conferir que não sobrou nenhuma pendência no PDF.

## Se o professor exigir o modelo da SBC

No Overleaf: **New Project → Templates → "SBC Conferences Template"**. Copie o conteúdo entre
`\begin{document}` e `\end{document}` deste `main.tex` para o modelo, suba o `referencias.bib` e troque
`\bibliographystyle{plainnat}` por `\bibliographystyle{sbc}`. O `\begin{abstract}` vira `\begin{resumo}` +
`\begin{abstract}`, conforme o modelo.

Os números das Tabelas 3 a 5 saem de `scripts/numeros_artigo.py` (rode `PYTHONPATH=src python scripts/numeros_artigo.py`).
