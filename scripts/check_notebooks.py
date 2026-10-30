"""Verifica os notebooks sem executá-los (não há GPU no container).

- JSON válido no formato nbformat 4;
- toda célula de código compila (linhas com `!` ou `%` do IPython são ignoradas);
- nenhuma saída salva (notebooks entram no git limpos);
- todo `from absa... import X` aponta para um nome que existe no pacote.

Uso: python scripts/check_notebooks.py [notebooks/*.ipynb]
"""
import ast
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check(path: Path) -> list[str]:
    erros = []
    nb = json.loads(path.read_text(encoding="utf-8"))
    if nb.get("nbformat") != 4:
        erros.append("nbformat diferente de 4")
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            erros.append(f"célula {i}: tem saídas salvas")
        src = "".join(line for line in cell["source"] if not line.lstrip().startswith(("!", "%")))
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            erros.append(f"célula {i}: erro de sintaxe na linha {e.lineno}: {e.msg}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("absa"):
                try:
                    mod = importlib.import_module(node.module)
                except ImportError as e:
                    if "torch" in str(e) or "transformers" in str(e) or "lime" in str(e) or "fairlearn" in str(e):
                        continue  # dependência opcional de GPU/análise ausente no container
                    erros.append(f"célula {i}: módulo {node.module} não importa ({e})")
                    continue
                for alias in node.names:
                    if not hasattr(mod, alias.name):
                        erros.append(f"célula {i}: {node.module} não define {alias.name}")
    return erros


def main(paths):
    paths = [Path(p) for p in paths] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    falhas = 0
    for p in paths:
        erros = check(p)
        falhas += len(erros)
        print(f"{'ok  ' if not erros else 'ERRO'} {p.name}")
        for e in erros:
            print(f"     {e}")
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
