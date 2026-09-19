"""Configuracao raiz do pytest.

Garante que o pacote `src` (implementacao cumulativa da LLM, usada por
todas as sprints) seja importavel a partir de qualquer teste do projeto,
independente da pasta de sprint em que ele esteja localizado
(ex.: sprints/ch02/tests/, sprints/ch01/tests/, etc.).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
