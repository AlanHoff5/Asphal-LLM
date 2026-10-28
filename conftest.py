# Configuração raiz do pytest.
# 
# Garante que os módulos e pacotes de /src, local onde está a LLM por completa
# com todas as sprints e esses módulos e pacotes, podem ser usados para testes 
# em qualquer parte do projeto.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
