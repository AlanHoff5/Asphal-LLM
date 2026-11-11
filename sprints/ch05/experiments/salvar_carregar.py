"""Sprint 5 - Entregavel 6: salvamento e carregamento dos pesos (secao 5.4).

Usa o modelo salvo pelo treino do livro (models/ch05/treino_124m.pth) e mostra:

  Parte 1  o que tem dentro do arquivo salvo
  Parte 2  carregar o modelo e conferir que ele e O MESMO do fim do treino:
           - a perda de validacao recalculada bate com a registrada no JSON
           - o texto gerado (gulosa) e identico a ultima amostra do treino
  Parte 3  tamanho do arquivo so com os pesos x pesos + otimizador
  Parte 4  continuar o treino por mais 1 epoca a partir do arquivo

Uso (a partir da raiz do repositorio, depois de rodar treino_modelo.py):
    python sprints/ch05/experiments/salvar_carregar.py
    python sprints/ch05/experiments/salvar_carregar.py --modelo models/ch05/outro.pth --resultado resultados/outro.json
"""

from pathlib import Path
import argparse
import json
import os
import sys
import tempfile
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import get_bpe_tokenizer
from src.model.generation import generate_text_simple, text_to_token_ids, token_ids_to_text
from src.model.gpt_model import contar_parametros
from src.training.checkpoint import carregar_modelo, carregar_para_continuar_treino
from src.training.dados import criar_loaders_treino_validacao
from src.training.dispositivo import descrever_dispositivo, escolher_dispositivo
from src.training.losses import calc_loss_loader
from src.training.treino import train_model_simple

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = EXPERIMENTS_DIR / "resultados"


def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def mb(n_bytes):
    return n_bytes / 1024 ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=str(PROJECT_ROOT / "models" / "ch05" / "treino_124m.pth"))
    ap.add_argument("--resultado", default=str(RESULTADOS_DIR / "treino_124m.json"),
                    help="JSON gerado pelo treino_modelo.py para comparar")
    ap.add_argument("--dados", default=str(PROJECT_ROOT / "data" / "the-verdict.txt"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sem-continuar", action="store_true", help="pula a Parte 4")
    args = ap.parse_args()

    caminho = Path(args.modelo).resolve()
    if not caminho.exists():
        sys.exit(f"Arquivo {caminho} nao existe. Rode antes: python sprints/ch05/experiments/treino_modelo.py")
    device = escolher_dispositivo(args.device)
    tokenizer = get_bpe_tokenizer()
    registro = json.loads(Path(args.resultado).read_text(encoding="utf-8")) if Path(args.resultado).exists() else None
    resultado = {"arquivo": str(caminho.relative_to(PROJECT_ROOT)), "dispositivo": descrever_dispositivo(device)}

    # ------------------------------------------------------------------
    titulo("Parte 1 - O que tem dentro do arquivo")
    # ------------------------------------------------------------------
    inicio = time.time()
    model, checkpoint = carregar_modelo(caminho, device)
    tempo_carga = time.time() - inicio
    print(f"Arquivo: {caminho.relative_to(PROJECT_ROOT)}  ({mb(caminho.stat().st_size):,.0f} MB)")
    print(f"Chaves:  {list(checkpoint.keys())}")
    print(f"cfg:     {checkpoint['cfg']}")
    print(f"Pesos (state_dict): {len(checkpoint['model_state_dict'])} tensores, "
          f"{contar_parametros(model):,} parametros")
    print(f"Exemplos de nomes: {list(checkpoint['model_state_dict'])[:3]}")
    print(f"Carregado em {tempo_carga:.1f} s no dispositivo {descrever_dispositivo(device)}")
    resultado["chaves"] = list(checkpoint.keys())
    resultado["tamanho_mb"] = mb(caminho.stat().st_size)

    # ------------------------------------------------------------------
    titulo("Parte 2 - O modelo carregado e o mesmo do fim do treino?")
    # ------------------------------------------------------------------
    # os mesmos parametros do treino, gravados no proprio arquivo
    treino = checkpoint.get("parametros", {})
    torch.manual_seed(treino.get("seed", 123))
    train_loader, val_loader = criar_loaders_treino_validacao(
        Path(args.dados).read_text(encoding="utf-8"), tokenizer, checkpoint["cfg"],
        batch_size=treino.get("batch_size", 2))
    perda_val = calc_loss_loader(val_loader, model, device)
    print(f"Perda de validacao recalculada: {perda_val:.6f}")
    if registro:
        registrada = registro["perdas"]["validacao_final"]
        print(f"Perda de validacao registrada no fim do treino: {registrada:.6f} "
              f"(diferenca {abs(perda_val - registrada):.1e})")
        resultado["perda_val_registrada"] = registrada
    resultado["perda_val_recalculada"] = perda_val

    ids = text_to_token_ids("Every effort moves you", tokenizer).to(device)
    with torch.no_grad():
        saida = generate_text_simple(model, ids, max_new_tokens=50,
                                     context_size=checkpoint["cfg"]["context_length"])
    texto = token_ids_to_text(saida.cpu(), tokenizer)
    print(f"\nTexto gerado pelo modelo carregado (gulosa, 50 tokens):\n  {texto!r}")
    if registro:
        ultima = registro["amostras"][-1]["texto"]
        igual = ultima == texto
        print(f"Igual a ultima amostra do treino? {'SIM' if igual else 'NAO'}")
        resultado["texto_igual_ao_treino"] = igual
    resultado["texto_gerado"] = texto

    # ------------------------------------------------------------------
    titulo("Parte 3 - So os pesos x pesos + otimizador")
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as pasta:
        so_pesos = Path(pasta) / "so_pesos.pth"
        torch.save(model.state_dict(), so_pesos)          # o model.pth do livro
        tamanho_pesos = so_pesos.stat().st_size
    tamanho_total = caminho.stat().st_size
    print(f"So os pesos (model.state_dict()):         {mb(tamanho_pesos):>8,.0f} MB")
    print(f"Pesos + otimizador AdamW (+ cfg):         {mb(tamanho_total):>8,.0f} MB")
    print(f"O otimizador ocupa {mb(tamanho_total - tamanho_pesos):,.0f} MB: o AdamW guarda duas medias")
    print("por peso. So e preciso salva-lo para CONTINUAR o treino; para gerar texto, bastam os pesos.")
    resultado["tamanho_so_pesos_mb"] = mb(tamanho_pesos)

    # ------------------------------------------------------------------
    if not args.sem_continuar:
        titulo("Parte 4 - Continuar o treino a partir do arquivo (mais 1 epoca)")
        # --------------------------------------------------------------
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model, optimizer, checkpoint = carregar_para_continuar_treino(
            caminho, device, lr=treino.get("lr", 0.0004), weight_decay=treino.get("weight_decay", 0.1))
        estado = optimizer.state_dict()["state"]
        print(f"Otimizador restaurado: estado de {len(estado)} tensores de pesos "
              f"(passo do AdamW = {int(next(iter(estado.values()))['step'])})")
        model.eval()
        antes = calc_loss_loader(train_loader, model, device)
        historico = train_model_simple(model, train_loader, val_loader, optimizer, device,
                                       num_epochs=1, eval_freq=5, eval_iter=5,
                                       start_context="Every effort moves you", tokenizer=tokenizer)
        model.eval()
        depois = calc_loss_loader(train_loader, model, device)
        print(f"Perda de treino: {antes:.3f} -> {depois:.3f} depois de mais 1 epoca")
        print("O treino continuou de onde parou (sem voltar para ~11). O arquivo original nao foi alterado.")
        resultado["continuar_treino"] = {"perda_treino_antes": antes, "perda_treino_depois": depois,
                                         "val_losses": historico["val_losses"]}

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    saida_json = RESULTADOS_DIR / "salvar_carregar.json"
    saida_json.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em {saida_json.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
