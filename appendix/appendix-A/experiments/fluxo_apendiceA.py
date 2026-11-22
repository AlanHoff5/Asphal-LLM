# Appendix A - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM

# Demonstra o fluxo completo do Apendice A (Introducao ao PyTorch) e roda
# tres experimentos curtos: taxa de aprendizado, tamanho do lote e CPU x GPU.

# Secoes reproduzidas do livro:
#   A.2 tensores (criacao, dtype, reshape/view/transposta/matmul)
#   A.3 grafo computacional (regressao logistica de 1 neuronio)
#   A.4 autograd (grad() e backward(), conferidos com a derivada feita a mao)
#   A.5 perceptron multicamadas (2.213 parametros)
#   A.6 Dataset e DataLoader (shuffle e drop_last)
#   A.7 laco de treino e acuracia
#   A.8 salvar e carregar o state_dict
#   A.9 treino na GPU (ou CPU, se nao houver GPU)

# Uso (a partir da raiz do repositorio):
#    python Apendices/appendix-A/experiments/fluxo_apendiceA.py

# Os resultados numericos sao salvos em experiments/resultados/apendiceA_resultados.json


from pathlib import Path
import json
import math
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pytorch_basico.dados_toy import ToyDataset, criar_loaders_toy, dados_toy
from src.pytorch_basico.rede_neural import (
    NeuralNetwork,
    contar_parametros_treinaveis,
    parametros_esperados,
)
from src.pytorch_basico.treino_basico import compute_accuracy, treinar_rede
from src.training.dispositivo import descrever_dispositivo, escolher_dispositivo

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS = {}          # tudo que for medido vai para o JSON no final


# -------------
# Utilitarios |
# -------------

def titulo(texto):
    print()
    print("=" * 78)
    print(texto)
    print("=" * 78)


def tabela(linhas, colunas):
    larguras = [max(len(str(c)), *(len(str(l[i])) for l in linhas)) for i, c in enumerate(colunas)]
    print("  ".join(str(c).ljust(w) for c, w in zip(colunas, larguras)))
    print("  ".join("-" * w for w in larguras))
    for l in linhas:
        print("  ".join(str(v).ljust(w) for v, w in zip(l, larguras)))


# --------------------
# A.2 Tensores       |
# --------------------

def secao_tensores():
    titulo("A.2 Tensores: escalar, vetor, matriz e tensor 3D")

    tensor0d = torch.tensor(1)
    tensor1d = torch.tensor([1, 2, 3])
    tensor2d = torch.tensor([[1, 2], [3, 4]])
    tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tabela([(n, tuple(t.shape), t.ndim, str(t.dtype)) for n, t in
            [("tensor0d", tensor0d), ("tensor1d", tensor1d),
             ("tensor2d", tensor2d), ("tensor3d", tensor3d)]],
           ["nome", "shape", "ndim", "dtype"])

    # torch.tensor COPIA o array; torch.from_numpy COMPARTILHA a memoria
    ary3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    copia = torch.tensor(ary3d)
    compartilhado = torch.from_numpy(ary3d)
    ary3d[0, 0, 0] = 999
    print(f"\nDepois de ary3d[0,0,0] = 999:")
    print(f"  torch.tensor(ary3d)[0,0,0]     = {copia[0, 0, 0].item()}   (copia: nao muda)")
    print(f"  torch.from_numpy(ary3d)[0,0,0] = {compartilhado[0, 0, 0].item()} (mesma memoria: muda)")

    # dtypes: inteiros -> int64, floats -> float32 (precisao suficiente e
    # mais rapida na GPU que float64)
    print(f"\ndtype de [1, 2, 3]       : {torch.tensor([1, 2, 3]).dtype}")
    print(f"dtype de [1.0, 2.0, 3.0] : {torch.tensor([1.0, 2.0, 3.0]).dtype}")
    print(f"depois de .to(float32)   : {torch.tensor([1, 2, 3]).to(torch.float32).dtype}")

    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"\nt.shape            = {tuple(t.shape)}")
    print(f"t.reshape(3, 2)    = {t.reshape(3, 2).tolist()}")
    print(f"t.view(3, 2)       = {t.view(3, 2).tolist()}  (view exige memoria contigua)")
    print(f"t.T                = {t.T.tolist()}")
    print(f"t @ t.T            = {(t @ t.T).tolist()}")
    print(f"t.T.is_contiguous()= {t.T.is_contiguous()}  -> t.T.view(6) daria erro; t.T.reshape(6) funciona")

    RESULTADOS["tensores"] = {
        "shapes": {"0d": [], "1d": [3], "2d": [2, 2], "3d": [2, 2, 2]},
        "from_numpy_compartilha_memoria": compartilhado[0, 0, 0].item() == 999,
        "torch_tensor_copia": copia[0, 0, 0].item() == 1,
        "matmul": (t @ t.T).tolist(),
    }


# ------------------------------
# A.3 e A.4 Grafo e autograd   |
# ------------------------------

def secao_autograd():
    titulo("A.3 + A.4 Grafo computacional e autograd (regressao logistica)")

    y = torch.tensor([1.0])                       # rotulo verdadeiro
    x1 = torch.tensor([1.1])                      # feature de entrada
    w1 = torch.tensor([2.2], requires_grad=True)  # peso
    b = torch.tensor([0.0], requires_grad=True)   # vies

    z = x1 * w1 + b                               # entrada liquida
    a = torch.sigmoid(z)                          # ativacao / saida
    loss = F.binary_cross_entropy(a, y)

    grad_L_w1 = grad(loss, w1, retain_graph=True)
    grad_L_b = grad(loss, b, retain_graph=True)
    loss.backward()

    # Derivada feita a mao: para sigmoid + BCE, dL/dz = a - y
    #   dL/dw1 = (a - y) * x1      dL/db = (a - y)
    manual_w1 = ((a - y) * x1).item()
    manual_b = (a - y).item()

    print(f"z = {z.item():.4f}  a = sigmoid(z) = {a.item():.4f}  perda = {loss.item():.4f}")
    tabela([
        ("dL/dw1", f"{grad_L_w1[0].item():.4f}", f"{w1.grad.item():.4f}", f"{manual_w1:.4f}"),
        ("dL/db", f"{grad_L_b[0].item():.4f}", f"{b.grad.item():.4f}", f"{manual_b:.4f}"),
    ], ["gradiente", "torch.autograd.grad", ".backward()", "a mao: (a-y)*x"])

    RESULTADOS["autograd"] = {
        "perda": loss.item(), "dL_dw1": w1.grad.item(), "dL_db": b.grad.item(),
        "manual_dL_dw1": manual_w1, "manual_dL_db": manual_b,
    }


# ---------------------------------
# A.5 Perceptron multicamadas     |
# ---------------------------------

def secao_rede():
    titulo("A.5 Perceptron multicamadas: NeuralNetwork(50, 3)")

    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)
    print(model)
    n = contar_parametros_treinaveis(model)
    print(f"\nParametros treinaveis: {n:,}  (conta a mao: {parametros_esperados(50, 3):,})")
    print(f"Shape dos pesos da 1a camada: {tuple(model.layers[0].weight.shape)}  (saida x entrada)")

    torch.manual_seed(123)
    X = torch.rand((1, 50))
    with torch.no_grad():
        logits = model(X)
        probas = torch.softmax(logits, dim=1)
    print(f"logits  = {[round(v, 4) for v in logits[0].tolist()]}")
    print(f"softmax = {[round(v, 4) for v in probas[0].tolist()]}  (soma = {probas.sum().item():.4f})")

    RESULTADOS["rede"] = {"parametros": n, "logits": logits[0].tolist(), "softmax": probas[0].tolist()}


# ------------------------------
# A.6 Dataset e DataLoader     |
# ------------------------------

def secao_dataloader():
    titulo("A.6 Dataset e DataLoader")

    X_train, y_train, _, _ = dados_toy()
    ds = ToyDataset(X_train, y_train)
    x0, y0 = ds[0]
    print(f"len(train_ds) = {len(ds)}   train_ds[0] = ({[round(v, 2) for v in x0.tolist()]}, {y0.item()})")

    for drop_last in (False, True):
        loader, _ = criar_loaders_toy(batch_size=2, seed=123, drop_last=drop_last)
        tamanhos = [len(y) for _, y in loader]
        print(f"drop_last={drop_last!s:5}: {len(tamanhos)} lotes com tamanhos {tamanhos}")
    RESULTADOS["dataloader"] = {"lotes_sem_drop_last": 3, "lotes_com_drop_last": 2}


# ---------------------------------
# A.7 Laco de treino e acuracia   |
# ---------------------------------

def secao_treino():
    titulo("A.7 Laco de treino (SGD, lr=0.5, 3 epocas)")

    train_loader, test_loader = criar_loaders_toy()
    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    perdas = treinar_rede(model, train_loader, optimizer, num_epochs=3)

    acc_treino = compute_accuracy(model, train_loader)
    acc_teste = compute_accuracy(model, test_loader)
    print(f"\nAcuracia treino: {acc_treino:.2f}   Acuracia teste: {acc_teste:.2f}")
    RESULTADOS["treino"] = {"perdas": perdas, "acuracia_treino": acc_treino, "acuracia_teste": acc_teste}
    return model


# ----------------------------------
# A.8 Salvar e carregar o modelo   |
# ----------------------------------

def secao_salvar_carregar(model):
    titulo("A.8 Salvar e carregar o state_dict")

    X_train, _, _, _ = dados_toy()
    with tempfile.TemporaryDirectory() as tmp:
        arquivo = Path(tmp) / "model.pth"
        torch.save(model.state_dict(), arquivo)
        tamanho = arquivo.stat().st_size

        novo = NeuralNetwork(2, 2)                 # precisa ter a MESMA arquitetura
        novo.load_state_dict(torch.load(arquivo, weights_only=True))
    model.eval(); novo.eval()
    with torch.no_grad():
        iguais = torch.equal(model(X_train), novo(X_train))
    print(f"Chaves do state_dict: {list(model.state_dict())}")
    print(f"Arquivo: {tamanho:,} bytes  |  saidas identicas depois de carregar: {iguais}")
    RESULTADOS["salvar_carregar"] = {"bytes": tamanho, "saidas_identicas": iguais}


# -------------------------------
# A.9 Treino no dispositivo     |
# -------------------------------

def secao_gpu():
    titulo("A.9 Treino no dispositivo disponivel")

    device = escolher_dispositivo()
    print(f"Dispositivo: {descrever_dispositivo(device)}")
    train_loader, test_loader = criar_loaders_toy()
    torch.manual_seed(123)
    model = NeuralNetwork(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    treinar_rede(model, train_loader, optimizer, device=device, num_epochs=3, verbose=False)
    acc = compute_accuracy(model, test_loader, device=device)
    print(f"Acuracia de teste treinando em {device.type}: {acc:.2f}")
    print("Treino com varias GPUs (DDP, A.9.3) nao e reproduzido: exige 2+ GPUs e o backend")
    print("NCCL, que nao existe no Windows. Ver o README do apendice.")
    RESULTADOS["gpu"] = {"dispositivo": descrever_dispositivo(device), "acuracia_teste": acc}


# ----------------
# Experimentos   |
# ----------------

def experimento_taxa_de_aprendizado():
    titulo("Experimento 1: taxa de aprendizado (SGD, 10 epocas)")

    linhas, res = [], {}
    for lr in (0.005, 0.05, 0.5, 5.0):
        train_loader, test_loader = criar_loaders_toy()
        torch.manual_seed(123)
        model = NeuralNetwork(2, 2)
        perdas = treinar_rede(model, train_loader, torch.optim.SGD(model.parameters(), lr=lr),
                              num_epochs=10, verbose=False)
        final = perdas[-1]
        acc = compute_accuracy(model, test_loader)
        linhas.append((lr, f"{perdas[0]:.4f}", f"{final:.4f}" if math.isfinite(final) else "nan", f"{acc:.2f}"))
        res[str(lr)] = {"perda_inicial": perdas[0], "perda_final": final, "acuracia_teste": acc}
    tabela(linhas, ["lr", "perda 1o passo", "perda final", "acuracia teste"])
    print("lr pequeno demais quase nao sai do lugar; lr grande demais faz a perda divergir.")
    RESULTADOS["exp_taxa_de_aprendizado"] = res


def experimento_tamanho_do_lote():
    titulo("Experimento 2: tamanho do lote (SGD, lr=0.5, 3 epocas)")

    linhas, res = [], {}
    for bs in (1, 2, 5):
        train_loader, test_loader = criar_loaders_toy(batch_size=bs)
        torch.manual_seed(123)
        model = NeuralNetwork(2, 2)
        perdas = treinar_rede(model, train_loader, torch.optim.SGD(model.parameters(), lr=0.5),
                              num_epochs=3, verbose=False)
        acc = compute_accuracy(model, test_loader)
        linhas.append((bs, len(train_loader), len(perdas), f"{perdas[-1]:.4f}", f"{acc:.2f}"))
        res[str(bs)] = {"lotes_por_epoca": len(train_loader), "passos": len(perdas),
                        "perda_final": perdas[-1], "acuracia_teste": acc}
    tabela(linhas, ["batch_size", "lotes/epoca", "passos", "perda final", "acuracia teste"])
    print("Lotes menores = mais atualizacoes por epoca, cada uma mais ruidosa.")
    RESULTADOS["exp_tamanho_do_lote"] = res


def experimento_cpu_vs_gpu(n=2048, repeticoes=5):
    titulo(f"Experimento 3: multiplicacao de matrizes {n}x{n} (CPU x dispositivo)")

    def medir(device):
        a = torch.rand(n, n, device=device)
        b = torch.rand(n, n, device=device)
        a @ b                                       # aquecimento
        if device.type == "cuda":
            torch.cuda.synchronize()
        tempos = []
        for _ in range(repeticoes):
            t0 = time.perf_counter()
            a @ b
            if device.type == "cuda":
                torch.cuda.synchronize()            # a GPU roda assincrona
            tempos.append(time.perf_counter() - t0)
        return min(tempos) * 1000

    cpu_ms = medir(torch.device("cpu"))
    res = {"cpu_ms": cpu_ms}
    device = escolher_dispositivo()
    if device.type != "cpu":
        dev_ms = medir(device)
        res[f"{device.type}_ms"] = dev_ms
        print(f"CPU: {cpu_ms:.1f} ms   {device.type.upper()}: {dev_ms:.1f} ms   ({cpu_ms / dev_ms:.1f}x mais rapido)")
    else:
        print(f"CPU: {cpu_ms:.1f} ms   (sem GPU disponivel para comparar)")
    RESULTADOS["exp_cpu_vs_gpu"] = res


def main():
    print(f"PyTorch {torch.__version__}")
    secao_tensores()
    secao_autograd()
    secao_rede()
    secao_dataloader()
    model = secao_treino()
    secao_salvar_carregar(model)
    secao_gpu()
    experimento_taxa_de_aprendizado()
    experimento_tamanho_do_lote()
    experimento_cpu_vs_gpu()

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    arquivo = RESULTADOS_DIR / "apendiceA_resultados.json"
    arquivo.write_text(json.dumps(RESULTADOS, indent=2, ensure_ascii=False), encoding="utf-8")
    titulo("FLUXO DO APENDICE A CONCLUIDO")
    print(f"Resultados salvos em {arquivo.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
