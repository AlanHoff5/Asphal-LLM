# Testes do Apendice A - Introducao ao PyTorch

import sys
from pathlib import Path

import numpy as np
import pytest
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


# ----------------
# A.2 Tensores   |
# ----------------


def test_from_numpy_compartilha_memoria_e_tensor_copia():
    ary = np.array([1, 2, 3])
    copia = torch.tensor(ary)
    compartilhado = torch.from_numpy(ary)
    ary[0] = 999
    assert copia[0].item() == 1
    assert compartilhado[0].item() == 999


def test_dtypes_padrao():
    assert torch.tensor([1, 2, 3]).dtype == torch.int64
    assert torch.tensor([1.0, 2.0, 3.0]).dtype == torch.float32
    assert torch.tensor([1, 2, 3]).to(torch.float32).dtype == torch.float32


def test_reshape_view_transposta_matmul():
    t = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert t.reshape(3, 2).tolist() == [[1, 2], [3, 4], [5, 6]]
    assert torch.equal(t.view(3, 2), t.reshape(3, 2))
    assert t.T.shape == (3, 2)
    assert (t @ t.T).tolist() == [[14, 32], [32, 77]]
    # a transposta nao e contigua: view falha, reshape funciona
    assert not t.T.is_contiguous()
    with pytest.raises(RuntimeError):
        t.T.view(6)
    assert t.T.reshape(6).tolist() == [1, 4, 2, 5, 3, 6]


# -------------------------------
# A.3 e A.4 Grafo e autograd    |
# -------------------------------


def test_autograd_valores_do_livro_e_derivada_a_mao():
    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    a = torch.sigmoid(x1 * w1 + b)
    loss = F.binary_cross_entropy(a, y)

    grad_L_w1 = grad(loss, w1, retain_graph=True)[0]
    loss.backward()

    assert loss.item() == pytest.approx(0.0852, abs=1e-4)
    assert w1.grad.item() == pytest.approx(-0.0898, abs=1e-4)     # valor do livro
    assert b.grad.item() == pytest.approx(-0.0817, abs=1e-4)
    assert torch.allclose(grad_L_w1, w1.grad)
    # sigmoid + BCE: dL/dz = a - y
    assert w1.grad.item() == pytest.approx(((a - y) * x1).item(), abs=1e-6)


def test_backward_acumula_gradientes():
    w = torch.tensor([2.0], requires_grad=True)
    (w * 3).sum().backward()
    (w * 3).sum().backward()
    assert w.grad.item() == pytest.approx(6.0)     # 3 + 3: por isso zero_grad()


# ---------------------------------
# A.5 Perceptron multicamadas     |
# ---------------------------------


def test_numero_de_parametros_do_livro():
    model = NeuralNetwork(50, 3)
    assert contar_parametros_treinaveis(model) == 2213 == parametros_esperados(50, 3)


def test_forward_e_softmax():
    torch.manual_seed(123)
    model = NeuralNetwork(50, 3)
    with torch.no_grad():
        logits = model(torch.rand(4, 50))
    assert logits.shape == (4, 3)
    assert torch.allclose(torch.softmax(logits, dim=1).sum(dim=1), torch.ones(4))


def test_no_grad_nao_monta_grafo():
    model = NeuralNetwork(2, 2)
    x = torch.rand(1, 2)
    assert model(x).grad_fn is not None
    with torch.no_grad():
        assert model(x).grad_fn is None


# ------------------------------
# A.6 Dataset e DataLoader     |
# ------------------------------


def test_toy_dataset():
    X_train, y_train, X_test, y_test = dados_toy()
    ds = ToyDataset(X_train, y_train)
    assert len(ds) == 5 and len(ToyDataset(X_test, y_test)) == 2
    x, y = ds[3]
    assert x.tolist() == pytest.approx([2.3, -1.1]) and y.item() == 1


@pytest.mark.parametrize("drop_last, tamanhos", [(False, [2, 2, 1]), (True, [2, 2])])
def test_drop_last(drop_last, tamanhos):
    train_loader, _ = criar_loaders_toy(batch_size=2, drop_last=drop_last)
    assert [len(y) for _, y in train_loader] == tamanhos


# ---------------------------------
# A.7 Laco de treino e acuracia   |
# ---------------------------------


def test_treino_reproduz_o_livro():
    train_loader, test_loader = criar_loaders_toy()
    torch.manual_seed(123)
    model = NeuralNetwork(2, 2)
    perdas = treinar_rede(model, train_loader, torch.optim.SGD(model.parameters(), lr=0.5),
                          num_epochs=3, verbose=False)
    # livro: 0.75, 0.65, 0.44, 0.13, 0.03, 0.00
    assert [round(p, 2) for p in perdas] == [0.75, 0.65, 0.44, 0.13, 0.03, 0.00]
    assert compute_accuracy(model, train_loader) == 1.0
    assert compute_accuracy(model, test_loader) == 1.0


def test_compute_accuracy_com_rede_fixa():
    class Fixa(torch.nn.Module):          # sempre prevê a classe 0
        def forward(self, x):
            return torch.tensor([[1.0, 0.0]]).repeat(len(x), 1)
    _, test_loader = criar_loaders_toy()  # rotulos de teste: [0, 1]
    assert compute_accuracy(Fixa(), test_loader) == 0.5


# ----------------------------------
# A.8 Salvar e carregar            |
# ----------------------------------


def test_salvar_e_carregar_state_dict(tmp_path):
    torch.manual_seed(123)
    model = NeuralNetwork(2, 2)
    arquivo = tmp_path / "model.pth"
    torch.save(model.state_dict(), arquivo)

    torch.manual_seed(999)
    novo = NeuralNetwork(2, 2)            # pesos diferentes antes de carregar
    novo.load_state_dict(torch.load(arquivo, weights_only=True))
    x = torch.rand(3, 2)
    with torch.no_grad():
        assert torch.equal(model(x), novo(x))


def test_carregar_em_arquitetura_diferente_falha(tmp_path):
    arquivo = tmp_path / "model.pth"
    torch.save(NeuralNetwork(2, 2).state_dict(), arquivo)
    with pytest.raises(RuntimeError):
        NeuralNetwork(3, 2).load_state_dict(torch.load(arquivo, weights_only=True))


# -------------------------------
# A.9 Dispositivo               |
# -------------------------------


def test_treino_no_dispositivo_disponivel():
    from src.training.dispositivo import escolher_dispositivo
    device = escolher_dispositivo()
    train_loader, test_loader = criar_loaders_toy()
    torch.manual_seed(123)
    model = NeuralNetwork(2, 2)
    treinar_rede(model, train_loader, torch.optim.SGD(model.parameters(), lr=0.5),
                 device=device, num_epochs=3, verbose=False)
    assert next(model.parameters()).device.type == device.type
    assert compute_accuracy(model, test_loader, device=device) == 1.0
