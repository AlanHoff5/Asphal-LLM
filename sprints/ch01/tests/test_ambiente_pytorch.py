"""Sprint 0-1 (Capitulo 1) - valida se o ambiente PyTorch esta configurado corretamente.

O professor apontou que o repositorio nao tinha nenhum teste que confirmasse
o uso da GPU. Este arquivo cobre isso: verifica se o PyTorch enxerga uma
placa de video e se ela realmente consegue executar operacoes (nao apenas
se `is_available()` retorna True).

O teste e agnostico ao fabricante da GPU:
- NVIDIA / CUDA (Windows, Linux)
- AMD com ROCm (Linux) -> tambem aparece como backend "cuda" no PyTorch
- Apple Silicon (MPS)
- Sem GPU -> cai para CPU e o teste passa normalmente

Ou seja, roda em qualquer maquina, mas quando existe uma GPU disponivel,
ele prova que o codigo consegue de fato rodar nela - exatamente como o
livro faz a partir do Capitulo 5 (torch.device dinamico, movendo tensores
e o modelo com `.to(device)`).
"""

import torch
import pytest


def selecionar_device():
    """Escolhe o melhor dispositivo disponivel, em ordem de preferencia.

    Retorna a tupla (torch.device, rotulo legivel) para uso nos testes e
    para ser reaproveitada pelo restante do projeto (treino/inferencia),
    seguindo o mesmo padrao usado no livro do Raschka.
    """
    if torch.cuda.is_available():
        indice = torch.cuda.current_device()
        nome = torch.cuda.get_device_name(indice)
        return torch.device("cuda", indice), f"cuda:{indice} ({nome})"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps"), "mps (Apple Silicon GPU)"

    return torch.device("cpu"), "cpu"


def test_pytorch_instalado():
    # Sprint 0, item 2: "todos os integrantes do grupo deverao conseguir
    # executar programas em Python utilizando o PyTorch".
    assert torch.__version__


def test_dispositivo_disponivel_e_funcional():
    """Confirma que o dispositivo escolhido de fato executa operacoes."""
    device, rotulo = selecionar_device()
    print(f"\nDispositivo selecionado automaticamente: {rotulo}")

    a = torch.rand(64, 64, device=device)
    b = torch.rand(64, 64, device=device)
    c = a @ b

    assert c.device.type == device.type
    assert c.shape == (64, 64)
    assert torch.isfinite(c).all()


def test_modelo_roda_no_dispositivo_selecionado():
    """Reproduz o padrao do livro: mover um modulo inteiro com `.to(device)`."""
    device, _ = selecionar_device()

    modelo = torch.nn.Linear(16, 4).to(device)
    entrada = torch.rand(8, 16, device=device)
    saida = modelo(entrada)

    assert saida.device.type == device.type
    assert saida.shape == (8, 4)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Nenhuma GPU CUDA/ROCm disponivel nesta maquina",
)
def test_gpu_reconhecida_quando_disponivel():
    """So roda quando ha GPU: prova que ela e reconhecida e utilizavel."""
    nome = torch.cuda.get_device_name(0)
    print(f"\nGPU detectada: {nome}")
    assert torch.cuda.device_count() >= 1

    tensor_gpu = torch.rand(4, 4, device="cuda")
    assert tensor_gpu.is_cuda


if __name__ == "__main__":
    device, rotulo = selecionar_device()
    print(f"Dispositivo selecionado: {rotulo}")
    print("PyTorch:", torch.__version__)
    print("CUDA disponivel:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
