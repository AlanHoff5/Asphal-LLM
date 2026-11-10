# Chapter 5 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - escolha do dispositivo (GPU ou CPU)

# 5.2 (o livro usa: device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))

import torch


def escolher_dispositivo(preferencia="auto"):
    """Devolve o torch.device onde o modelo e os lotes vao rodar.

    "auto" tenta, nesta ordem:
      1. cuda  -> GPU NVIDIA (precisa do PyTorch instalado com CUDA)
      2. xpu   -> GPU Intel (precisa do PyTorch instalado com XPU)
      3. mps   -> GPU de Mac (Apple Silicon)
      4. cpu   -> processador (sempre funciona, mas e o mais lento)

    Tambem aceita um nome fixo, por exemplo "cpu" ou "cuda".
    """
    if preferencia != "auto":
        return torch.device(preferencia)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def descrever_dispositivo(device):
    """Texto curto para imprimir no log: qual GPU/CPU esta sendo usada."""
    if device.type == "cuda":
        nome = torch.cuda.get_device_name(device)
        memoria = torch.cuda.get_device_properties(device).total_memory / 1024 ** 3
        return f"GPU NVIDIA: {nome} ({memoria:.1f} GB)"
    if device.type == "xpu":
        return f"GPU Intel: {torch.xpu.get_device_name(device)}"
    if device.type == "mps":
        return "GPU Apple (MPS)"
    return f"CPU ({torch.get_num_threads()} threads)"


def diagnostico_gpu():
    """Mostra por que a GPU esta (ou nao esta) sendo usada.

    O caso mais comum: o PyTorch foi instalado na versao "+cpu", que nao
    tem suporte a GPU nenhuma, mesmo que o computador tenha uma.
    """
    linhas = [f"PyTorch {torch.__version__}"]
    if torch.version.cuda:
        linhas.append(f"Compilado com CUDA {torch.version.cuda}")
    elif "+cpu" in torch.__version__:
        linhas.append("Versao SO PARA CPU: esta instalacao nao consegue usar GPU.")
    linhas.append(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if hasattr(torch, "xpu"):
        linhas.append(f"torch.xpu.is_available():  {torch.xpu.is_available()}")
    linhas.append(f"Dispositivo escolhido: {descrever_dispositivo(escolher_dispositivo())}")
    return "\n".join(linhas)


if __name__ == "__main__":
    # python -m src.training.dispositivo
    print(diagnostico_gpu())
