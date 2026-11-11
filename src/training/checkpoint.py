# Chapter 5 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - salvar e carregar os pesos do modelo

# 5.4

import torch

from src.model.gpt_model import GPTModel


def salvar_checkpoint(caminho, model, optimizer=None, cfg=None, extras=None):
    """Salva o estado do treino em um unico arquivo .pth.

    - model_state_dict:     os pesos (state_dict) do modelo
    - optimizer_state_dict: o historico do AdamW; so e necessario para
                            CONTINUAR treinando (sem ele o otimizador recomeca)
    - cfg:                  a configuracao do GPT, necessaria para recriar o
                            modelo com o mesmo formato antes de carregar os pesos
    - extras:               qualquer informacao a mais (parametros do treino, perdas...)

    E o mesmo formato do livro (model_and_optimizer.pth), mais a cfg.
    """
    dados = {"model_state_dict": model.state_dict(), "cfg": dict(cfg or model.cfg)}
    if optimizer is not None:
        dados["optimizer_state_dict"] = optimizer.state_dict()
    if extras:
        dados.update(extras)
    torch.save(dados, caminho)
    return caminho


def carregar_modelo(caminho, device="cpu"):
    """Recria o GPTModel e carrega os pesos salvos. Devolve (modelo, checkpoint).

    1. torch.load le o arquivo. map_location="cpu" deixa tudo na memoria RAM
       primeiro: assim a GPU (4 GB na GTX 1650) recebe so o que for usado;
    2. GPTModel(cfg) cria um modelo com a MESMA arquitetura, com pesos aleatorios;
    3. load_state_dict copia os pesos salvos para dentro dele;
    4. .to(device) leva o modelo para a GPU (ou fica na CPU);
    5. model.eval() desliga o dropout, porque o uso aqui e para inferencia.
    """
    # weights_only=True: so aceita tensores e tipos simples (seguro).
    # TorchVersion e liberado porque arquivos antigos guardaram torch.__version__.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(caminho, map_location="cpu", weights_only=True)
    model = GPTModel(checkpoint["cfg"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def carregar_para_continuar_treino(caminho, device="cpu", lr=0.0004, weight_decay=0.1):
    """Carrega modelo E otimizador para retomar o treino de onde parou
    (como no fim da secao 5.4 do livro). Devolve (modelo, otimizador, checkpoint)."""
    model, checkpoint = carregar_modelo(caminho, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("Este arquivo tem so os pesos; nao da para restaurar o otimizador.")
    # o load_state_dict do otimizador move o estado para o dispositivo dos pesos
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()
    return model, optimizer, checkpoint
