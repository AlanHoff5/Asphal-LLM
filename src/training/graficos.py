# Chapter 5 - Build a Large Language Model (from scratch) - Sebastian Raschka
# Asphal LLM - grafico das perdas de treino e validacao

# 5.2 (figura 5.12 do livro)


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, arquivo, titulo=None):
    """Desenha as perdas de treino e validacao.

    Eixo de baixo: epocas. Eixo de cima: tokens vistos (mesma curva, outra
    escala). Se a curva de validacao para de cair enquanto a de treino
    continua caindo, e sinal de overfitting.
    """
    import matplotlib
    matplotlib.use("Agg")                      # salva em arquivo, sem abrir janela
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs_seen, train_losses, label="Treino")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validacao")
    ax1.set_xlabel("Epocas")
    ax1.set_ylabel("Perda (cross-entropy)")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twiny()                          # segundo eixo x, em cima
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens vistos")

    if titulo:
        ax1.set_title(titulo)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)
    return arquivo


def plot_comparacao(series, chave, arquivo, titulo):
    """Uma curva por experimento no mesmo grafico.

    series: {nome_do_experimento: historico}, com historico["epocas"] e
    historico[chave] (por exemplo "val_losses" ou "train_losses").
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nome, hist in series.items():
        ax.plot(hist["epocas"], hist[chave], label=nome)
    ax.set_xlabel("Epocas")
    ax.set_ylabel("Perda (cross-entropy)")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150)
    plt.close(fig)
    return arquivo
