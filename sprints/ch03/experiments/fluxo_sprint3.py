"""Demonstra o fluxo completo de atencao da Sprint 3 e roda os experimentos
sugeridos no roteiro: dimensao de embedding, numero de heads, dimensao de
head, Self-Attention vs Multi-Head Attention, mascara causal, atencao com e
sem escala, e visualizacao das matrizes de atencao.
"""

from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.attention.causal_attention import CausalAttention
from src.attention.multi_head_attention import MultiHeadAttention, MultiHeadAttentionWrapper
from src.attention.self_attention import SelfAttention_v2, attention_scores_simples
from src.data.dataloader import create_dataloader_v1
from src.data.embeddings import GPTInputEmbeddings
from src.data.tokenizer import create_tokenizer

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"

# Frases curtas do dominio de Seguranca da Informacao, para os experimentos
# com dados reais (item "diferentes sequencias de entrada").
FRASES_SEGURANCA = [
    "Um firewall bloqueia ataques que tentam explorar vulnerabilidades da rede.",
    "Um firewall bem configurado impede acessos nao autorizados a rede interna.",
    "Ataques de phishing tentam enganar usuarios para roubar credenciais de acesso.",
]


def mostrar_fluxo_basico():
    """Reproduz o fluxo minimo de atencao sobre o exemplo classico do
    livro, dos scores sem pesos treinaveis ate a saida multi-head."""
    entradas = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your
            [0.55, 0.87, 0.66],  # journey
            [0.57, 0.85, 0.64],  # starts
            [0.22, 0.58, 0.33],  # with
            [0.77, 0.25, 0.10],  # one
            [0.05, 0.80, 0.55],  # step
        ]
    )

    print("FLUXO BASICO DE ATENCAO (SPRINT 3)")

    print("\n3.3 SELF-ATTENTION SEM PESOS TREINAVEIS")
    pesos_simples, contexto_simples = attention_scores_simples(entradas)
    print("Pesos de atencao (linha 'journey'):", pesos_simples[1])
    print("Vetor de contexto de 'journey':", contexto_simples[1])

    print("\n3.4 SELF-ATTENTION COM PESOS TREINAVEIS (SCALED DOT-PRODUCT)")
    torch.manual_seed(123)
    self_attn = SelfAttention_v2(d_in=3, d_out=2)
    saida_self_attn = self_attn(entradas)
    print("Shape da saida:", tuple(saida_self_attn.shape))
    print("Saida:\n", saida_self_attn)

    print("\n3.5 CAUSAL ATTENTION (LOTE COM MASCARA)")
    lote = torch.stack([entradas])
    torch.manual_seed(123)
    causal = CausalAttention(d_in=3, d_out=2, context_length=6, dropout=0.0)
    saida_causal = causal(lote)
    print("Shape da saida:", tuple(saida_causal.shape))

    print("\n3.6 MULTI-HEAD ATTENTION (VERSAO EFICIENTE)")
    torch.manual_seed(123)
    mha = MultiHeadAttention(d_in=3, d_out=4, context_length=6, dropout=0.0, num_heads=2)
    saida_mha = mha(lote)
    print("Shape da saida:", tuple(saida_mha.shape))
    print("Numero de parametros:", sum(p.numel() for p in mha.parameters()))


def experimento_atencao_com_e_sem_escala():
    print("\nEXPERIMENTO 1 - ATENCAO COM E SEM ESCALA (SCALED DOT-PRODUCT)")
    print("Simula scores brutos em dimensao alta (d_out=100) e compara o")
    print("softmax COM e SEM a divisao por sqrt(d_out).")

    d_out = 100
    torch.manual_seed(0)
    scores_brutos = torch.randn(6, 6) * d_out**0.5

    pesos_sem_escala = torch.softmax(scores_brutos, dim=-1)
    pesos_com_escala = torch.softmax(scores_brutos / d_out**0.5, dim=-1)

    print("Maior peso medio por linha, SEM escala:", pesos_sem_escala.max(dim=-1).values.mean().item())
    print("Maior peso medio por linha, COM escala:", pesos_com_escala.max(dim=-1).values.mean().item())
    print("-> sem escala, o softmax fica mais 'saturado' (concentrado em 1 token),")
    print("   o que deixa os gradientes proximos de zero durante o treinamento.")


def experimento_dimensao_de_embedding():
    print("\nEXPERIMENTO 2 - DIFERENTES DIMENSOES DE EMBEDDING (d_out)")
    entradas = torch.rand(6, 3)

    for d_out in (2, 8, 32, 128):
        torch.manual_seed(123)
        camada = SelfAttention_v2(d_in=3, d_out=d_out)
        saida = camada(entradas)
        n_params = sum(p.numel() for p in camada.parameters())
        print(f"d_out={d_out:>4} | shape da saida={tuple(saida.shape)} | parametros={n_params}")


def experimento_numero_de_heads():
    print("\nEXPERIMENTO 3 - DIFERENTES NUMEROS DE HEADS (d_out total fixo)")
    entradas = torch.stack([torch.rand(6, 8)])
    d_out = 24  # precisa ser divisivel por todos os num_heads testados

    for num_heads in (1, 2, 3, 4, 6):
        torch.manual_seed(123)
        camada = MultiHeadAttention(d_in=8, d_out=d_out, context_length=6, num_heads=num_heads)
        saida = camada(entradas)
        print(
            f"num_heads={num_heads} | head_dim={camada.head_dim} | "
            f"shape da saida={tuple(saida.shape)} | parametros={sum(p.numel() for p in camada.parameters())}"
        )


def experimento_dimensao_de_head():
    print("\nEXPERIMENTO 4 - DIFERENTES DIMENSOES DE HEAD (num_heads fixo, d_out varia)")
    entradas = torch.stack([torch.rand(6, 8)])
    num_heads = 4

    for head_dim in (2, 4, 8, 16):
        d_out = head_dim * num_heads
        torch.manual_seed(123)
        camada = MultiHeadAttention(d_in=8, d_out=d_out, context_length=6, num_heads=num_heads)
        saida = camada(entradas)
        print(
            f"head_dim={head_dim:>3} | d_out total={d_out:>3} | "
            f"shape da saida={tuple(saida.shape)} | parametros={sum(p.numel() for p in camada.parameters())}"
        )


def experimento_self_attention_vs_multi_head():
    print("\nEXPERIMENTO 5 - SELF-ATTENTION (1 CABECA) VS MULTI-HEAD ATTENTION")
    lote = torch.stack([torch.rand(6, 8)])
    d_out = 8

    torch.manual_seed(123)
    uma_cabeca = MultiHeadAttention(d_in=8, d_out=d_out, context_length=6, num_heads=1)
    saida_uma_cabeca = uma_cabeca(lote)
    params_uma_cabeca = sum(p.numel() for p in uma_cabeca.parameters())

    torch.manual_seed(123)
    quatro_cabecas = MultiHeadAttention(d_in=8, d_out=d_out, context_length=6, num_heads=4)
    saida_quatro_cabecas = quatro_cabecas(lote)
    params_quatro_cabecas = sum(p.numel() for p in quatro_cabecas.parameters())

    print(f"1 cabeca  (head_dim={uma_cabeca.head_dim})  | parametros={params_uma_cabeca} | shape={tuple(saida_uma_cabeca.shape)}")
    print(f"4 cabecas (head_dim={quatro_cabecas.head_dim}) | parametros={params_quatro_cabecas} | shape={tuple(saida_quatro_cabecas.shape)}")
    print("-> mesmo d_out final e a mesma contagem de parametros (a diferenca")
    print("   esta em COMO o espaco de representacao e dividido internamente:")
    print("   1 cabeca aprende 1 unico 'padrao' de relacao entre tokens; 4")
    print("   cabecas podem aprender 4 padroes diferentes em paralelo.")


def experimento_mascara_causal():
    print("\nEXPERIMENTO 6 - COMPORTAMENTO DA MASCARA CAUSAL")
    entradas = torch.rand(6, 8)
    lote = torch.stack([entradas])

    torch.manual_seed(123)
    camada = CausalAttention(d_in=8, d_out=4, context_length=6, dropout=0.0)
    with torch.no_grad():
        q = camada.W_query(lote)
        k = camada.W_key(lote)
        scores = q @ k.transpose(1, 2)

        pesos_sem_mascara = torch.softmax(scores / camada.d_out**0.5, dim=-1)[0]

        scores_mascarados = scores.clone()
        scores_mascarados.masked_fill_(camada.mask.bool()[:6, :6], -torch.inf)
        pesos_com_mascara = torch.softmax(scores_mascarados / camada.d_out**0.5, dim=-1)[0]

    print("Soma dos pesos 'futuros' SEM mascara (token 0 olhando para 1..5):",
          pesos_sem_mascara[0, 1:].sum().item())
    print("Soma dos pesos 'futuros' COM mascara (deve ser 0):",
          pesos_com_mascara[0, 1:].sum().item())
    print("-> sem a mascara, um token 'enxerga' informacao de tokens futuros,")
    print("   o que invalidaria a tarefa de Next-Word Prediction durante o treino.")


def experimento_sequencias_reais_de_seguranca():
    print("\nEXPERIMENTO 7 - DIFERENTES SEQUENCIAS DE ENTRADA (corpus do projeto)")
    texto_completo = " ".join(FRASES_SEGURANCA)
    tokenizer = create_tokenizer(texto_completo)

    for frase in FRASES_SEGURANCA:
        ids = tokenizer.encode(frase)
        print(f"'{frase[:50]}...' | tokens={len(ids)}")

    max_length = 6
    dataloader = create_dataloader_v1(
        texto_completo, tokenizer, batch_size=2, max_length=max_length,
        stride=max_length, shuffle=False, drop_last=True,
    )
    inputs, _ = next(iter(dataloader))

    torch.manual_seed(123)
    embedding_dim = 16
    emb = GPTInputEmbeddings(vocab_size=len(tokenizer.str_to_int), context_length=max_length, embedding_dim=embedding_dim)
    x = emb(inputs)

    torch.manual_seed(123)
    mha = MultiHeadAttention(d_in=embedding_dim, d_out=embedding_dim, context_length=max_length, num_heads=4)
    saida = mha(x)
    print("Shape da entrada (embeddings):", tuple(x.shape))
    print("Shape da saida (multi-head attention):", tuple(saida.shape))

    return tokenizer, inputs, x, mha


def salvar_visualizacao_matriz_de_atencao(tokenizer, inputs, x, mha):
    """Gera um heatmap da matriz de atencao de cada cabeca, para a primeira
    sequencia do lote, e salva em resultados/matriz_de_atencao.png."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib nao esta instalado - pulando a visualizacao.")
        print("Instale com: pip install matplotlib")
        return

    with torch.no_grad():
        b, n, _ = x.shape
        q = mha.W_query(x).view(b, n, mha.num_heads, mha.head_dim).transpose(1, 2)
        k = mha.W_key(x).view(b, n, mha.num_heads, mha.head_dim).transpose(1, 2)
        scores = q @ k.transpose(2, 3)
        scores.masked_fill_(mha.mask.bool()[:n, :n], -torch.inf)
        pesos = torch.softmax(scores / mha.head_dim**0.5, dim=-1)

    tokens = [tokenizer.int_to_str[i] for i in inputs[0].tolist()]
    num_heads = pesos.shape[1]

    fig, eixos = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        eixos = [eixos]

    for cabeca, eixo in enumerate(eixos):
        matriz = pesos[0, cabeca].numpy()
        im = eixo.imshow(matriz, cmap="viridis", vmin=0, vmax=1)
        eixo.set_title(f"Cabeca {cabeca}")
        eixo.set_xticks(range(len(tokens)))
        eixo.set_xticklabels(tokens, rotation=45, ha="right")
        eixo.set_yticks(range(len(tokens)))
        eixo.set_yticklabels(tokens)
        eixo.set_xlabel("Key (token consultado)")
        if cabeca == 0:
            eixo.set_ylabel("Query (token que esta 'olhando')")

    fig.suptitle("Matriz de atencao por cabeca - Multi-Head Attention (mascara causal)")
    fig.tight_layout()

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    caminho = RESULTADOS_DIR / "matriz_de_atencao.png"
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    print(f"\nVisualizacao salva em: {caminho}")


if __name__ == "__main__":
    mostrar_fluxo_basico()
    experimento_atencao_com_e_sem_escala()
    experimento_dimensao_de_embedding()
    experimento_numero_de_heads()
    experimento_dimensao_de_head()
    experimento_self_attention_vs_multi_head()
    experimento_mascara_causal()
    tokenizer, inputs, x, mha = experimento_sequencias_reais_de_seguranca()
    salvar_visualizacao_matriz_de_atencao(tokenizer, inputs, x, mha)

    print("\nFLUXO DA SPRINT 3 CONCLUIDO")
