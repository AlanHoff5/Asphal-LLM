"""Sprint 5 - Entregavel 2: avaliacao do modelo (secoes 5.1.2 e 5.1.3).

Mostra, passo a passo, como o livro mede o erro do GPT ANTES do treino:

  Parte 1  onde o modelo vai rodar (GPU ou CPU)
  Parte 2  a cross-entropy calculada "na mao" com o exemplo do livro
           ("every effort moves" / "I really like") e conferida com o PyTorch
  Parte 3  perplexity
  Parte 4  divisao do texto em treino (90%) e validacao (10%)
  Parte 5  perda e perplexity de treino e validacao do modelo sem treino

Uso (a partir da raiz do repositorio):
    python sprints/ch05/experiments/avaliacao_modelo.py
    python sprints/ch05/experiments/avaliacao_modelo.py --config mini
    python sprints/ch05/experiments/avaliacao_modelo.py --dados data/outro_texto.txt

Os numeros sao salvos em experiments/resultados/avaliacao_inicial.json.
"""

from pathlib import Path
import argparse
import json
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import get_bpe_tokenizer
from src.model.config import GPT_CONFIG_124M, GPT_CONFIG_MINI, criar_config
from src.model.generation import generate_text_simple, text_to_token_ids, token_ids_to_text
from src.model.gpt_model import GPTModel, contar_parametros
from src.training.dados import contar_tokens_do_loader, criar_loaders_treino_validacao
from src.training.dispositivo import descrever_dispositivo, diagnostico_gpu, escolher_dispositivo
from src.training.losses import (
    calc_loss_batch,
    calc_loss_loader,
    perda_chute_uniforme,
    perplexidade,
)

RESULTADOS_DIR = Path(__file__).resolve().parent / "resultados"
RESULTADOS = {}

# O livro treina o GPT-2 124M com contexto de 256 tokens (e nao 1024),
# para caber num notebook comum. "mini" e a versao reduzida do projeto.
CONFIGS = {
    "124m": criar_config(GPT_CONFIG_124M, context_length=256),
    "mini": criar_config(GPT_CONFIG_MINI),
}


def titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Parte 1 - Dispositivo
# ---------------------------------------------------------------------------
def parte1_dispositivo(preferencia):
    titulo("Parte 1 - Onde o modelo vai rodar")
    print(diagnostico_gpu())
    device = escolher_dispositivo(preferencia)
    print(f"\nUsando: {descrever_dispositivo(device)}")
    RESULTADOS["dispositivo"] = {"torch": str(torch.__version__), "device": descrever_dispositivo(device)}
    return device


# ---------------------------------------------------------------------------
# Parte 2 - Cross-entropy passo a passo (secao 5.1.2)
# ---------------------------------------------------------------------------
def parte2_cross_entropy(model, tokenizer, device):
    titulo("Parte 2 - Cross-entropy passo a passo (exemplo do livro)")

    # Duas frases de 3 tokens. O alvo e a mesma frase deslocada 1 posicao:
    # em cada posicao o modelo deveria prever o token seguinte.
    inputs = torch.tensor([[16833, 3626, 6100],    # "every effort moves"
                           [40, 1107, 588]])        # "I really like"
    targets = torch.tensor([[3626, 6100, 345],     # " effort moves you"
                            [1107, 588, 11311]])    # " really like chocolate"
    for i in range(2):
        print(f"Entrada {i + 1}: {tokenizer.decode(inputs[i].tolist())!r:28} "
              f"-> alvo: {tokenizer.decode(targets[i].tolist())!r}")

    # 1. logits: uma nota para cada palavra do vocabulario, em cada posicao
    with torch.no_grad():
        logits = model(inputs.to(device)).cpu()
    print(f"\n1. logits                {tuple(logits.shape)}  (lote, tokens, vocabulario)")

    # 2. softmax: logits -> probabilidades (cada linha soma 1)
    probas = torch.softmax(logits, dim=-1)
    print(f"2. probabilidades        {tuple(probas.shape)}  soma de uma linha = {probas[0, 0].sum():.4f}")

    # 3. o que o modelo preveria hoje (argmax) x o que deveria prever
    previsto = torch.argmax(probas, dim=-1)
    print(f"3. modelo preve:  {tokenizer.decode(previsto[0].tolist())!r}")
    print(f"   deveria ser:   {tokenizer.decode(targets[0].tolist())!r}")

    # 4. probabilidade que o modelo deu ao token CERTO em cada posicao
    p_alvo_1 = probas[0, [0, 1, 2], targets[0]]
    p_alvo_2 = probas[1, [0, 1, 2], targets[1]]
    print(f"4. prob. do alvo certo, frase 1: {p_alvo_1.tolist()}")
    print(f"   prob. do alvo certo, frase 2: {p_alvo_2.tolist()}")
    print(f"   (chute uniforme seria 1/50257 = {1 / 50257:.7f})")

    # 5. log de cada probabilidade (numeros negativos)
    log_probas = torch.log(torch.cat((p_alvo_1, p_alvo_2)))
    print(f"5. log das probabilidades:       {[round(v, 4) for v in log_probas.tolist()]}")

    # 6. media dos logs e troca de sinal = cross-entropy
    media = torch.mean(log_probas)
    perda_manual = -media
    print(f"6. media dos logs = {media.item():.4f}  ->  trocando o sinal: perda = {perda_manual.item():.4f}")

    # 7. o mesmo resultado com a funcao do PyTorch (o que usamos no treino)
    perda_pytorch = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    print(f"7. torch.nn.functional.cross_entropy:  {perda_pytorch.item():.4f}")
    print(f"   logits achatados {tuple(logits.flatten(0, 1).shape)}, alvos achatados {tuple(targets.flatten().shape)}")

    perda_lote = calc_loss_batch(inputs, targets, model, device).item()
    print(f"   calc_loss_batch (src/training):     {perda_lote:.4f}")

    RESULTADOS["exemplo_livro"] = {
        "prob_alvos_frase1": p_alvo_1.tolist(),
        "prob_alvos_frase2": p_alvo_2.tolist(),
        "media_log": media.item(),
        "perda_manual": perda_manual.item(),
        "perda_pytorch": perda_pytorch.item(),
    }
    return perda_pytorch.item()


# ---------------------------------------------------------------------------
# Parte 3 - Perplexity
# ---------------------------------------------------------------------------
def parte3_perplexity(perda, vocab_size):
    titulo("Parte 3 - Perplexity")
    ref = perda_chute_uniforme(vocab_size)
    print(f"perda do exemplo           {perda:.4f}  -> perplexity = e^{perda:.4f} = {perplexidade(perda):,.1f}")
    print(f"chute uniforme ln({vocab_size}) {ref:.4f}  -> perplexity = {perplexidade(ref):,.0f}")
    print("Leitura: o modelo sem treino esta 'em duvida' entre quase o vocabulario inteiro.")
    print("Referencia: perda 2,5 -> perplexity ~12; perda 1,0 -> perplexity ~2,7.")
    RESULTADOS["perplexity_exemplo"] = perplexidade(perda)
    RESULTADOS["perda_chute_uniforme"] = ref


# ---------------------------------------------------------------------------
# Parte 4 - Treino x validacao (secao 5.1.3)
# ---------------------------------------------------------------------------
def parte4_dados(caminho, tokenizer, cfg, batch_size):
    titulo("Parte 4 - Divisao em treino (90%) e validacao (10%)")
    texto = Path(caminho).read_text(encoding="utf-8")
    n_tokens = len(tokenizer.encode(texto))
    print(f"Arquivo: {caminho}")
    print(f"Caracteres: {len(texto):,}  |  Tokens: {n_tokens:,}")

    torch.manual_seed(123)
    train_loader, val_loader = criar_loaders_treino_validacao(
        texto, tokenizer, cfg, batch_size=batch_size, train_ratio=0.90
    )
    tok_treino = contar_tokens_do_loader(train_loader)
    tok_val = contar_tokens_do_loader(val_loader)
    print(f"Contexto (max_length = stride): {cfg['context_length']}  |  batch_size: {batch_size}")
    print(f"Treino:    {len(train_loader)} lotes  |  {tok_treino:,} tokens")
    print(f"Validacao: {len(val_loader)} lotes  |  {tok_val:,} tokens")

    x, y = next(iter(train_loader))
    print(f"Formato de um lote: entrada {tuple(x.shape)}, alvo {tuple(y.shape)}")
    print("Obs.: o drop_last=True do treino descarta o ultimo lote incompleto.")

    RESULTADOS["dados"] = {
        "arquivo": str(caminho), "caracteres": len(texto), "tokens": n_tokens,
        "contexto": cfg["context_length"], "batch_size": batch_size,
        "lotes_treino": len(train_loader), "tokens_treino": tok_treino,
        "lotes_validacao": len(val_loader), "tokens_validacao": tok_val,
    }
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Parte 5 - Perdas iniciais (modelo sem treino)
# ---------------------------------------------------------------------------
def parte5_perdas_iniciais(model, train_loader, val_loader, device, vocab_size):
    titulo("Parte 5 - Perda de treino e validacao ANTES do treino")
    model.eval()   # desliga o dropout: a avaliacao tem que ser deterministica
    perda_treino = calc_loss_loader(train_loader, model, device)
    perda_val = calc_loss_loader(val_loader, model, device)
    ref = perda_chute_uniforme(vocab_size)

    print(f"{'':<12}{'perda':>10}{'perplexity':>14}")
    print(f"{'treino':<12}{perda_treino:>10.4f}{perplexidade(perda_treino):>14,.1f}")
    print(f"{'validacao':<12}{perda_val:>10.4f}{perplexidade(perda_val):>14,.1f}")
    print(f"{'uniforme':<12}{ref:>10.4f}{perplexidade(ref):>14,.1f}")
    print("\nAs duas perdas ficam perto de ln(50257): com pesos aleatorios o modelo")
    print("praticamente chuta. Treino e validacao quase iguais = ainda nao ha overfitting.")
    print("O objetivo do treino (Entregavel 3) e fazer esses numeros cairem.")

    RESULTADOS["perdas_iniciais"] = {
        "treino": perda_treino, "validacao": perda_val,
        "perplexity_treino": perplexidade(perda_treino),
        "perplexity_validacao": perplexidade(perda_val),
    }


def parte_extra_texto(model, tokenizer, device, cfg):
    titulo("Extra - Texto gerado pelo modelo sem treino")
    torch.manual_seed(123)
    ids = text_to_token_ids("Every effort moves you", tokenizer).to(device)
    saida = generate_text_simple(model, ids, max_new_tokens=10, context_size=cfg["context_length"])
    texto = token_ids_to_text(saida.cpu(), tokenizer)
    print(repr(texto))
    print("Sem treino o texto nao faz sentido: e isso que o treinamento vai corrigir.")
    RESULTADOS["texto_sem_treino"] = texto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=sorted(CONFIGS), default="124m")
    ap.add_argument("--dados", default=str(PROJECT_ROOT / "data" / "the-verdict.txt"))
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="auto", help="auto, cpu ou cuda")
    args = ap.parse_args()

    cfg = CONFIGS[args.config]
    tokenizer = get_bpe_tokenizer()
    device = parte1_dispositivo(args.device)

    torch.manual_seed(123)
    model = GPTModel(cfg).to(device).eval()
    print(f"\nModelo: config '{args.config}' {cfg}")
    print(f"Parametros: {contar_parametros(model):,}")
    RESULTADOS["config"] = {"nome": args.config, **cfg, "parametros": contar_parametros(model)}

    perda = parte2_cross_entropy(model, tokenizer, device)
    parte3_perplexity(perda, cfg["vocab_size"])
    train_loader, val_loader = parte4_dados(args.dados, tokenizer, cfg, args.batch_size)
    parte5_perdas_iniciais(model, train_loader, val_loader, device, cfg["vocab_size"])
    parte_extra_texto(model, tokenizer, device, cfg)

    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    arquivo = RESULTADOS_DIR / f"avaliacao_inicial_{args.config}.json"
    arquivo.write_text(json.dumps(RESULTADOS, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados salvos em {arquivo.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
