"""
antifragile_index.py — Calcula indice de antifragilidade
Compara modelos em clean acc vs robust acc e gera ranking antifragil

O indice mede quao "menos frágil" o modelo e sob corrupcoes,
combinando:
  1. Clean Accuracy (CA) — baseline
  2. Mean Corruption Accuracy (mCA) — robustez media
  3. Degradation Rate (DR) — quanta performance se perde com ruido
  4. Antifragile Score (AF) — quao melhor o modelo se comporta
     comparado ao baseline em termos relativos
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path


def load_robustness_results(outputs_dir):
    """Carrega todos os JSONs de robustez do diretorio."""
    results = []
    out = Path(outputs_dir)
    for f in sorted(out.glob("*_robustness.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results


def compute_antifragile_index(clean_acc, mean_corruption_acc, clean_acc_baseline):
    """
    Calcula indice de antifragilidade.

    O DR (degradation rate) = (CA - mCA) / CA
    Quanto menor o DR, menos o modelo degrada sob corrupcao.

    AF Index compara o mCA do modelo com a baseline normalizado
    pela degradacao da baseline:
        AF = (mCA_modelo / mCA_baseline) / (1 + DR_baseline)

    Se AF > 1 -> o modelo e melhor que o baseline sob corrupcoes
    Se AF < 1 -> o modelo e pior
    """
    # Degradation rate
    dr = (clean_acc - mean_corruption_acc) / clean_acc * 100

    # mCA normalizado (0 a 1)
    mca_norm = mean_corruption_acc / 100.0

    # AF index relativo a baseline
    if clean_acc_baseline > 0:
        dr_baseline = (clean_acc_baseline - mean_corruption_acc) / clean_acc_baseline * 100
        af_score = (mean_corruption_acc / max(clean_acc, 0.01)) * 100
    else:
        af_score = 0.0

    return {
        "clean_acc": round(clean_acc, 2),
        "mean_corruption_acc": round(mean_corruption_acc, 2),
        "degradation_rate": round(dr, 2),
        "mca_normalized": round(mca_norm, 4),
        "antifragile_score": round(af_score, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Calcula indice de antifragilidade")
    parser.add_argument("--outputs_dir", type=str, default="./outputs")
    parser.add_argument("--baseline_json", type=str, default=None,
                        help="JSON do baseline para comparacao (ex: baseline_results.json)")
    parser.add_argument("--output_file", type=str, default="antifragile_index.json")
    args = parser.parse_args()

    results = load_robustness_results(args.outputs_dir)
    if not results:
        print(f"Nenhum *_robustness.json encontrado em {args.outputs_dir}")
        return

    # Carrega baseline se disponivel
    baseline_acc = None
    if args.baseline_json and os.path.exists(args.baseline_json):
        with open(args.baseline_json) as f:
            bdata = json.load(f)
        baseline_acc = bdata.get("best_clean_acc", bdata.get("clean_acc"))
        print(f"Baseline clean acc: {baseline_acc:.2f}%")

    # Calcula indice para cada modelo
    all_indices = []
    for r in results:
        clean_acc = r.get("clean_acc", 0)
        mean_ca = r.get("mean_corruption_acc", 0)
        model_name = r.get("model_name", "unknown")

        idx = compute_antifragile_index(clean_acc, mean_ca, baseline_acc or clean_acc)
        idx["model_name"] = model_name
        all_indices.append(idx)

        print(f"\n{model_name}:")
        print(f"  CA:    {idx['clean_acc']:.2f}%")
        print(f"  mCA:   {idx['mean_corruption_acc']:.2f}%")
        print(f"  DR:    {idx['degradation_rate']:.2f}%")
        print(f"  AF:    {idx['antifragile_score']:.2f}")

    # Ranking
    print(f"\n{'='*65}")
    print(f"{'RANKING POR ANTIFRAGILITY':^65}")
    print(f"{'='*65}")
    print(f"{'Modelo':<35} {'CA':>8} {'mCA':>8} {'DR%':>8} {'AF':>8}")
    print("-" * 65)
    for idx in sorted(all_indices, key=lambda x: x["antifragile_score"], reverse=True):
        print(f"{idx['model_name']:<35} {idx['clean_acc']:>7.2f}% {idx['mean_corruption_acc']:>7.2f}% {idx['degradation_rate']:>7.2f} {idx['antifragile_score']:>8.2f}")

    # Salva JSON
    output = {
        "baseline_clean_acc": baseline_acc,
        "antifragile_index": sorted(all_indices, key=lambda x: x["antifragile_score"], reverse=True),
    }
    out_path = os.path.join(args.outputs_dir, args.output_file)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSalvo: {out_path}")


if __name__ == "__main__":
    main()
