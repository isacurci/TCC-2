"""
plot_results.py — Gera graficos a partir dos resultados (fase 3)
Requer matplotlib. Le os JSONs gerados, plota:
  1. Clean acc vs mCA para cada modelo
  2. Degradação por severidade
  3. Heatmap de corrupcoes x modelos
"""

import json
import os
import argparse
import matplotlib
matplotlib.use("Agg")  # sem display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_phase3(path):
    with open(path) as f:
        return json.load(f)


def plot_clean_vs_mca(data, output_dir):
    """Barplot comparando clean acc e mean corruption acc."""
    results = data["results"]
    labels = list(results.keys())
    clean_accs = [results[k]["best_test_acc"] for k in labels]
    mca_accs = [results[k].get("mean_corruption_acc", 0) for k in labels]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 5))
    ax.bar(x - w / 2, clean_accs, w, label="Clean Acc", color="#4C72B0")
    ax.bar(x + w / 2, mca_accs, w, label="Mean Corruption Acc", color="#DD8452")

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Clean Accuracy vs Mean Corruption Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 105)

    # Annotate values
    for i, (c, m) in enumerate(zip(clean_accs, mca_accs)):
        ax.text(i - w / 2, c + 0.5, f"{c:.1f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w / 2, m + 0.5, f"{m:.1f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = os.path.join(output_dir, "plot_clean_vs_mca.png")
    plt.savefig(out, dpi=150)
    print(f"Salvo: {out}")
    plt.close()


def plot_degradation_by_severity(data, output_dir):
    """Lineplot mostrando degradacao de acc conforme severidade aumenta."""
    results = data["results"]
    labels = list(results.keys())

    # Assume severities 1-5 e 19 corrupcoes
    severities = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for i, label in enumerate(labels):
        corr = results[label].get("corruption_results")
        if not corr:
            continue
        # Media por severidade
        means = []
        for sev in severities:
            key = f"sev{sev}"
            vals = [corr[c][key] for c in corr if key in corr[c]]
            means.append(np.mean(vals))
        ax.plot(severities, means, marker="o", label=label, color=colors[i])

    ax.set_xlabel("Severidade")
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_title("Degradation by Corruption Severity")
    ax.set_xticks(severities)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "plot_degradation_severity.png")
    plt.savefig(out, dpi=150)
    print(f"Salvo: {out}")
    plt.close()


def plot_heatmap(data, output_dir):
    """Heatmap: modelos x corrupcoes (media das severidades)."""
    results = data["results"]
    labels = list(results.keys())

    corruptions = list(next(iter(results.values())).get("corruption_results", {}).keys())
    if not corruptions:
        print("Nenhum corruption_results encontrado. Pulando heatmap.")
        return

    matrix = np.zeros((len(labels), len(corruptions)))
    for i, label in enumerate(labels):
        corr = results[label].get("corruption_results", {})
        for j, c in enumerate(corruptions):
            vals = list(corr.get(c, {}).values())
            matrix[i, j] = np.mean(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(max(12, len(corruptions) * 0.6), max(6, len(labels) * 0.8)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels([c.replace("_", "\n") for c in corruptions], fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate
    for i in range(len(labels)):
        for j in range(len(corruptions)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                    fontsize=6, color="black" if matrix[i, j] > 50 else "white")

    ax.set_title("Mean Corruption Accuracy by Model")
    ax.set_xlabel("Corruption Type")
    fig.colorbar(im, ax=ax, label="Accuracy (%)")

    plt.tight_layout()
    out = os.path.join(output_dir, "plot_heatmap.png")
    plt.savefig(out, dpi=150)
    print(f"Salvo: {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Gera graficos dos resultados")
    parser.add_argument("--results_json", type=str,
                        default="./outputs/phase3_results.json",
                        help="Caminho do JSON de resultados")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    if not os.path.exists(args.results_json):
        print(f"Arquivo nao encontrado: {args.results_json}")
        print("Rode run_experiments.py primeiro.")
        return

    data = load_phase3(args.results_json)
    os.makedirs(args.output_dir, exist_ok=True)

    plot_clean_vs_mca(data, args.output_dir)
    plot_degradation_by_severity(data, args.output_dir)
    plot_heatmap(data, args.output_dir)
    print("\nTodos os graficos gerados.")


if __name__ == "__main__":
    main()
