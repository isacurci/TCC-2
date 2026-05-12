"""
plot_results.py — Gera gráficos a partir dos resultados (fase 3)
Estilo científico IEEE/NeurIPS — apto para publicação acadêmica.

Dependências obrigatórias : matplotlib, numpy, seaborn
Dependências opcionais   : scienceplots (pip install SciencePlots)
                           cmcrameri     (pip install cmcrameri)      — colormaps perceptualmente uniformes
                           adjustText    (pip install adjusttext)     — evita sobreposição de labels

Uso:
    python plot_results.py --results_json ./outputs/phase3_results.json --output_dir ./outputs
"""

from __future__ import annotations

import json
import os
import argparse
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless — deve vir antes de qualquer outro import matplotlib

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np

# Seaborn importado depois do backend Agg
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Libs opcionais — degradação graciosa se ausentes
# ─────────────────────────────────────────────────────────────────────────────
try:
    import scienceplots  # noqa: F401  — registra estilos ao importar
    _HAS_SCIENCEPLOTS = True
except ImportError:
    _HAS_SCIENCEPLOTS = False

try:
    from cmcrameri import cm as cmc  # Crameri (2023) perceptually uniform
    _HAS_CRAMERI = True
except ImportError:
    _HAS_CRAMERI = False


# ─────────────────────────────────────────────────────────────────────────────
# Estilo global — IEEE/NeurIPS-like
# ─────────────────────────────────────────────────────────────────────────────
def apply_global_style() -> None:
    """
    Configura rcParams para estilo de publicação acadêmica:
    - Fonte sans-serif hierarquizada (Helvetica/Arial/DejaVu)
    - Spines mínimas (esquerda + baixo apenas)
    - Grid suave com zorder controlado
    - DPI 300 no savefig
    - Fundo branco consistente
    """
    if _HAS_SCIENCEPLOTS:
        plt.style.use(["science", "no-latex"])
    else:
        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update({
        "font.family"          : "sans-serif",
        "font.sans-serif"      : ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size"            : 10,
        "axes.titlesize"       : 13,
        "axes.titleweight"     : "bold",
        "axes.titlepad"        : 10,
        "axes.labelsize"       : 11,
        "axes.labelweight"     : "normal",
        "xtick.labelsize"      : 9,
        "ytick.labelsize"      : 9,
        "legend.fontsize"      : 8.5,
        "legend.title_fontsize": 9,
        "axes.spines.top"      : False,
        "axes.spines.right"    : False,
        "axes.spines.left"     : True,
        "axes.spines.bottom"   : True,
        "axes.linewidth"       : 0.8,
        "axes.grid"            : True,
        "axes.axisbelow"       : True,
        "grid.color"           : "#E0E0E0",
        "grid.linewidth"       : 0.6,
        "grid.linestyle"       : "--",
        "xtick.direction"      : "out",
        "ytick.direction"      : "out",
        "xtick.major.size"     : 3.5,
        "ytick.major.size"     : 3.5,
        "xtick.major.width"    : 0.8,
        "ytick.major.width"    : 0.8,
        "legend.frameon"       : True,
        "legend.framealpha"    : 0.92,
        "legend.edgecolor"     : "#CCCCCC",
        "legend.borderpad"     : 0.5,
        "figure.dpi"           : 150,
        "savefig.dpi"          : 300,
        "savefig.bbox"         : "tight",
        "savefig.pad_inches"   : 0.05,
        "figure.facecolor"     : "white",
        "axes.facecolor"       : "white",
    })


apply_global_style()

# ─────────────────────────────────────────────────────────────────────────────
# Paleta Wong (2011) — Color Universal Design, segura para daltônicos
# ─────────────────────────────────────────────────────────────────────────────
_CUD_PALETTE = [
    "#0072B2",  # azul
    "#E69F00",  # laranja
    "#009E73",  # verde
    "#CC79A7",  # rosa
    "#56B4E9",  # ciano claro
    "#D55E00",  # vermelho-laranja
    "#F0E442",  # amarelo
    "#000000",  # preto
]

_BLUE   = "#0072B2"
_ORANGE = "#E69F00"

# Dashes distintos para separação em escala de cinza
_DASHES = [
    (None, None),
    (5, 2),
    (2, 2),
    (5, 2, 2, 2),
    (1, 1),
    (7, 2, 2, 2, 2, 2),
    (None, None),
    (5, 2),
    (2, 2),
    (5, 2, 2, 2),
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p"]


def _cycle_colors(n: int) -> list[str]:
    return [_CUD_PALETTE[i % len(_CUD_PALETTE)] for i in range(n)]


def _heatmap_cmap():
    """Crameri 'vik' se disponível, senão 'RdYlGn'."""
    if _HAS_CRAMERI:
        return cmc.vik
    return "RdYlGn"


def _relative_luminance(rgba: tuple) -> float:
    """Luminância relativa sRGB → linear (WCAG 2.1)."""
    r, g, b = rgba[:3]
    return 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_phase3(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Barplot — Clean Acc vs Mean Corruption Acc
# ─────────────────────────────────────────────────────────────────────────────
def plot_clean_vs_mca(data: dict, output_dir: str) -> None:
    """
    Barplot agrupado: Clean Accuracy vs Mean Corruption Accuracy.

    Melhorias:
    - Largura de barra adaptativa
    - Bordas brancas finas (edgecolor="white") — evita peso visual
    - Anotações com deslocamento proporcional ao range do eixo
    - Linha de referência pontilhada em 50 %
    - Minor ticks no eixo Y (leitura precisa)
    - Paleta CUD (acessível)
    - Legenda com título descritivo
    """
    results = data["results"]
    labels  = list(results.keys())
    clean   = np.array([results[k]["best_test_acc"]              for k in labels])
    mca     = np.array([results[k].get("mean_corruption_acc", 0) for k in labels])

    n = len(labels)
    x = np.arange(n)
    w = min(0.38, 0.80 / 2)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.3), 5))

    b1 = ax.bar(x - w / 2, clean, w, label="Clean Accuracy",
                color=_BLUE,   edgecolor="white", linewidth=0.5, zorder=3)
    b2 = ax.bar(x + w / 2, mca,   w, label="Mean Corruption Acc.",
                color=_ORANGE, edgecolor="white", linewidth=0.5, zorder=3)

    y_max  = max(clean.max(), mca.max()) if n else 100
    offset = y_max * 0.013

    def _annotate(bars, color):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + offset,
                    f"{h:.1f}",
                    ha="center", va="bottom",
                    fontsize=7.5, color=color, fontweight="semibold",
                )

    _annotate(b1, _BLUE)
    _annotate(b2, _ORANGE)

    ax.axhline(50, color="#999999", linewidth=0.9, linestyle=":", zorder=1, label="50 % ref.")

    ax.set_ylabel("Accuracy (%)", labelpad=6)
    ax.set_title("Clean Accuracy vs. Mean Corruption Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, y_max + y_max * 0.13)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.tick_params(axis="y", which="minor", length=2.5, color="#AAAAAA")
    ax.legend(loc="upper right", title="Metric")

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "plot_clean_vs_mca.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Lineplot — Degradação por severidade
# ─────────────────────────────────────────────────────────────────────────────
def plot_degradation_by_severity(data: dict, output_dir: str) -> None:
    """
    Lineplot: queda de acurácia por nível de severidade.

    Melhorias:
    - Paleta CUD + dashes distintos (dupla codificação cor + forma)
    - Markers com borda branca (destaque sobre grid)
    - fill_between ±1 std entre corrupções (variância informativa)
    - Grid apenas horizontal (foco na tendência vertical)
    - Labels do eixo X explícitos ("Sev. 1" … "Sev. 5")
    - Legenda externa quando há muitos modelos (>6)
    """
    results    = data["results"]
    labels     = list(results.keys())
    severities = [1, 2, 3, 4, 5]
    colors     = _cycle_colors(len(labels))
    many       = len(labels) > 6

    fig, ax = plt.subplots(figsize=(10 if many else 8, 5))

    ax.yaxis.grid(True,  linewidth=0.6, linestyle="--", color="#E0E0E0", zorder=0)
    ax.xaxis.grid(False)

    for i, label in enumerate(labels):
        corr = results[label].get("corruption_results")
        if not corr:
            continue

        means, stds = [], []
        for sev in severities:
            key  = f"sev{sev}"
            vals = [corr[c][key] for c in corr if key in corr[c]]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals)   if vals else 0.0)

        means = np.array(means, dtype=float)
        stds  = np.array(stds,  dtype=float)
        color = colors[i]
        dash  = _DASHES[i % len(_DASHES)]
        mk    = _MARKERS[i % len(_MARKERS)]

        line, = ax.plot(
            severities, means,
            marker=mk, markersize=6,
            markeredgewidth=0.9, markeredgecolor="white", markerfacecolor=color,
            linewidth=1.8, color=color, label=label, zorder=4,
        )
        if dash[0] is not None:
            line.set_dashes(dash)

        ax.fill_between(
            severities, means - stds, means + stds,
            color=color, alpha=0.10, zorder=2,
        )

    ax.set_xlabel("Corruption Severity Level", labelpad=6)
    ax.set_ylabel("Mean Accuracy (%)", labelpad=6)
    ax.set_title("Accuracy Degradation by Corruption Severity")
    ax.set_xticks(severities)
    ax.set_xticklabels([f"Sev. {s}" for s in severities])
    ax.set_xlim(0.7, 5.3)

    if many:
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.0),
                  borderaxespad=0, title="Model")
        fig.subplots_adjust(right=0.78)
    else:
        ax.legend(loc="upper right", title="Model")

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "plot_degradation_severity.png"))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Heatmap — Modelos × Corrupções
# ─────────────────────────────────────────────────────────────────────────────
def plot_heatmap(data: dict, output_dir: str) -> None:
    """
    Heatmap: acurácia média por modelo e tipo de corrupção.

    Melhorias:
    - Modelos ordenados por robustez (mean corruption acc decrescente)
    - TwoSlopeNorm centrada em 50 %: divergência visual em torno da metade da escala
    - Crameri 'vik' ou 'RdYlGn' — perceptualmente uniforme, acessível
    - Texto adaptativo por luminância WCAG 2.1 (preto ou branco conforme fundo)
    - Linhas brancas separando células (axhline/axvline)
    - Labels do eixo X: underscore → espaço, rotação moderada
    - Colorbar com ticks a cada 10 % e label descritivo
    - Tamanho de figura adaptativo
    """
    results = data["results"]

    corruptions = list(
        next(iter(results.values())).get("corruption_results", {}).keys()
    )
    if not corruptions:
        print("  ⚠  Nenhum corruption_results encontrado — heatmap ignorado.")
        return

    # Ordenar modelos por robustez
    labels_sorted = sorted(
        results.keys(),
        key=lambda k: results[k].get("mean_corruption_acc", 0),
        reverse=True,
    )

    n_m = len(labels_sorted)
    n_c = len(corruptions)
    matrix = np.zeros((n_m, n_c))

    for i, label in enumerate(labels_sorted):
        corr = results[label].get("corruption_results", {})
        for j, c in enumerate(corruptions):
            vals = list(corr.get(c, {}).values())
            matrix[i, j] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(max(13, n_c * 0.70), max(4, n_m * 0.65)))

    norm = TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
    cmap = _heatmap_cmap()

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", zorder=2)

    # Separação visual entre células
    for y in np.arange(-0.5, n_m,  1): ax.axhline(y, color="white", linewidth=0.8, zorder=3)
    for x in np.arange(-0.5, n_c + 0.5, 1): ax.axvline(x, color="white", linewidth=0.8, zorder=3)

    ax.set_xticks(range(n_c))
    ax.set_xticklabels(
        [c.replace("_", " ") for c in corruptions],
        rotation=40, ha="right", rotation_mode="anchor", fontsize=8,
    )
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(labels_sorted, fontsize=9)

    # Anotações com contraste garantido (WCAG 2.1)
    sm = ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap) if isinstance(cmap, str) else cmap)
    fontsize_cell = max(5.5, min(8.0, 80 / max(n_c, 10)))

    for i in range(n_m):
        for j in range(n_c):
            val = matrix[i, j]
            lum = _relative_luminance(sm.to_rgba(val))
            txt = "black" if lum > 0.18 else "white"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=fontsize_cell, color=txt, zorder=4)

    ax.set_title("Mean Corruption Accuracy per Model  (sorted by robustness ↓)", pad=12)
    ax.set_xlabel("Corruption type", labelpad=6)
    ax.set_ylabel("Model", labelpad=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label("Mean Accuracy (%)", fontsize=9, labelpad=8)
    cbar.set_ticks(range(0, 101, 10))
    cbar.ax.tick_params(labelsize=8)
    cbar.outline.set_linewidth(0.5)

    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "plot_heatmap.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera gráficos de publicação a partir dos resultados da Fase 3."
    )
    parser.add_argument("--results_json", type=str,
                        default="./outputs/phase3_results.json",
                        help="JSON de resultados gerado por run_experiments.py")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Diretório de saída dos PNGs")
    args = parser.parse_args()

    if not os.path.exists(args.results_json):
        print(f"[ERRO] Arquivo não encontrado: {args.results_json}")
        print("       Execute run_experiments.py primeiro.")
        return

    data = load_phase3(args.results_json)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Gerando gráficos...")
    plot_clean_vs_mca(data, args.output_dir)
    plot_degradation_by_severity(data, args.output_dir)
    plot_heatmap(data, args.output_dir)
    print("\nTodos os gráficos gerados com sucesso.")


if __name__ == "__main__":
    main()