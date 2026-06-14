# Antifragilidade em Machine Learning

TCC sobre antifragilidade em redes neurais. O trabalho investiga se modelos treinados com regularização (Dropout e L2) se tornam mais robustos — e potencialmente antifrágeis — quando expostos a distribuições corrompidas (CIFAR-10-C). Dois modelos (ResNet18 e EfficientNetB0) são treinados em dois cenários (CN: treino/avaliação no CIFAR-10 limpo; CR: treino/avaliação no CIFAR-10-C corrompido) com um grid de Dropout {0.0, 0.2, 0.5} × L2 {0.0, 1e-4, 5e-4}, totalizando 36 combinações por cenário.

**Autora:** Isabella Curci de Barros
**Orientador:** Prof. Me. Andre Roberto Ortoncelli

---

## Estrutura

```
tcc/
├── scripts/
│   ├── run_experiments.py   ← pipeline principal (treinamento + avaliação)
│   └── matriz.py            ← geração das matrizes de confusão (Figuras 3–6)
├── outputs/
│   ├── {Modelo}_{Cenario}_D{dropout}_L{l2}_best.pt   ← checkpoints finais (36 arquivos)
│   ├── confusion_matrices/  ← matrizes de confusão dos 4 melhores experimentos
│   ├── phase3_v2_results.json     ← resultados CN (Tabelas 1–2)
│   └── phase3_v2_cr_results.json  ← resultados CR (Tabelas 3–4)
└── _descartado/             ← scripts e checkpoints de fases exploratórias anteriores
```

---

## Como rodar

```bash
python scripts/run_experiments.py --epochs 30
```

---

## Experimentos

| # | Modelo         | Cenário | Treino       | Avaliação    |
|---|----------------|---------|--------------|--------------|
| 1 | ResNet18       | CN      | CIFAR-10     | CIFAR-10     |
| 2 | ResNet18       | CR      | CIFAR-10-C   | CIFAR-10-C   |
| 3 | EfficientNetB0 | CN      | CIFAR-10     | CIFAR-10     |
| 4 | EfficientNetB0 | CR      | CIFAR-10-C   | CIFAR-10-C   |

Cada experimento roda o grid completo: Dropout ∈ {0.0, 0.2, 0.5} × L2 ∈ {0.0, 1e-4, 5e-4}.

---

## _descartado/

Contém scripts de fases exploratórias anteriores (`train_dropout.py`, `train_l2.py`, `eval_robustness.py`, `plot_results.py`, `antifragile_index.py`, `baseline`) e checkpoints da execução preliminar com data leakage no cenário CR (não usados nos resultados finais). Mantidos para rastreabilidade, mas não fazem parte do pipeline final.

---

## Requisitos

- Python 3.9+
- PyTorch com CUDA (`--index-url https://download.pytorch.org/whl/cu126`)
- torchvision, matplotlib, numpy, tqdm, Pillow
