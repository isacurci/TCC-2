# Antifragilidade em Machine Learning

TCC sobre antifragilidade em redes neurais — avaliar como modelos se beneficiam de volatilidade, erro e estresse usando regularizacao (dropout e L2) e avaliacao no dataset CIFAR-10-C (corrompido).

**Autora:** Isabella Curci de Barros
**Orientador:** Prof. Me. Andre Roberto Ortoncelli
**Universidade:** UTFPR

---

## Estrutura

```
tcc/
├── baseline            ← Script original: ResNet-18 + CIFAR-10 + avaliacao amostral no CIFAR-10-C
├── scripts/
│   ├── train_dropout.py     ← Treina com diferentes taxas de dropout
│   ├── train_l2.py          ← Treina com diferentes valores de regularizacao L2 (weight_decay)
│   ├── eval_robustness.py   ← Avalia 1 modelo em TODAS as 19 corrupcoes x 5 severidades
│   ├── run_experiments.py   ← Pipeline completo: grid (dropout x L2) + avaliacao
│   ├── plot_results.py      ← Gera graficos (barplot, degradacao, heatmap)
│   └── antifragile_index.py ← Calcula indice de antifragilidade e ranking
├── data/               ← Datasets baixados automaticamente (CIFAR-10, CIFAR-10-C)
├── outputs/            ← Checkpoints (.pt), resultados (.json), graficos (.png)
└── venv/               ← Ambiente Python
```

---

## Scripts

### `baseline` (original)
Script que valida o pipeline completo. Treina ResNet-18 no CIFAR-10 limpo por 5 epocas e avalia amostralmente em 3 corrupcoes do CIFAR-10-C.

**Uso:**
```bash
python scripts/baseline
```

**Saida:** `outputs/baseline_best.pt`, `outputs/baseline_results.json`

---

### `train_dropout.py`
Treina varios modelos com diferentes taxas de dropout (default: 0.0, 0.1, 0.2, 0.3, 0.5) para estudar absorcao de erro como mecanismo antifragil. Cada modelo e treinado e avaliado no CIFAR-10 limpo.

**Uso:**
```bash
# Valores padrao (5 taxas de dropout)
python scripts/train_dropout.py

# Customizado
python scripts/train_dropout.py --epochs 20 --rates 0.0,0.3,0.5
```

**Saida:** Checkpoints por taxa (`outputs/dropout_XX_best.pt`) + `outputs/dropout_scan_results.json`

---

### `train_l2.py`
Treina varios modelos com diferentes valores de regularizacao L2 / weight_decay (default: 0.0, 1e-5, 5e-4, 1e-3, 5e-3). Same idea: comparar impacto na robustez.

**Uso:**
```bash
python scripts/train_l2.py

# Customizado
python scripts/train_l2.py --epochs 20 --l2_values 0.0,1e-4,1e-3
```

**Saida:** Checkpoints por valor (`outputs/l2_Xe+XX_best.pt`) + `outputs/l2_scan_results.json`

---

### `eval_robustness.py`
Avaliacao completa de **um modelo** no CIFAR-10-C: teste em todas as 19 corrupcoes x 5 niveis de severidade (95 avaliacoes). Gera JSON e CSV.

**Uso:**
```bash
python scripts/eval_robustness.py --model outputs/baseline_best.pt --model_name baseline

# Com modelo que tem dropout
python scripts/eval_robustness.py --model outputs/do02_l25e-04_best.pt --dropout 0.2 --model_name do02_l25e-04
```

**Saida:** `outputs/{model}_robustness.json`, `outputs/{model}_robustness.csv`

---

### `run_experiments.py` (recomendado)
Pipeline completo da Fase 3. Faz tudo de uma vez:

1. Treina grid de combinacoes (dropout x L2)
2. Avalia cada modelo no CIFAR-10-C completo (19 corrupcoes x 5 severidades)
3. Gera tabela comparativa com degradacao (clean - mCA)

**Uso:**
```bash
# Padrao: 3 dropouts x 3 L2 = 9 modelos
python scripts/run_experiments.py --epochs 10

# Customizado
python scripts/run_experiments.py --epochs 20 --dropouts 0.0,0.1,0.2,0.3,0.5 --l2_values 0.0,1e-5,5e-4,1e-3

# Dry run (so mostra config sem treinar)
python scripts/run_experiments.py --dry_run
```
**Saida:** Checkpoints de todos modelos + `outputs/phase3_results.json`

---

### `plot_results.py`
Gera graficos PNG a partir dos resultados do `run_experiments.py`:

1. **`plot_clean_vs_mca.png`** — Barplot: clean accuracy vs mean corruption accuracy
2. **`plot_degradation_severity.png`** — Lineplot: como a accuracy degrada conforme severidade aumenta
3. **`plot_heatmap.png`** — Heatmap: modelo x tipo de corrupcao

**Uso:**
```bash
python scripts/plot_results.py
```

---

### `antifragile_index.py`
Calcula indice de antifragilidade e gera ranking comparativo entre modelos:

- **Clean Acc (CA)** — performance baseline
- **Mean Corruption Acc (mCA)** — robustez media
- **Degradation Rate (DR)** — quanto perde sob corrupcao
- **Antifragile Score (AF)** — ranking relativo

**Uso:**
```bash
# Le todos os *_robustness.json do outputs/
python scripts/antifragile_index.py

# Com baseline para comparacao
python scripts/antifragile_index.py --baseline_json outputs/baseline_results.json
```
**Saida:** `outputs/antifragile_index.json`

---

## Fluxo recomendado

```bash
# 1. Validar pipeline (se ainda nao rodou)
python scripts/baseline

# 2. Treinar tudo + avaliar (o maior passo)
python scripts/run_experiments.py --epochs 10

# 3. Gerar graficos
python scripts/plot_results.py

# 4. (Opcional) Ranking de antifragilidade
python scripts/antifragile_index.py --baseline_json outputs/baseline_results.json
```

---

## Requisitos

- Python 3.9+
- PyTorch com CUDA (`--index-url https://download.pytorch.org/whl/cu126`)
- torchvision, kagglehub, matplotlib, numpy, tqdm, PIL/Pillow

---

## Dataset

- **CIFAR-10** — baixado automaticamente pelo torchvision
- **CIFAR-10-C** — baixado via kagglehub (`harshadakhatu/cifar-10-c`)

19 tipos de corrupcao: gaussian_noise, shot_noise, impulse_noise, speckle_noise, gaussian_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression, saturate, defocus_blur, spatter.
5 niveis de severidade (1-5).
