import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import os

from run_experiments import (
    ResNet18WithDropout,
    EfficientNetB0WithAttention,
    get_cifar10_loader,
    load_cifar10c_dataset,
    split_base_image_indices,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def get_preds(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbls.numpy())
    return np.array(all_labels), np.array(all_preds)

# --- Recriar os loaders de teste exatamente como no treino (sem leakage) ---
cr_train_idx, cr_val_idx, cr_test_idx = split_base_image_indices(seed=42)

ds_noise_test = load_cifar10c_dataset(
    "./data/CIFAR-10-C", image_indices=cr_test_idx
)
loader_cr_test = DataLoader(ds_noise_test, batch_size=128, shuffle=False)

loader_cn_test = get_cifar10_loader(
    "./data", train=False, batch_size=128, indices=cr_test_idx
)

# --- Carregar e combinar resultados ---
with open("outputs/phase3_v2_results.json") as f:
    results_old = json.load(f)

with open("outputs/phase3_v2_cr_results.json") as f:
    results_cr_new = json.load(f)

# Mantem apenas CN do arquivo antigo, e usa CR do arquivo novo (sem leakage)
results = {k: v for k, v in results_old.items() if v["cenario_treino"] == "CN"}
results.update(results_cr_new)

print(f"Total combinado: {len(results)} "
      f"(CN={sum(1 for v in results.values() if v['cenario_treino']=='CN')}, "
      f"CR={sum(1 for v in results.values() if v['cenario_treino']=='CR')})")

def find_ckpt(key):
    """Tenta achar o checkpoint certo, com ou sem sufixo _noleak_cr."""
    candidates = [
        f"outputs/{key}_noleak_cr_best.pt",
        f"outputs/{key}_best.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Checkpoint nao encontrado para {key}: {candidates}")

def load_model(key):
    v = results[key]
    do = v["dropout"]
    is_effnet = "EfficientNetB0" in key
    Cls = EfficientNetB0WithAttention if is_effnet else ResNet18WithDropout
    model = Cls(num_classes=10, dropout_rate=do).to(device)
    ckpt = find_ckpt(key)
    print(f"  carregando: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model

# --- Melhor modelo de cada experimento (1 a 4), com base no F1 ---
def best_key(scenario, test_set_metric_key):
    subset = {k: v for k, v in results.items() if v["cenario_treino"] == scenario}
    return max(subset, key=lambda k: subset[k][test_set_metric_key]["f1"])

experiments = {
    "exp1": (best_key("CN", "cifar_normal"), loader_cn_test, "cifar_normal"),
    "exp2": (best_key("CN", "cifar_ruido"),  loader_cr_test, "cifar_ruido"),
    "exp3": (best_key("CR", "cifar_normal"), loader_cn_test, "cifar_normal"),
    "exp4": (best_key("CR", "cifar_ruido"),  loader_cr_test, "cifar_ruido"),
}

os.makedirs("outputs/confusion_matrices", exist_ok=True)

for exp_name, (key, loader, metric_key) in experiments.items():
    print(f"{exp_name}: melhor modelo = {key} "
          f"(F1={results[key][metric_key]['f1']:.4f})")
    model = load_model(key)
    y_true, y_pred = get_preds(model, loader)
    cm = confusion_matrix(y_true, y_pred)
    np.save(f"outputs/confusion_matrices/cm_{exp_name}_{key}.npy", cm)
    acc = (y_true == y_pred).mean()
    print(f"  acc real = {acc:.4f} | salvo cm_{exp_name}_{key}.npy")