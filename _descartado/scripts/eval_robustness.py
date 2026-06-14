"""
eval_robustness.py — Avaliacao completa no CIFAR-10-C
Testa TODAS as 19 corrupcoes x 5 severidades + CIFAR-10 limpo
Gera JSON + CSV com resultados para gerar graficos/tabelas
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os
import json
import csv
import argparse
from datetime import datetime


# ─────────────────────────────────────────────
# CROPORRUPCOES CIFAR-10-C
# ─────────────────────────────────────────────
CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform",
    "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
    "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
    "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"
]


# ─────────────────────────────────────────────
# CIFAR-10-C Dataset
# ─────────────────────────────────────────────
class CIFAR10C(Dataset):
    def __init__(self, root, corruption, severity, transform=None):
        data_path   = os.path.join(root, f"{corruption}.npy")
        labels_path = os.path.join(root, "labels.npy")
        data   = np.load(data_path)
        labels = np.load(labels_path)
        idx_start = (severity - 1) * 10000
        idx_end   = severity * 10000
        self.data   = data[idx_start:idx_end]
        self.labels = labels[idx_start:idx_end]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        img = self.data[idx]
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(Image.fromarray(img))
        return img, label


class ResNet18Eval(nn.Module):
    def __init__(self, dropout_rate=0.0, num_classes=10):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = nn.Identity()
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for inputs, targets in tqdm(loader, desc="  Eval", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        correct += outputs.max(1)[1].eq(targets).sum().item()
        total += inputs.size(0)
    return 100.0 * correct / total


def get_clean_loader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                             download=True, transform=transform)
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)


def get_cifar10c_loader(cifar10c_dir, corruption, severity, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    dataset = CIFAR10C(cifar10c_dir, corruption=corruption,
                       severity=severity, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Avaliacao robustez completa")
    parser.add_argument("--model", type=str, required=True,
                        help="Caminho do checkpoint (.pt)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Nome para identificacao nos resultados")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Taxa de dropout do modelo (se tiver dropout layer)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--severities", type=str, default="1,2,3,4,5",
                        help="Severidades a testar")
    parser.add_argument("--corruptions", type=str, default=None,
                        help="Corrupcoes especificas (separadas por virgula). Default: TODAS")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Carrega modelo
    model = ResNet18Eval(dropout_rate=args.dropout).to(device)
    state = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    model_name = args.model_name or os.path.splitext(os.path.basename(args.model))[0]
    print(f"\nModelo: {model_name}")
    print(f"Dispositivo: {device}")

    # Severidades
    severities = [int(s) for s in args.severities.split(",")]

    # Corrupcoes
    if args.corruptions:
        corruptions = [c.strip() for c in args.corruptions.split(",")]
    else:
        corruptions = CORRUPTIONS[:]

    # CIFAR-10 limpo
    print("\nAvaliando CIFAR-10 limpo...")
    clean_loader = get_clean_loader(args.data_dir, args.batch_size)
    clean_acc = evaluate(model, clean_loader, device)
    print(f"  CIFAR-10 clean acc: {clean_acc:.2f}%")

    # CIFAR-10-C completo
    print("\nAvaliando CIFAR-10-C (todas corrupcoes x severidades)...")

    # Busca path do CIFAR-10-C
    import kagglehub
    cifar10c_path = kagglehub.dataset_download("harshadakhatu/cifar-10-c")
    cifar10c_dir = os.path.join(cifar10c_path, "CIFAR-10-C")

    results_matrix = {}
    all_entries = []

    for corruption in corruptions:
        print(f"\n  {corruption}:")
        results_matrix[corruption] = {}
        for sev in severities:
            loader = get_cifar10c_loader(cifar10c_dir, corruption, sev, args.batch_size)
            acc = evaluate(model, loader, device)
            results_matrix[corruption][f"sev{sev}"] = round(acc, 2)
            all_entries.append({
                "model": model_name,
                "corruption": corruption,
                "severity": sev,
                "accuracy": round(acc, 2),
            })
            print(f"    sev={sev}: {acc:.2f}%")

    # Media por severidade
    print(f"\n  Media por severidade:")
    avg_by_sev = {}
    for sev in severities:
        vals = [results_matrix[c][f"sev{sev}"] for c in corruptions]
        avg = np.mean(vals)
        avg_by_sev[sev] = round(avg, 2)
        print(f"    sev={sev}: {avg:.2f}%")

    # Media geral (CE)
    mean_corruption_accuracy = np.mean([
        results_matrix[c][f"sev{sev}"]
        for c in corruptions for sev in severities
    ])

    # Tabela resumo
    print(f"\n{'='*60}")
    print(f"  RESUMO: {model_name}")
    print(f"  Clean acc:         {clean_acc:.2f}%")
    print(f"  Mean corr. acc:    {mean_corruption_accuracy:.2f}%")
    print(f"  mCE (normalized):  {(1 - mean_corruption_accuracy/100)*100:.2f}")
    print(f"{'='*60}")

    # JSON
    results = {
        "model_name": model_name,
        "dropout": args.dropout,
        "clean_acc": round(clean_acc, 2),
        "mean_corruption_acc": round(mean_corruption_accuracy, 2),
        "results_matrix": results_matrix,
        "average_by_severity": avg_by_sev,
        "severities": severities,
        "corruptions_tested": corruptions,
        "timestamp": datetime.now().isoformat(),
    }
    json_path = os.path.join(args.output_dir, f"{model_name}_robustness.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # CSV
    csv_path = os.path.join(args.output_dir, f"{model_name}_robustness.csv")
    fieldnames = ["model", "corruption", "severity", "accuracy"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_entries)

    print(f"\nJSON:  {json_path}")
    print(f"CSV:   {csv_path}")


if __name__ == "__main__":
    main()
