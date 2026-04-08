"""
run_experiments.py — Pipeline completo fase 3
Treina grid de (dropout, L2), salva checkpoints, roda avaliacao CIFAR-10-C.

Uso:
  python run_experiments.py --epochs 10 --dry_run   # So mostra config
  python run_experiments.py --epochs 10             # Treina tudo
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
import json
import os
import argparse
import copy
from datetime import datetime
import kagglehub


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULTS = {
    "seed": 42,
    "batch_size": 128,
    "epochs": 10,
    "lr": 0.01,
    "momentum": 0.9,
    "data_dir": "./data",
    "output_dir": "./outputs",
    "dropout_rates": [0.0, 0.1, 0.2, 0.3, 0.5],
    "l2_values": [0.0, 1e-5, 5e-4, 1e-3],
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# MODELO
# ─────────────────────────────────────────────
class ResNet18Exp(nn.Module):
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
# DATALOADERS
# ─────────────────────────────────────────────
def get_loaders(data_dir, batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                              download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                              download=True, transform=transform_test)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
    )


class CIFAR10C(Dataset):
    CORRUPTIONS = [
        "brightness", "contrast", "defocus_blur", "elastic_transform",
        "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
        "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
        "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"
    ]

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


# ─────────────────────────────────────────────
# TREINO E AVALIACAO
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in tqdm(loader, desc="    Eval", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.max(1)[1].eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def train_one_run(dropout_rate, l2_value, epochs, lr, momentum,
                  train_loader, test_loader, device, label):
    model = ResNet18Exp(dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=l2_value)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_state = 0.0, None
    history = []

    print(f"\n  Treinando: {label} (dropout={dropout_rate}, L2={l2_value})")
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_corr, t_total = 0.0, 0, 0
        for inputs, targets in tqdm(train_loader, desc=f"    Ep {epoch}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * inputs.size(0)
            t_corr += outputs.max(1)[1].eq(targets).sum().item()
            t_total += inputs.size(0)

        train_acc = 100.0 * t_corr / t_total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_acc": round(train_acc, 2),
            "test_acc": round(test_acc, 2),
        })
        print(f"    Ep {epoch:02d}: train {train_acc:.2f}% | test {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

    return best_acc, history, best_state


# ─────────────────────────────────────────────
# AVALIACAO CIFAR-10-C
# ─────────────────────────────────────────────
@torch.no_grad()
def eval_corruptions(model, cifar10c_dir, severities, batch_size, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    results = {}
    total_sum, total_n = 0.0, 0

    for corruption in CIFAR10C.CORRUPTIONS:
        results[corruption] = {}
        for sev in severities:
            dataset = CIFAR10C(cifar10c_dir, corruption=corruption, severity=sev, transform=transform)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            model.eval()
            correct, total = 0, 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                correct += model(x).max(1)[1].eq(y).sum().item()
                total += y.size(0)
            acc = 100.0 * correct / total
            results[corruption][f"sev{sev}"] = round(acc, 2)
            total_sum += acc
            total_n += 1
        avg = np.mean(list(results[corruption].values()))
        print(f"    {corruption:<20} avg={avg:.2f}%")

    return results, round(total_sum / max(total_n, 1), 2)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Pipeline completo fase 3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--dropouts", type=str, default="0.0,0.2,0.5")
    parser.add_argument("--l2_values", type=str, default="0.0,5e-4,1e-3")
    parser.add_argument("--severities", type=str, default="1,2,3,4,5")
    parser.add_argument("--dry_run", action="store_true",
                        help="Mostra config sem treinar")
    parser.add_argument("--skip_corruption", action="store_true",
                        help="Pula avaliacao CIFAR-10-C")
    args = parser.parse_args()

    dropouts = [float(x) for x in args.dropouts.split(",")]
    l2_vals = [float(x) for x in args.l2_values.split(",")]
    severities = [int(x) for x in args.severities.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    combos = []
    for d in dropouts:
        for l2 in l2_vals:
            combos.append({"dropout": d, "l2": l2})

    print(f"\n{'='*60}")
    print(f"  FASE 3: {len(combos)} experimentos")
    print(f"  Dropouts:   {dropouts}")
    print(f"  L2 values:  {l2_vals}")
    print(f"  Epocas:     {args.epochs}")
    print(f"  Severidades:{severities}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n[DRY RUN] Parando aqui. Remova --dry_run para treinar.")
        return

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")

    criterion = nn.CrossEntropyLoss()

    all_results = {}

    for combo in combos:
        d, l2 = combo["dropout"], combo["l2"]
        label = f"do{str(d).replace('.','')}_l2{l2:.0e}"

        train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)
        best_acc, history, state = train_one_run(
            dropout_rate=d, l2_value=l2, epochs=args.epochs, lr=args.lr,
            momentum=0.9, train_loader=train_loader, test_loader=test_loader,
            device=device, label=label
        )
        del train_loader, test_loader  # libera memoria

        torch.save(state, os.path.join(args.output_dir, f"{label}_best.pt"))

        entry = {
            "dropout": d,
            "l2": l2,
            "best_test_acc": round(best_acc, 2),
            "history": history,
        }

        if not args.skip_corruption:
            c10c_path = kagglehub.dataset_download("harshadakhatu/cifar-10-c")
            c10c_dir = os.path.join(c10c_path, "CIFAR-10-C")

            model = ResNet18Exp(dropout_rate=d).to(device)
            model.load_state_dict(state)
            corr_results, mca = eval_corruptions(
                model, c10c_dir, severities, args.batch_size, device
            )
            entry["corruption_results"] = corr_results
            entry["mean_corruption_acc"] = mca
            entry["degradation"] = round(best_acc - mca, 2)

            print(f"\n  -> {label}: clean={best_acc:.2f}% | mCA={mca:.2f}% | degrad={best_acc - mca:.2f} pp")

        all_results[label] = entry

    # Salva tudo
    results = {
        "method": "full_experiment_phase3",
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "dropouts": dropouts,
            "l2_values": l2_vals,
            "severities": severities,
        },
        "device": str(device),
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = os.path.join(args.output_dir, "phase3_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Tabela final
    print(f"\n\n{'='*70}")
    print(f"{'FASE 3 - RESULTADOS':^70}")
    print(f"{'='*70}")
    print(f"{'Modelo':<25} {'Clean':>8} {'mCA':>8} {'Degr':>8}")
    print("-" * 70)
    for label, r in all_results.items():
        clean = r["best_test_acc"]
        mca = r.get("mean_corruption_acc")
        deg = r.get("degradation")
        mca_str = f"{mca:.2f}%" if isinstance(mca, float) else "-"
        deg_str = f"{deg:.2f}" if isinstance(deg, float) else "-"
        print(f"{label:<25} {clean:>7.2f}% {mca_str:>8} {deg_str:>8}")

    print(f"\nSalvo: {out_path}")


if __name__ == "__main__":
    main()
