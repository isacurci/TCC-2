"""
train_dropout.py — Treino com diferentes taxas de dropout
Objetivo: avaliar absorcao de erro via dropout como mecanismo antifragil
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import os
import argparse
from datetime import datetime
import copy


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "seed": 42,
    "batch_size": 128,
    "epochs": 10,
    "lr": 0.01,
    "momentum": 0.9,
    "data_dir": "./data",
    "output_dir": "./outputs",
    "dropout_rates": [0.0, 0.1, 0.2, 0.3, 0.5],
}


# ─────────────────────────────────────────────
# REPRODUCIBILIDADE
# ─────────────────────────────────────────────
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# MODELO — ResNet-18 com Dropout configuravel
# ─────────────────────────────────────────────
class ResNet18Dropout(nn.Module):
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
def get_dataloaders(data_dir, batch_size):
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
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────
# TREINO E AVALIACAO
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in tqdm(loader, desc="  Treino", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.max(1)[1].eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in tqdm(loader, desc="  Teste", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.max(1)[1].eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


# ─────────────────────────────────────────────
# SCAN DE DROPOUT
# ─────────────────────────────────────────────
def run_scan(dropout_rates, config):
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDispositivo: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, test_loader = get_dataloaders(config["data_dir"], config["batch_size"])
    os.makedirs(config["output_dir"], exist_ok=True)

    all_results = {}

    for rate in dropout_rates:
        print(f"\n{'='*60}")
        print(f"  DROPOUT = {rate}")
        print(f"{'='*60}")

        model = ResNet18Dropout(dropout_rate=rate).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                              momentum=config["momentum"], weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

        history = []
        best_acc, best_state = 0.0, None

        for epoch in range(1, config["epochs"] + 1):
            t_loss, t_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            v_loss, v_acc = evaluate(model, test_loader, criterion, device)
            scheduler.step()

            history.append({
                "epoch": epoch,
                "train_loss": round(t_loss, 4),
                "train_acc": round(t_acc, 2),
                "test_loss": round(v_loss, 4),
                "test_acc": round(v_acc, 2),
            })
            print(f"  Ep {epoch:02d} | train_acc: {t_acc:.2f}% | test_acc: {v_acc:.2f}%")

            if v_acc > best_acc:
                best_acc = v_acc
                best_state = copy.deepcopy(model.state_dict())

        # Salva modelo
        model_name = f"dropout_{str(rate).replace('.', '')}"
        state = best_state or model.state_dict()
        torch.save(state, os.path.join(config["output_dir"], f"{model_name}_best.pt"))

        all_results[str(rate)] = {
            "dropout_rate": rate,
            "best_test_acc": round(best_acc, 2),
            "history": history,
        }

    # JSON agregado
    results = {
        "method": "dropout_scan",
        "config": config,
        "device": str(device),
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = os.path.join(config["output_dir"], "dropout_scan_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResultados salvos em: {out_path}")

    # Tabela resumo
    print(f"\n{'Dropout':<12} | {'Best Acc':>10}")
    print("-" * 28)
    for rate_str, r in all_results.items():
        print(f"{float(rate_str):<12.1f} | {r['best_test_acc']:>9.2f}%")

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Treino com dropout variado")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--rates", type=str, default="0.0,0.1,0.2,0.3,0.5",
                        help="Taxas de dropout separadas por virgula")
    args = parser.parse_args()

    config = {
        **DEFAULT_CONFIG,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "dropout_rates": [float(x) for x in args.rates.split(",")],
    }
    print(f"Config: {json.dumps(config, indent=2)}")
    run_scan(config["dropout_rates"], config)


if __name__ == "__main__":
    main()
