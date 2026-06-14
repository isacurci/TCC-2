"""
run_experiments_v2.py  —  TCC-2  

Correções aplicadas:

  [C1] EfficientNetB0 adaptado para CIFAR-10 (32×32 px):
       features[0][0].stride alterado de (2,2) para (1,1), mantendo
       espaço espacial suficiente ao longo de toda a rede.

  [C2] Validação CN separada do conjunto de teste (elimina data snooping):
       10% do conjunto de TREINAMENTO do CIFAR-10 (5.000 imagens, índices
       fixados pela seed) é reservado como validação. O early stopping usa
       exclusivamente esse subconjunto. O teste CN usa o conjunto de teste
       original completo de 10.000 imagens.

  [C3] Grid de L2 corrigido para {0,0 ; 1e-4 ; 5e-4} (consistente com texto).

  [C4] Script único e definitivo: run_experiments.py original (SGD, sem early
       stopping) foi descontinuado. Este arquivo é a fonte de verdade.

Cenários de treinamento:
  CN   treinado com CIFAR-10 limpo
  CR   treinado com CIFAR-10-C corrompido

Para cada cenário, avalia em:
  - CIFAR-10 limpo   → coluna "cifar_normal"
  - CIFAR-10-C       → coluna "cifar_ruido"

Grid de regularização: 3 Dropout × 3 L2 = 9 combinações por arquitetura
Arquiteturas: ResNet-18 e EfficientNet-B0 (CBAM leve)
Otimizador: Adam, lr=1e-3, cosine annealing
Early stopping: paciência de 5 épocas sobre val loss
Máximo de épocas: 30
"""

import argparse
import json
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import resnet18, efficientnet_b0
from tqdm import tqdm


# ------------------------------------------------------------------------------
# 1.  EARLY STOPPING
# ------------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = None
        self.best_state = None
        self.stop       = False

    def step(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)


# ------------------------------------------------------------------------------
# 2.  MECANISMO DE ATENÇÃO DE CANAL (CBAM simplificado)
#     Inserido no EfficientNet-B0 entre features e classificador
# ------------------------------------------------------------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg(x))
        max_out = self.fc(self.max(x))
        scale   = self.sig(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class EfficientNetB0WithAttention(nn.Module):
    """
    EfficientNet-B0 + atenção de canal inserida antes do classificador.

    [CORREÇÃO C1] O primeiro bloco convolucional (features[0][0]) tem seu
    stride alterado de (2, 2) para (1, 1). Sem essa modificação, imagens
    32×32 do CIFAR-10 são reduzidas a mapas 1×1 antes de features[6],
    provocando falha de BatchNorm (apenas 1 valor por canal) e tornando a
    rede incapaz de aprender representações espaciais úteis para a tarefa.
    Com stride=1 no primeiro conv, o mapa 32×32 é mantido até features[2],
    atingindo 2×2 ao final de features[8], que o AdaptiveAvgPool2d(1)
    então colapsa corretamente para 1×1.
    """
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super().__init__()
        base = efficientnet_b0(weights=None)

        # [CORREÇÃO C1] Reduz stride do primeiro conv de 2 para 1
        # para manter resolução espacial suficiente com entrada 32×32.
        base.features[0][0].stride = (1, 1)

        self.features   = base.features          # saída: (B, 1280, 2, 2)
        self.attention  = ChannelAttention(1280)
        self.avgpool    = nn.AdaptiveAvgPool2d(1) # (B, 1280, 1, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ------------------------------------------------------------------------------
# 3.  RESNET-18 COM DROPOUT
#     Adaptação para CIFAR-10: conv1 3×3 stride=1, sem maxpool
# ------------------------------------------------------------------------------
class ResNet18WithDropout(nn.Module):
    """
    ResNet-18 adaptada para CIFAR-10 (32×32 px).
    O primeiro bloco (conv1 7×7 stride=2 + maxpool stride=2 da ResNet-18
    original) é substituído por conv1 3×3 stride=1 + Identity, evitando
    redução excessiva da resolução espacial.
    """
    def __init__(self, num_classes=10, dropout_rate=0.0):
        super().__init__()
        base = resnet18(weights=None)
        # Substituição do conv1 7×7 stride=2 por conv1 3×3 stride=1
        base.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                  padding=1, bias=False)
        # Remoção do maxpool (stride=2) para manter resolução
        base.maxpool = nn.Identity()
        # Substitui a FC original por dropout + nova FC
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.dropout  = nn.Dropout(p=dropout_rate)
        self.fc       = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


# ------------------------------------------------------------------------------
# 4.  DATASETS
# ------------------------------------------------------------------------------

def ensure_cifar10c(cifar10c_dir: str) -> str:
    """
    Garante que os arquivos .npy do CIFAR-10-C existam em cifar10c_dir.
    Se não existirem, baixa via kagglehub. Retorna o caminho real dos .npy.
    """
    labels_path = os.path.join(cifar10c_dir, "labels.npy")
    if os.path.exists(labels_path):
        print(f"  CIFAR-10-C encontrado em: {cifar10c_dir}")
        return cifar10c_dir

    print("  CIFAR-10-C não encontrado — baixando via kagglehub …")
    try:
        import kagglehub
    except ImportError:
        raise ImportError(
            "kagglehub não está instalado.\n"
            "Instale com:  pip install kagglehub\n"
            "Ou baixe manualmente e aponte --cifar10c_dir para a pasta com os .npy"
        )

    downloaded_path = kagglehub.dataset_download("harshadakhatu/cifar-10-c")
    print(f"  Download concluído. Arquivos em: {downloaded_path}")

    for root, dirs, files in os.walk(downloaded_path):
        if "labels.npy" in files:
            print(f"  labels.npy encontrado em: {root}")
            return root

    raise FileNotFoundError(
        f"labels.npy não encontrado dentro de {downloaded_path}.\n"
        "Verifique o dataset baixado ou aponte --cifar10c_dir manualmente."
    )


# As 15 corrupções utilizadas neste trabalho (das 19 originais do CIFAR-10-C).
# Excluídas: speckle_noise, gaussian_blur, spatter e saturate — ver Seção 3.2.
CIFAR10C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def get_cifar10_normal_dataset(data_dir="./data", train=True):
    """Retorna o dataset CIFAR-10 com augmentation (apenas treino)."""
    if train:
        t = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    return torchvision.datasets.CIFAR10(root=data_dir, train=train,
                                        download=True, transform=t)


def get_cifar10_loader(data_dir="./data", train=True, batch_size=128,
                       num_workers=0, indices=None):
    """Cria DataLoader para CIFAR-10, com subsetting opcional por índice."""
    ds = get_cifar10_normal_dataset(data_dir=data_dir, train=train)
    if indices is not None:
        ds = Subset(ds, list(indices))
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, pin_memory=True)


class CIFAR10CDataset(Dataset):
    """
    Dataset lazy para CIFAR-10-C.

    Mantém os .npy em disco e normaliza apenas uma imagem por vez, evitando
    criar um tensor float32 gigante com todas as corrupções na memória.
    """
    def __init__(self, cifar10c_dir, corruptions, severities=(1, 2, 3, 4, 5),
                 image_indices=None):
        self.cifar10c_dir   = cifar10c_dir
        self.severities     = tuple(severities)
        self.image_indices  = list(range(10000)) if image_indices is None \
                              else list(image_indices)
        self.corruption_paths = []

        for corruption in corruptions:
            path = os.path.join(cifar10c_dir, f"{corruption}.npy")
            if os.path.exists(path):
                self.corruption_paths.append(path)
            else:
                print(f"  [warn] {corruption}.npy não encontrado — pulando")

        if not self.corruption_paths:
            raise FileNotFoundError(
                f"Nenhum arquivo .npy de corrupção encontrado em {cifar10c_dir}"
            )

        self.label_path      = os.path.join(cifar10c_dir, "labels.npy")
        self.labels          = None
        self.data_arrays     = None
        self.per_corruption  = len(self.severities) * len(self.image_indices)
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)

    def _ensure_open(self):
        if self.labels is None:
            self.labels = np.load(self.label_path, mmap_mode="r")
        if self.data_arrays is None:
            self.data_arrays = [
                np.load(path, mmap_mode="r") for path in self.corruption_paths
            ]

    def __len__(self):
        return len(self.corruption_paths) * self.per_corruption

    def __getitem__(self, idx):
        self._ensure_open()
        corruption_idx      = idx // self.per_corruption
        pos_in_corruption   = idx % self.per_corruption
        severity_idx        = pos_in_corruption // len(self.image_indices)
        base_image_idx      = self.image_indices[pos_in_corruption % len(self.image_indices)]
        severity            = self.severities[severity_idx]
        image_idx           = (severity - 1) * 10000 + base_image_idx

        img   = np.array(self.data_arrays[corruption_idx][image_idx], copy=True)
        img   = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        img   = (img - self.mean) / self.std
        label = int(self.labels[image_idx])
        return img, label


def load_cifar10c_dataset(cifar10c_dir, severities=(1, 2, 3, 4, 5),
                          image_indices=None):
    """Retorna um CIFAR10CDataset lazy com as 15 corrupções utilizadas."""
    label_path = os.path.join(cifar10c_dir, "labels.npy")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"labels.npy não encontrado em {cifar10c_dir}")
    return CIFAR10CDataset(cifar10c_dir, CIFAR10C_CORRUPTIONS, severities,
                           image_indices=image_indices)


# ------------------------------------------------------------------------------
# [CORREÇÃO C2] Split do CIFAR-10-C por imagem-base (sem data leakage)
# Mantido sem alteração — a lógica já estava correta no v2 original.
# ------------------------------------------------------------------------------
def split_base_image_indices(seed=42, n_images=10000,
                             train_frac=0.8, val_frac=0.1):
    """
    Divide os 10.000 índices-base do CIFAR-10-C em treino/validação/teste.

    Garante ausência de vazamento: todas as corrupções e severidades de uma
    mesma imagem-base ficam exclusivamente em um único subconjunto.
    """
    generator = torch.Generator().manual_seed(seed)
    perm      = torch.randperm(n_images, generator=generator).tolist()
    n_train   = int(train_frac * n_images)
    n_val     = int(val_frac * n_images)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]


# ------------------------------------------------------------------------------
# [CORREÇÃO C2] Split de validação CN independente do conjunto de teste
# O conjunto de validação CN é extraído do CONJUNTO DE TREINAMENTO (train=True),
# nunca do conjunto de teste. Isso elimina o data snooping identificado no v2.
# ------------------------------------------------------------------------------
def split_cn_train_val(seed=42, n_train=50000, val_frac=0.1):
    """
    Divide os 50.000 exemplos do TREINAMENTO do CIFAR-10 em treino e validação.

    Retorna (train_indices, val_indices) — ambos disjuntos e separados do
    conjunto de TESTE (10.000 imagens), que permanece exclusivo para avaliação.
    """
    generator  = torch.Generator().manual_seed(seed)
    perm       = torch.randperm(n_train, generator=generator).tolist()
    n_val      = int(val_frac * n_train)
    val_idx    = perm[:n_val]      # 5.000 imagens de validação
    train_idx  = perm[n_val:]      # 45.000 imagens de treino efetivo
    return train_idx, val_idx


# ------------------------------------------------------------------------------
# 5.  TREINO / AVALIAÇÃO
# ------------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


def build_metrics(model, test_loader_normal, test_loader_noise, device):
    """Retorna dict com acurácia/precisão/recall/f1 macro nos dois cenários."""
    from sklearn.metrics import precision_score, recall_score, f1_score

    def get_preds(loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs  = imgs.to(device)
                preds = model(imgs).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(lbls.numpy())
        return np.array(all_labels), np.array(all_preds)

    results = {}
    for name, loader in [("cifar_normal", test_loader_normal),
                         ("cifar_ruido",  test_loader_noise)]:
        y_true, y_pred = get_preds(loader)
        acc  = (y_true == y_pred).mean()
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
        results[name] = dict(acuracia=round(float(acc),  4),
                             precisao=round(float(prec), 4),
                             recall  =round(float(rec),  4),
                             f1      =round(float(f1),   4))
    return results


def train_model(model_cls, model_kwargs, train_loader, val_loader,
                epochs, l2, device, patience=5, tag=""):
    model     = model_cls(**model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=l2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    es        = EarlyStopping(patience=patience)

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc   = train_one_epoch(model, train_loader,
                                             optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        es.step(val_loss, model)

        if ep % 5 == 0 or es.stop:
            print(f"  [{tag}] ep {ep:3d} | "
                  f"train {tr_acc:.3f} | val {val_acc:.3f} | "
                  f"ES patience {es.counter}/{patience}")

        if es.stop:
            print(f"  [{tag}] Early stop na época {ep}")
            break

    es.restore_best(model)
    return model


# ------------------------------------------------------------------------------
# 6.  PIPELINE PRINCIPAL
# ------------------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir,
                            f"phase3_v2{('_' + args.run_suffix) if args.run_suffix else ''}_results.json")

    # [CORREÇÃO C3] Grid de L2: {0,0 ; 1e-4 ; 5e-4} — consistente com texto
    dropout_rates = [float(x) for x in args.dropouts.split(",")]
    l2_values     = [float(x) for x in args.l2_values.split(",")]

    print(f"\nDropout rates : {dropout_rates}")
    print(f"L2 values     : {l2_values}")
    print(f"Épocas máximas: {args.epochs}")
    print(f"Early stopping: paciência {args.patience}")

    # --------------------------------------------------------------------------
    # Carregar CIFAR-10
    # --------------------------------------------------------------------------
    print("\n[1/4] Preparando CIFAR-10 …")

    # [CORREÇÃO C2] Validação CN extraída do conjunto de TREINAMENTO
    cn_train_idx, cn_val_idx = split_cn_train_val(seed=args.split_seed)
    print(f"  Split CN — treino efetivo: {len(cn_train_idx)} | "
          f"validação independente: {len(cn_val_idx)}")

    loader_cn_train = get_cifar10_loader(args.data_dir, train=True,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         indices=cn_train_idx)
    loader_cn_val   = get_cifar10_loader(args.data_dir, train=True,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         indices=cn_val_idx)
    # Conjunto de TESTE CN: 10.000 imagens completas (train=False, sem indices)
    loader_cn_test  = get_cifar10_loader(args.data_dir, train=False,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers)

    # --------------------------------------------------------------------------
    # Carregar CIFAR-10-C
    # --------------------------------------------------------------------------
    print("[2/4] Verificando/baixando CIFAR-10-C …")
    cifar10c_dir = ensure_cifar10c(args.cifar10c_dir)

    cr_train_idx, cr_val_idx, cr_test_idx = split_base_image_indices(
        seed=args.split_seed
    )
    print(f"  Split CR — treino: {len(cr_train_idx)} | "
          f"val: {len(cr_val_idx)} | teste: {len(cr_test_idx)} imagens-base")

    ds_noise_tr   = load_cifar10c_dataset(cifar10c_dir,
                                           image_indices=cr_train_idx)
    ds_noise_val  = load_cifar10c_dataset(cifar10c_dir,
                                           image_indices=cr_val_idx)
    ds_noise_test = load_cifar10c_dataset(cifar10c_dir,
                                           image_indices=cr_test_idx)

    loader_cr_train = DataLoader(ds_noise_tr,   batch_size=args.batch_size,
                                 shuffle=True,  num_workers=args.num_workers)
    loader_cr_val   = DataLoader(ds_noise_val,  batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
    loader_cr_test  = DataLoader(ds_noise_test, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)

    # --------------------------------------------------------------------------
    # Configurações
    # --------------------------------------------------------------------------
    all_results = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                all_results = json.load(f)
            print(f"Resultados parciais carregados de: {out_path}")
        except json.JSONDecodeError:
            print(f"  [warn] Não foi possível ler {out_path}; iniciando vazio")
            all_results = {}

    MODEL_CONFIGS = {
        "ResNet18"      : (ResNet18WithDropout,       {}),
        "EfficientNetB0": (EfficientNetB0WithAttention, {}),  # C1 já no __init__
    }

    TRAIN_SCENARIOS = {
        "CN": (loader_cn_train, loader_cn_val,  "Treinado com CIFAR-10 limpo"),
        "CR": (loader_cr_train, loader_cr_val,  "Treinado com CIFAR-10-C"),
    }
    selected_scenarios = [
        x.strip().upper() for x in args.train_scenarios.split(",") if x.strip()
    ]
    invalid = sorted(set(selected_scenarios) - set(TRAIN_SCENARIOS))
    if invalid:
        raise ValueError(f"Cenários inválidos em --train_scenarios: {invalid}")
    TRAIN_SCENARIOS = {k: v for k, v in TRAIN_SCENARIOS.items()
                       if k in selected_scenarios}

    total_runs = (len(MODEL_CONFIGS) * len(TRAIN_SCENARIOS)
                  * len(dropout_rates) * len(l2_values))
    run_i = 0

    # --------------------------------------------------------------------------
    # Loop principal
    # --------------------------------------------------------------------------
    for model_name, (ModelCls, extra_kwargs) in MODEL_CONFIGS.items():
        for scenario, (tr_loader, vl_loader, _) in TRAIN_SCENARIOS.items():
            for do_rate in dropout_rates:
                for l2_val in l2_values:
                    run_i += 1
                    tag = f"{model_name}_{scenario}_D{do_rate}_L{l2_val}"
                    print(f"\n[{run_i}/{total_runs}] {tag}")

                    if args.dry_run:
                        print("  (dry-run — pulando treino)")
                        continue

                    model_kwargs = dict(num_classes=10,
                                       dropout_rate=do_rate,
                                       **extra_kwargs)
                    ckpt_path = os.path.join(
                        args.output_dir,
                        f"{tag}{('_' + args.run_suffix) if args.run_suffix else ''}_best.pt"
                    )

                    if tag in all_results:
                        print("  resultado já existe — pulando")
                        continue

                    if os.path.exists(ckpt_path):
                        print("  checkpoint encontrado — avaliando sem retreinar")
                        model = ModelCls(**model_kwargs).to(device)
                        state = torch.load(ckpt_path, map_location=device)
                        model.load_state_dict(state)
                    else:
                        model = train_model(
                            ModelCls, model_kwargs,
                            tr_loader, vl_loader,
                            epochs=args.epochs,
                            l2=l2_val,
                            device=device,
                            patience=args.patience,
                            tag=tag,
                        )
                        torch.save(model.state_dict(), ckpt_path)

                    # Avaliação nos dois cenários de teste
                    metrics = build_metrics(
                        model,
                        loader_cn_test,   # 10.000 imagens completas (C2)
                        loader_cr_test,
                        device,
                    )

                    all_results[tag] = {
                        "modelo"         : f"{model_name} + D{do_rate} + R{l2_val}",
                        "cenario_treino" : scenario,
                        "dropout"        : do_rate,
                        "l2"             : l2_val,
                        **metrics,
                    }
                    with open(out_path, "w") as f:
                        json.dump(all_results, f, indent=2)
                    print(f"  resultado parcial salvo em: {out_path}")

                    print(f"  Normal → acc={metrics['cifar_normal']['acuracia']:.4f}  "
                          f"f1={metrics['cifar_normal']['f1']:.4f}")
                    print(f"  Ruído  → acc={metrics['cifar_ruido']['acuracia']:.4f}  "
                          f"f1={metrics['cifar_ruido']['f1']:.4f}")

    # --------------------------------------------------------------------------
    # Salvar resultado final
    # --------------------------------------------------------------------------
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResultados salvos em: {out_path}")

    # --------------------------------------------------------------------------
    # Tabela resumo
    # --------------------------------------------------------------------------
    if not args.dry_run:
        print("\n" + "=" * 82)
        print(f"{'Modelo':<42} {'Cen':>3} {'CN-Acc':>7} {'CR-Acc':>7} "
              f"{'CN-F1':>7} {'CR-F1':>7}")
        print("-" * 82)
        for k, v in all_results.items():
            print(f"{v['modelo']:<42} {v['cenario_treino']:>3} "
                  f"{v['cifar_normal']['acuracia']:>7.4f} "
                  f"{v['cifar_ruido']['acuracia']:>7.4f} "
                  f"{v['cifar_normal']['f1']:>7.4f} "
                  f"{v['cifar_ruido']['f1']:>7.4f}")


# ------------------------------------------------------------------------------
# 7.  ARGPARSE
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TCC-2 — Experimentos reformulados (versão corrigida)"
    )
    parser.add_argument("--epochs",          type=int,  default=30)
    parser.add_argument("--patience",        type=int,  default=5,
                        help="Paciência do early stopping")
    parser.add_argument("--batch_size",      type=int,  default=128)
    parser.add_argument("--num_workers",     type=int,  default=0,
                        help="Workers do DataLoader (use 0 no Windows)")
    # [CORREÇÃO C3] L2 padrão consistente com texto: {0,0 ; 1e-4 ; 5e-4}
    parser.add_argument("--dropouts",        type=str,  default="0.0,0.2,0.5",
                        help="Taxas de dropout separadas por vírgula")
    parser.add_argument("--l2_values",       type=str,  default="0.0,1e-4,5e-4",
                        help="Valores de L2 separados por vírgula")
    parser.add_argument("--train_scenarios", type=str,  default="CN,CR",
                        help="Cenários de treino separados por vírgula: CN,CR")
    parser.add_argument("--data_dir",        type=str,  default="./data")
    parser.add_argument("--cifar10c_dir",    type=str,  default="./data/CIFAR-10-C",
                        help="Pasta com os arquivos .npy do CIFAR-10-C")
    parser.add_argument("--output_dir",      type=str,  default="./outputs")
    parser.add_argument("--run_suffix",      type=str,  default="",
                        help="Sufixo para resultados/checkpoints desta rodada")
    parser.add_argument("--split_seed",      type=int,  default=42,
                        help="Seed do split dos dados")
    parser.add_argument("--dry_run",         action="store_true",
                        help="Só mostra configuração sem treinar")
    args = parser.parse_args()
    run(args)