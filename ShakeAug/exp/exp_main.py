import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from data_loader.dataloader_cifar import CIFARDataset
from data_loader.dataloader_minist import MNISTDataset
from data_loader.dataloader_imagenet import ImageNetDataset
from exp.exp_basic import Experiment_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_metrics, get_results_path, save_metrics_to_csv, plot_loss_curve, plot_roc_curve

from shake_aug import ShakeRecipe, ShakeAugDataset, ShakeAugBatchCollator, apply_shake_recipe_batch

warnings.filterwarnings('ignore')


class RepeatAugDataset(Dataset):
    """
    Expand dataset length by repeating each sample R times.
    Each access still goes through base dataset + its (random) transform,
    so repeated copies are different augmented views.
    """
    def __init__(self, base: Dataset, repeat: int = 1):
        assert repeat >= 1
        self.base = base
        self.repeat = int(repeat)

    def __len__(self):
        return len(self.base) * self.repeat

    def __getitem__(self, idx):
        base_idx = idx % len(self.base)
        return self.base[base_idx]


class ShakeOnlyBatchCollator:
    """
    For FAIR mode shake (no dataset expansion, still N).
    We keep baseline transforms in dataset (ToTensor+Normalize),
    then apply shake on the batch here (pixel-transport in normalized space too).
    NOTE: shake_aug expects float in [0,1] ideally, but your pipeline normalizes.
    Best practice: apply shake BEFORE normalize.
    However user要求: “baseline transforms（含 Normalize/ToTensor），然后 DataLoader 层再套 Shake”
    so we follow that. (If you later want shake before normalize, I can adjust.)
    """
    def __init__(self, recipes, p_apply: float = 1.0, base_seed: int = 0, dynamic: bool = True):
        self.recipes = list(recipes)
        self.p_apply = float(p_apply)
        self.base_seed = int(base_seed)
        self.dynamic = bool(dynamic)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __call__(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)
        y = torch.as_tensor(ys)

        # one recipe (or randomly choose among K recipes)
        B = x.shape[0]
        device = x.device

        # seeds per-sample
        idx = torch.arange(B, device=device, dtype=torch.int64)
        seeds = (self.base_seed
                 + idx * 1_000_003
                 + (self.epoch * 10_000_019 if self.dynamic else 0)).to(torch.int64)

        # choose recipe per-sample (if K>1)
        if len(self.recipes) == 1:
            ridx = torch.zeros((B,), device=device, dtype=torch.int64)
        else:
            # deterministic choose recipe id by seed
            ridx = (seeds % len(self.recipes)).to(torch.int64)

        x_out = x
        for k in torch.unique(ridx):
            kk = int(k.item())
            mask = (ridx == kk)
            if mask.any():
                xb = x_out[mask]
                sb = seeds[mask]
                x_out[mask] = apply_shake_recipe_batch(xb, self.recipes[kk], seeds=sb)

        # p_apply gating: if p_apply<1, keep original for some samples
        if self.p_apply < 1.0:
            u = torch.rand((B,), device=device)
            keep = (u >= self.p_apply)
            if keep.any():
                x_out[keep] = x[keep]

        return x_out, y


class Experiment_Main(Experiment_Basic):
    def __init__(self, args):
        super(Experiment_Main, self).__init__(args)
        self._shake_collator = None  # for fair shake: set_epoch each epoch

    def _build_model(self):
        _ = self._get_data(flag="TRAIN")
        _ = self._get_data(flag="TEST")

        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # Transforms for MNIST/CIFAR/ImageNet
    # aug_mode: baseline / traditional / shake
    def _build_transforms(self, dataset_name: str, is_train: bool):
        dataset_name = dataset_name.lower()
        aug_mode = getattr(self.args, "aug_mode", "traditional").lower()

        # baseline-only transforms (no randomness)
        def baseline_mnist():
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        def baseline_cifar(mean, std):
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        def baseline_imagenet(img_size: int):
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        # MNIST
        if dataset_name in ["mnist", "minist"]:
            if (not is_train) or (aug_mode in ["baseline", "shake"]):
                return baseline_mnist()
            # traditional aug (train only)
            return transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        # CIFAR10/100
        if dataset_name in ["cifar10", "cifar100"]:
            if dataset_name == "cifar10":
                mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            else:
                mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

            if (not is_train) or (aug_mode in ["baseline", "shake"]):
                return baseline_cifar(mean, std)

            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        # ImageNet
        if dataset_name == "imagenet":
            img_size = int(self.args.img_size)
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

            if not is_train:
                return transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

            if aug_mode in ["baseline", "shake"]:
                return transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])

            return transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

        raise ValueError(f"Unknown dataset: {dataset_name}")

    def _make_shake_recipes(self):
        K = int(getattr(self.args, "shake_k", 1))
        N = int(getattr(self.args, "shake_N", 2))
        beta = float(getattr(self.args, "shake_beta", 0.03))
        alpha = float(getattr(self.args, "shake_alpha", 0.2))
        cell_h = int(getattr(self.args, "shake_cell_h", 8))
        cell_w = int(getattr(self.args, "shake_cell_w", 8))
        p_apply = float(getattr(self.args, "shake_p_apply", 1.0))
        offset = getattr(self.args, "shake_offset", "random_each_iter")
        if isinstance(offset, str) and offset.lower() == "none":
            offset = None

        recipes = [
            ShakeRecipe(N=N, beta=beta, alpha=alpha, S=(cell_h, cell_w), offset=offset, p_apply=p_apply)
            for _ in range(K)
        ]
        return recipes

    def _get_data(self, flag):
        is_train = (flag.upper() == "TRAIN")
        dataset_name = self.args.dataset.lower()
        aug_mode = getattr(self.args, "aug_mode", "traditional").lower()
        exp_mode = getattr(self.args, "exp_mode", "fair").lower()  # fair / expand

        transform = self._build_transforms(dataset_name, is_train=is_train)
        data_path = self.args.train_data_path if is_train else self.args.test_data_path

        # base dataset
        if dataset_name in ["mnist", "minist"]:
            dataset = MNISTDataset(data_path, train=is_train, transform=transform)
        elif dataset_name in ["cifar10", "cifar100"]:
            dataset = CIFARDataset(data_path, dataset=dataset_name, train=is_train, transform=transform)
        elif dataset_name == "imagenet":
            dataset = ImageNetDataset(data_path, transform=transform, return_path=False)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        collate_fn = None

        # =========================
        # A) FAIR MODE: all are N
        # baseline: baseline transforms
        # traditional: random transforms (no Repeat)
        # shake: baseline transforms + online shake in collate (no expansion)
        # =========================
        if is_train and exp_mode == "fair":
            if aug_mode == "traditional":
                # no repeat in fair mode
                pass
            elif aug_mode == "shake":
                recipes = self._make_shake_recipes()
                self._shake_collator = ShakeOnlyBatchCollator(
                    recipes,
                    p_apply=float(getattr(self.args, "shake_p_apply", 1.0)),
                    base_seed=int(getattr(self.args, "shake_base_seed", 0)),
                    dynamic=bool(getattr(self.args, "shake_dynamic", True)),
                )
                collate_fn = self._shake_collator

        # =========================
        # B) EXPAND MODE: allow expansion
        # traditional + RepeatAug(R)
        # shake => (K+1)x expansion
        # =========================
        if is_train and exp_mode == "expand":
            if aug_mode == "traditional":
                trad_repeat = int(getattr(self.args, "trad_repeat", 1))
                if trad_repeat > 1:
                    dataset = RepeatAugDataset(dataset, repeat=trad_repeat)

            elif aug_mode == "shake":
                recipes = self._make_shake_recipes()
                dataset = ShakeAugDataset(
                    base=dataset,
                    recipes=recipes,
                    include_original=True,  # (K+1)x
                    dynamic=bool(getattr(self.args, "shake_dynamic", True)),
                    base_seed=int(getattr(self.args, "shake_base_seed", 0)),
                    apply_in_getitem=False
                )
                collate_fn = ShakeAugBatchCollator(recipes)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=is_train,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=is_train,
            collate_fn=collate_fn
        )
        return dataset, dataloader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def _evaluate(self, data_loader, criterion, desc="Eval"):
        total_loss = []
        all_preds = []
        all_trues = []

        self.model.eval()
        with torch.no_grad():
            for imgs, label in data_loader:
                imgs = imgs.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(imgs)
                loss = criterion(outputs, label.long().squeeze(-1))
                total_loss.append(loss.item())

                all_preds.append(outputs.cpu())
                all_trues.append(label.cpu())

        total_loss = float(np.mean(total_loss)) if len(total_loss) else 0.0
        preds = torch.cat(all_preds, dim=0)
        trues = torch.cat(all_trues, dim=0)

        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).numpy()
        trues_np = trues.flatten().numpy()

        metrics = cal_metrics(predictions, trues_np)

        self.model.train()
        return total_loss, metrics, trues_np, probs.numpy()

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join("./checkpoints", setting)
        os.makedirs(path, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = get_results_path(setting, f"training_log_{timestamp}.csv")
        results_prob_path = get_results_path(setting, f"probs_log_{timestamp}.csv")

        pd.DataFrame(columns=[
            "epoch", "learning_rate", "train_loss", "train_acc", "train_precision", "train_recall", "train_f1",
            "test_loss", "test_acc", "test_precision", "test_recall", "test_f1", "test_class_report"
        ]).to_csv(results_path, index=False)

        pd.DataFrame(columns=[
            "epoch", "learning_rate", "train_trues", "train_probs", "test_trues", "test_probs"
        ]).to_csv(results_prob_path, index=False)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        early_stopping = EarlyStopping(
            patience=getattr(self.args, "patience", 7),
            verbose=True,
            delta=getattr(self.args, "earlystop_delta", 0.0)
        )

        for epoch in range(self.args.train_epochs):
            # for FAIR shake: update epoch so dynamic seed changes each epoch
            if self._shake_collator is not None:
                self._shake_collator.set_epoch(epoch)

            self.model.train()
            train_start_time = time.time()

            for imgs, label in train_loader:
                imgs = imgs.float().to(self.device)
                label = label.to(self.device)

                model_optim.zero_grad()
                outputs = self.model(imgs)
                loss = criterion(outputs, label.long().squeeze(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            train_loss_avg, metrics_train, train_trues, train_probs = self._evaluate(train_loader, criterion, desc="Train")
            test_start_time = time.time()
            test_loss_avg, metrics_test, test_trues, test_probs = self._evaluate(test_loader, criterion, desc="Test")

            early_stopping(metrics_test['accuracy'], self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stop training.")
                break

            print(
                f"Epoch {epoch+1}/{self.args.train_epochs} "
                f"| Train Loss: {train_loss_avg:.4f} | Train Acc: {metrics_train['accuracy']:.4f} "
                f"| Train Pre: {metrics_train['precision']:.4f} | Train Recall: {metrics_train['recall']:.4f} | Train F1: {metrics_train['f1']:.4f} "
                f"| Test Loss: {test_loss_avg:.4f} | Test Acc: {metrics_test['accuracy']:.4f} "
                f"| Test Pre: {metrics_test['precision']:.4f} | Test Recall: {metrics_test['recall']:.4f} | Test F1: {metrics_test['f1']:.4f} "
                f"| Train Time: {time.time()-train_start_time:.2f}s"
                f"| Test Time: {time.time()-test_start_time:.2f}s"
            )

            print(metrics_test['report'])
            print("\n" + "=" * 60 + "\n")

            save_metrics_to_csv(results_prob_path, {
                "epoch": epoch + 1,
                "learning_rate": model_optim.param_groups[0]['lr'],
                "train_trues": train_trues,
                "train_probs": train_probs,
                "test_trues": test_trues,
                "test_probs": test_probs
            })

            save_metrics_to_csv(results_path, {
                "epoch": epoch + 1,
                "learning_rate": model_optim.param_groups[0]['lr'],
                "train_loss": train_loss_avg,
                "train_acc": metrics_train['accuracy'],
                "train_precision": metrics_train['precision'],
                "train_recall": metrics_train['recall'],
                "train_f1": metrics_train['f1'],
                "test_loss": test_loss_avg,
                "test_acc": metrics_test['accuracy'],
                "test_precision": metrics_test['precision'],
                "test_recall": metrics_test['recall'],
                "test_f1": metrics_test['f1'],
                "test_class_report": metrics_test['report']
            })

            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        torch.save(self.model.state_dict(), os.path.join(path, f'checkpoint_last_{timestamp}.pth'))

        best_ckpt = os.path.join(path, "checkpoint_best.pth")
        if os.path.exists(best_ckpt):
            self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
            print(f"Loaded best model from: {best_ckpt}")

        plot_loss_curve(results_path, save_path=get_results_path(setting, f"loss_curve_{timestamp}.png"))
        plot_roc_curve(train_trues, train_probs, self.args.num_class, save_path=get_results_path(setting, f"ROC_Train_{timestamp}.png"))
        plot_roc_curve(test_trues, test_probs, self.args.num_class, save_path=get_results_path(setting, f"ROC_Test_{timestamp}.png"))

        return self.model
