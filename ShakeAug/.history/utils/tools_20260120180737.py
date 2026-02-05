import collections
import os
import sys
import numpy as np
import pandas as pd
import torch
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss



def adjust_learning_rate(optimizer, epoch, args):
    lr = None

    if args.lradj == 'type1':
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 1))

    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        lr = lr_adjust.get(epoch, None)  # 只有在表里才更新

    elif args.lradj == 'cosine':
        lr = args.learning_rate * 0.5 * (1 + math.cos(math.pi * epoch / args.train_epochs))

    elif args.lradj == 'exp':
        decay_rate = getattr(args, 'decay_rate', 0.96)
        decay_steps = getattr(args, 'decay_steps', 1)
        lr = args.learning_rate * (decay_rate ** (epoch // decay_steps))

    else:
        # 未知策略：不更新
        return

    # type2 可能返回 None（不在指定 epoch），这时不更新
    if lr is None:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(f'Updating learning rate to {lr}')



def cal_metrics(predictions, trues):

    # main metrics
    acc = accuracy_score(trues, predictions)
    precision = precision_score(trues, predictions, average = "weighted", zero_division = 0)
    recall = recall_score(trues, predictions, average = "weighted", zero_division = 0)
    f1 = f1_score(trues, predictions, average = "weighted", zero_division = 0)

    # classification report
    report = classification_report(trues, predictions, digits = 4, zero_division = 0)
    metrics_dict = {
        "accuracy":  acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report
    }
    return metrics_dict


def get_results_path(setting, filename):
    """
    args:
        setting (str): current expertment name
        filename (str): fime name
    """
    floder_path = os.path.join("./results", setting)
    os.makedirs(floder_path, exist_ok = True)
    return os.path.join(floder_path, filename)


def save_metrics_to_csv(csv_path, new_row):
    """
    args:
        csv_path (str): file path
        new_row (dict): save data
    """
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_existing = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
        df_existing.to_csv(csv_path, index=False)
    else:
        pd.DataFrame([new_row]).to_csv(csv_path, index=False)


def plot_loss_curve(csv_path, save_path = None):
    """
    args:
        csv_path (str): training log csv path
        save_path (str): save path
    """
    if not os.path.exists(csv_path):
        print(f"[Plot Loss] CSV file not found: {csv_path}")
        return
    
    # read csv 
    df = pd.read_csv(csv_path)

    if "train_loss" not in df.columns or "test_loss" not in df.columns:
        print("[Plot Loss] CSV does not include train_loss/test_loss, can not plot curve")
        return
    
    plt.figure(figsize = (8 ,6))

    # plot training loss
    plt.plot(df['epoch'], df["train_loss"], label = "Train Loss", marker = 'o', color = 'blue')
    plt.plot(df['epoch'], df["test_loss"], label = "Test Loss", marker = 'o', color = 'red')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()

    print(f"[Plot Loss] Loss curve saved to {save_path}")


def plot_roc_curve(y_true, y_score, num_class, save_path):
    """
    args:
        y_true (n_samples): ground truth
        y_score (n_samples, num_class): prob of each class
        nun_classes: the number of classes
        save_path: save path
    """
    plt.figure(figsize = (8, 6))

    if num_class == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color = 'blue', lw = 2, label = f"ROC curve (AUC = {roc_auc: .2f})")
    else:
        y_true_bin = label_binarize(y_true, classes = range(num_class))
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_class):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw = 2, label = f"Class {i} (AUC = {roc_auc[i]:.2f})")

        # Micro-avarage
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"], lw=2, color='red', label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.savefig(save_path)
    plt.close()
    print(f"[Plot ROC] ROC curve saved to {save_path}")