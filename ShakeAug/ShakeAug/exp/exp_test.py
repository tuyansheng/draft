import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from data_loader.data_loader_cavitation import data_loader
from exp.exp_basic import Experiment_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_metrics, get_results_path, save_metrics_to_csv, plot_loss_curve, plot_roc_curve

warnings.filterwarnings('ignore')


class Experiment_Test(Experiment_Basic):
    def __init__(self, args):
        super(Experiment_Test, self).__init__(args)

    def _build_model(self):
        train_data, _ = self._get_data(flag="TRAIN")
        test_data, _ = self._get_data(flag="TEST")
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)

        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        return data_loader(self.args, flag)
    
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
            for batch_x, label, padding_mask in data_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)  
                loss = criterion(outputs, label.long().squeeze(-1))
                total_loss.append(loss.item())

                all_preds.append(outputs.cpu())
                all_trues.append(label.cpu())

        total_loss = np.mean(total_loss)
        preds = torch.cat(all_preds, dim=0)
        trues = torch.cat(all_trues, dim=0)

        probs = torch.nn.functional.softmax(preds, dim=1)
        predictions = torch.argmax(probs, dim=1).numpy()
        trues = trues.flatten().numpy()

        metrics = cal_metrics(predictions, trues)

        self.model.train()
        return total_loss, metrics

    def test(self, setting, load_best=True):
        test_data, test_loader = self._get_data(flag='TEST')

        if load_best:
            print("Loading best model...")
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints', setting, 'checkpoint.pth')))

        criterion = self._select_criterion()
        test_loss, metrics_test = self._evaluate(test_loader, criterion, desc="Test Final")


        # save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        test_csv_path = get_results_path(setting, f"{self.args.model}_final_test_metrics_{timestamp}.csv")

        save_metrics_to_csv(test_csv_path, {
        "setting": setting,
        "test_loss": test_loss,
        "test_acc": metrics_test['accuracy'],
        "test_precision": metrics_test['precision'],
        "test_recall": metrics_test['recall'],
        "test_f1": metrics_test['f1']
        })

        return metrics_test['accuracy']