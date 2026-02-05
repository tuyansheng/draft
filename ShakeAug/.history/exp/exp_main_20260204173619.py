import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from data_loader.dataloader_cifar import CIFARDataset
from data_loader.dataloader_minist import MNISTDataset
from data_loader.dataloader_imagenet import 
from exp.exp_basic import Experiment_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_metrics, get_results_path, save_metrics_to_csv, plot_loss_curve, plot_roc_curve
import torchvision.transforms as transforms
warnings.filterwarnings('ignore')


class Experiment_Main(Experiment_Basic):
    def __init__(self, args):
        super(Experiment_Main, self).__init__(args)

    def _build_model(self):
        train_data, _ = self._get_data(flag="TRAIN")
        test_data, _ = self._get_data(flag="TEST")

        # model init
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        if flag.upper() == "TRAIN":
            transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(degrees=5),
            transforms.RandomRotation(degrees=10),     
            transforms.RandomRotation(degrees=15),    
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(self.args.img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.614, 0.532, 0.288], std=[0.233, 0.226, 0.230])])

            dataset = CIFARDataset(self.args.train_data_path, self.args.train_physical_variable_label_path, transform = transform, return_phy_vars = self.args.return_phy_vars)

            
        elif flag.upper() == "TEST":
            transform = transforms.Compose([
            transforms.Resize(self.args.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

            dataset = CIFARDataset(self.args.test_data_path, self.args.test_physical_variable_label_path, transform = transform, return_phy_vars = self.args.return_phy_vars)
        else:
            raise ValueError(f"Unknown flag: {flag}")
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.args.batch_size, shuffle = (flag.upper()=="TRAIN"), num_workers = self.args.num_workers)

        return dataset, dataloader
    
    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr = self.args.learning_rate)
    
    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def _evaluate(self, data_loader, criterion, desc="Eval"):
        total_loss = []
        all_preds = []
        all_trues = []

        self.model.eval()
        with torch.no_grad():
            for imgs, label, external_vars in data_loader:
                imgs = imgs.float().to(self.device)
                external_vars = external_vars.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(imgs, external_vars)  
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
        return total_loss, metrics, trues, probs.numpy()


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join("./checkpoints", setting)
        os.makedirs(path, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_path = get_results_path(setting, f"training_log_{timestamp}.csv")
        results_prob_path = get_results_path(setting, f"probs_log_{timestamp}.csv")

        pd.DataFrame(columns = [
            "epoch", "learning_rate", "train_loss", "train_acc", "train_precision", "train_recall", "train_f1",
            "test_loss", "test_acc", "test_precision", "test_recall", "test_f1", "test_class_report"
        ]).to_csv(results_path, index = False)

        pd.DataFrame(columns = [
            "epoch", "learning_rate", "train_trues", "train_probs", "test_trues", "test_probs"
        ]).to_csv(results_prob_path, index = False)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []

            train_start_time = time.time()

            for imgs, label, external_vars in train_loader:
                imgs = imgs.float().to(self.device)
                external_vars = external_vars.float().to(self.device)
                label = label.to(self.device)

                model_optim.zero_grad()
                outputs = self.model(imgs, external_vars)
                loss = criterion(outputs, label.long().squeeze(-1))
                train_losses.append(loss.item())

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            # evaluate
            train_loss_avg, metrics_train, train_trues, train_probs = self._evaluate(train_loader, criterion, desc="Train")
            test_start_time = time.time()
            test_loss_avg, metrics_test, test_trues, test_probs= self._evaluate(test_loader, criterion, desc="Test")

            print(f"Epoch {epoch+1}/{self.args.train_epochs} "
            f"| Train Loss: {train_loss_avg:.4f} | Train Acc: {metrics_train['accuracy']:.4f} "
            f"| Train Pre: {metrics_train['precision']:.4f} | Train Recall: {metrics_train['recall']:.4f} | Train F1: {metrics_train['f1']:.4f} "
            f"| Test Loss: {test_loss_avg:.4f} | Test Acc: {metrics_test['accuracy']:.4f} "
            f"| Test Pre: {metrics_test['precision']:.4f} | Test Recall: {metrics_test['recall']:.4f} | Test F1: {metrics_test['f1']:.4f} "
            f"| Train Time: {time.time()-train_start_time:.2f}s"
            f"| Test Time: {time.time()-test_start_time:.2f}s")

            print(metrics_test['report'])
            print("\n" + "="*60 + "\n")

            # save into csv
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

            if self.args.target_acc is not None and metrics_test['accuracy'] >= self.args.target_acc:
                save_path = os.path.join(path, f"{metrics_test['accuracy']}_{timestamp}.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Achieve Target Accuracy {metrics_test['accuracy']}, Saved Model: {save_path}")

            # adjust learning rate
            if (epoch + 1) % 5 == 0:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        # save best model
        torch.save(self.model.state_dict(), os.path.join(path, f'checkpoint_last_{timestamp}.pth'))

        # plot 
        plot_loss_curve(results_path, save_path =  get_results_path(setting, f"loss_curve_{timestamp}.png"))
        plot_roc_curve(train_trues, train_probs, self.args.num_class, save_path = get_results_path(setting, f"ROC_Train_{timestamp}.png"))
        plot_roc_curve(test_trues, test_probs, self.args.num_class, save_path = get_results_path(setting, f"ROC_Test_{timestamp}.png"))

        return self.model


