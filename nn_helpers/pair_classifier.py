from enum import Enum
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics import roc_auc_score, average_precision_score

import numpy as np
import pandas as pd

import random

sys.path.append("/work/magroup/kaileyhu/synthetic_lethality/")

from prediction.sCilantro.nn_helpers.dataloader import dataset_classifier
from prediction.sCilantro.nn_helpers.net_layers import SLNet, ESMNet, GeneformerNet, SLNet32
from prediction.sCilantro.nn_helpers.split import test_train_split, test_train_split_CV1, split_validation

input_length = 64


class ModelType(Enum):
    COMBINED = 1
    ESM_ONLY = 2
    GENEFORMER_ONLY = 3
    DIM32 = 4


class pair_classifier:
    def __init__(self, df, model_type=ModelType.COMBINED):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Found device {device}")

        self.device = device

        self.df = df
        self.df["SL"] = self.df["SL"].astype(int)

        self.net = None
        self.loss_arr = None
        self.batch_size = None
        self.lr = None
        self.num_epochs = None

        self.dl_test = None
        self.dl_train = None
        self.test_names = None
        self.calib_names = None
        self.test_SL = None
        self.model_type = model_type
        self.validation = False

        if model_type == ModelType.ESM_ONLY:
            self.input_length = 512
        elif model_type == ModelType.GENEFORMER_ONLY:
            self.input_length = 1024
        elif model_type == ModelType.DIM32:
            self.input_length = 32
        else:
            self.input_length = 64

    def init_classification(self, batch_size, lr, num_epochs):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

    def train_nn(self, path):
        if self.model_type == ModelType.ESM_ONLY:
            net = ESMNet()
        elif self.model_type == ModelType.GENEFORMER_ONLY:
            net = GeneformerNet()
        elif self.model_type == ModelType.DIM32:
            print(f"This is the dim32 model")
            net = SLNet32()
        else:
            print(f"Using normal net")
            net = SLNet()
        net.to(self.device)

        if self.validation:
            if path is None:
                raise ValueError("Validation requires a path to save the best model")
            
            print("Training with validation set for early stopping")

        # Calculate class weights
        labels = torch.tensor(self.train_SL.tolist())

        print(labels)
        labels.to(self.device)
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()

        # normalize
        class_weights = class_weights / class_weights.sum()
        print(f"Class weights: {class_weights}")

        test_output = torch.randn(128, 2)
        test_labels = torch.randint(0, 2, (128,)).long()
        test_loss = nn.CrossEntropyLoss()(test_output, test_labels)
        print(f"Test loss with random data: {test_loss}")
        
        class_weights = class_weights.to(self.device)

        # Pass weights to CrossEntropyLoss
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(self.device) 

        optimizer = optim.Adam(net.parameters(), lr=self.lr)

        num_samples = len(self.dl_train)

        loss_arr = []

        # --- early stopping
        best_val_loss = float('inf')
        patience = 10
        epochs_no_improve = 0

        val_loss_arr = []
        # ---

        for epoch in tqdm(range(self.num_epochs)):  # loop over the dataset multiple times
            running_loss = 0.0
            net.train()
            for d in self.dl_train:
                inputs, labels = d

                inputs = inputs.float().to(self.device)
                labels = labels.squeeze().long().to(self.device)

                optimizer.zero_grad()
                outputs = net(inputs).squeeze()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch: {epoch} / {self.num_epochs}, Loss: {running_loss / num_samples}")
            loss_arr.append(running_loss / num_samples)

            if self.validation:
                net.eval()
                val_running_loss = 0.0
                num_val_batches = len(self.dl_val)
                with torch.no_grad():
                    for d in self.dl_val:
                        inputs, labels = d
                        inputs = inputs.float().to(self.device)
                        labels = labels.squeeze().long().to(self.device)

                        outputs = net(inputs).squeeze()
                        loss = criterion(outputs, labels)
                        val_running_loss += loss.item()
            
                avg_val_loss = val_running_loss / num_val_batches
                val_loss_arr.append(avg_val_loss)
                print(f"Epoch: {epoch} / {self.num_epochs}, Validation Loss: {avg_val_loss}")

                # early stop?
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    torch.save(net.state_dict(), path)
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs (Patience of {patience} reached).")
                    net.load_state_dict(torch.load(path))
                    break


        print("Finished Training")

        self.loss_arr = loss_arr
        self.net = net

        if path is not None:
            print(f"Saving model at {path}")
            torch.save(net.state_dict(), path)
        else:
            print("No path supplied, model not saved")

    def test_nn_with_set(self, test):
        if self.net is None:
            print("Error: no net exists")
            return

        test_data = dataset_classifier(test, input_length)
        self.dl_test = DataLoader(test_data, batch_size=1, shuffle=False)

        correct = 0
        total = 0

        vals = []
        corr = []
        conf_scores = []

        with torch.no_grad():
            for d in tqdm(self.dl_test):
                inputs, labels = d
                labels = labels.long().to(self.device)
                inputs = inputs.float().to(self.device)

                # calculate outputs by running test set vectors through the network
                outputs = self.net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, _ = torch.max(probs, 1)

                total += 1
                correct += predicted.item() == labels.item()
                vals.append(predicted.item())
                corr.append(labels.item())
                conf_scores.append(conf.item())

        print(f"Accuracy: {correct / total}")
        return vals, corr, conf_scores
    
    def _qhat(self, scores, alpha):
        """
        Exact finite-sample conformal quantile:
        order statistic at k = ceil((n+1)*(1-alpha)), clamped to [1, n].
        """
        s = np.sort(np.asarray(scores, dtype=float))
        n = s.size
        if n == 0:
            return 1.0  # maximally conservative if empty
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        k = min(max(k, 1), n)
        return s[k - 1]

    def calibrate(self, calibration_dl, mondrian_class_dict={}, extract_name=lambda x: x):
        if self.net is None:
            print("Error: no net exists")
            return

        if mondrian_class_dict is not None:
            print(f"Calibrating with {len(Counter(mondrian_class_dict.values()))} mondrian classes")

        calibration_scores = {}

        with torch.no_grad():
            for i, d in tqdm(enumerate(calibration_dl)):
                inputs, labels = d
                labels = labels.long().to(self.device)
                inputs = inputs.float().to(self.device)
                name = extract_name(self.calib_names.iloc[i])
                mondrian_class = mondrian_class_dict.get(name, -1)

                outputs = self.net(inputs)

                probs = torch.flatten(torch.nn.functional.softmax(outputs, dim=1))
                true_class_prob = probs[labels.item()].item()
                conformal_score = 1 - true_class_prob
                if mondrian_class not in calibration_scores:
                    calibration_scores[mondrian_class] = []
                calibration_scores[mondrian_class].append(conformal_score)

        # compute quantile
        # alpha_levels = np.arange(0.0, 1.0, 0.001)
        # print(f"Using alpha levels {alpha_levels}")

        # alpha_to_quantile = {}

        # for alpha in alpha_levels:
        #     alpha_to_quantile[alpha] = {}
        #     for mondrian_class, scores in calibration_scores.items():
        #         alpha_to_quantile[alpha][mondrian_class] = self._qhat(scores, alpha)

        # print("Finished calibration")
        # return alpha_to_quantile
        return calibration_scores

    def conformal_pred(
        self, calibration_dl, test_dl, mondrian_class_dict={}, extract_name=lambda x: x
    ):
        """
        calibration_dl: dataloader for calibration set for determining quantile based on conformal scores
        """
        if self.net is None:
            print("Error: no net exists")
            return

        calibration_scores = self.calibrate(
            calibration_dl, mondrian_class_dict=mondrian_class_dict, extract_name=extract_name
        )
        # alpha_levels = sorted(alpha_to_quantile.keys(), reverse=False)

        # print(f"Alpha to quantile is {alpha_to_quantile}")
        # print(f"Alpha levels are {alpha_levels}")

        vals = []
        corr = []
        conf_scores = []
        probs_list = []

        with torch.no_grad():
            for i, d in tqdm(enumerate(test_dl)):
                inputs, labels = d
                labels = labels.long().to(self.device)
                inputs = inputs.float().to(self.device)
                name = extract_name(self.test_names.iloc[i])
                mondrian_class = mondrian_class_dict.get(name, -1)

                outputs = self.net(inputs)

                probs = torch.flatten(torch.nn.functional.softmax(outputs, dim=1))


                _, predicted = torch.max(outputs.data, 1)

                best_pred_prob = probs[predicted.item()].item()
                worst_pred_prob = probs[1 - predicted.item()].item()
                
                vals.append(predicted.item())
                corr.append(labels.item())
                conf_scores.append((mondrian_class, 1 - best_pred_prob, 1 - worst_pred_prob))
                probs_list.append(probs)


        conf_df = pd.DataFrame(conf_scores, columns=["mondrian_class", "best_conf_score", "worst_conf_score"])
        conf_df['calib_scores'] = conf_df['mondrian_class'].parallel_apply(lambda x : calibration_scores.get(x, []))

        class_to_alpha_dict = {}

        for c in conf_df['mondrian_class'].unique():
            mondrian_calib_scores = calibration_scores.get(c, [])
            n_calib = len(mondrian_calib_scores)

            calib_scores_np = np.sort(np.array(mondrian_calib_scores))
            conf_df_c = conf_df[conf_df['mondrian_class'] == c]

            unique_conf_scores = conf_df_c['best_conf_score'].unique()
            unique_conf_scores = np.append(unique_conf_scores, conf_df_c['worst_conf_score'].unique())
            unique_conf_scores = list(set(unique_conf_scores))
            conf_scores_np = np.array(unique_conf_scores)
            print(f"Conf scores np: {conf_scores_np}")

            rank_less_than_conf = np.searchsorted(calib_scores_np, conf_scores_np, side='left')
            count_greater_equal = n_calib - rank_less_than_conf

            p_values_np = (count_greater_equal + 1) / (n_calib + 1)
            conf_scores_to_pval = dict(zip(unique_conf_scores, p_values_np))
            class_to_alpha_dict[c] = conf_scores_to_pval


        conf_df['p_value_higher'] = conf_df.apply(
            lambda x: class_to_alpha_dict.get(x['mondrian_class'], {}).get(x['best_conf_score'], -1.0), axis=1
        )
        conf_df['p_value_lower'] = conf_df.apply(
            lambda x: class_to_alpha_dict.get(x['mondrian_class'], {}).get(x['worst_conf_score'], -1.0), axis=1
        )

        conf_scores_best = conf_df['p_value_higher'].tolist()
        conf_scores_worst = conf_df['p_value_lower'].tolist()


        print("Finished conformal prediction")
        return vals, corr, list(zip(conf_scores_best, conf_scores_worst)), self.test_names, calibration_scores, probs_list

    def test_nn(self):
        if self.net is None:
            print("Error: no net exists")
            return

        correct = 0
        total = 0

        vals = []
        corr = []
        conf_scores = []

        with torch.no_grad():
            for d in tqdm(self.dl_test):
                inputs, labels = d
                labels = labels.long().to(self.device)
                inputs = inputs.float().to(self.device)

                # calculate outputs by running test set vectors through the network
                outputs = self.net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, _ = torch.max(probs, 1)

                total += 1
                correct += predicted.item() == labels.item()
                vals.append(predicted.item())
                corr.append(labels.item())
                conf_scores.append(conf.item())

        print(f"Accuracy: {correct / total}")
        return vals, corr, conf_scores

    def split_n(self, a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))

    # returns TEST, TRAIN
    def setup_cv(self, num_iters, cv=3, genes_only=False):
        if cv == 1:
            return test_train_split_CV1(
                list(self.df["gene 1"]), list(self.df["gene 2"]), self.df, num_iters, genes_only
            )
        return test_train_split(
            list(self.df["gene 1"]), list(self.df["gene 2"]), self.df, num_iters
        )

    def setup_dataloaders_from_sets(self, test, train, validation=False):
        print(f"Setting up dataloaders with input {self.input_length}, validation set? {validation}")
        self.validation = validation

        if validation:
            print("Creating validation set from training data")
            train, val = split_validation(train, val_frac=0.1)
            val_data = dataset_classifier(val, self.input_length)
            self.dl_val = DataLoader(val_data, batch_size=1, shuffle=False)

        train_data = dataset_classifier(train, self.input_length)
        test_data = dataset_classifier(test, self.input_length)
        self.dl_train = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.dl_test = DataLoader(test_data, batch_size=1, shuffle=False)
        self.test_names = test.iloc[:, :5]
        self.test_SL = test.iloc[:, -1]
        self.train_SL = train.iloc[:, -1]

    def get_calib_test(self, test, calib_frac, mondrian_class_dict=None, extract_name=None):
        calib_size = int(len(test) * calib_frac)
        if not mondrian_class_dict:
            calib = test.sample(n=calib_size,random_state=0)
        else:
            test_classes = pd.Series(mondrian_class_dict)
            test['name'] = test.apply(extract_name, axis = 1)
    
            test['class'] = test['name'].apply(lambda x : mondrian_class_dict[x])
    
            calib = (
                test
                .dropna(subset=['class']) # Only sample from rows with an assigned class
                .groupby('class')
                .sample(frac=calib_frac, random_state=0)
            )
    
            calib = calib.drop(columns=['class', 'name'])
            test = test.drop(columns=['class', 'name'])
        test = test.drop(calib.index)

        calib_data = dataset_classifier(calib, self.input_length)
        test_data = dataset_classifier(test, self.input_length)
        calib_dl = DataLoader(calib_data, batch_size=1, shuffle=False)
        test_dl = DataLoader(test_data, batch_size=1, shuffle=False)

        self.test_names = test.iloc[:, :5]
        self.calib_names = calib.iloc[:, :5]

        return calib_dl, test_dl

    def setup_dataloaders(self, test_size):
        genes = set(self.df["gene 1"]) | set(self.df["gene 2"])
        test_genes = set(random.sample(genes, int(len(genes) * test_size)))

        has_both = self.df[self.df["gene 1"].isin(test_genes) & self.df["gene 2"].isin(test_genes)]
        has_none = self.df[
            ~(self.df["gene 1"].isin(test_genes)) & ~(self.df["gene 2"].isin(test_genes))
        ]

        test = has_both
        train = has_none

        train_data = dataset_classifier(train, self.input_length)
        test_data = dataset_classifier(test, self.input_length)
        self.dl_train = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.dl_test = DataLoader(test_data, batch_size=1, shuffle=False)
        self.test_names = test.iloc[:, :5]

        # last index is SL or not
        self.test_SL = test.iloc[:, -1]
        self.train_SL = train.iloc[:, -1]

    def load_net(self, path):
        """
        Load already trained viability net from disk
        """
        if self.model_type == ModelType.ESM_ONLY:
            net = ESMNet()
        elif self.model_type == ModelType.DIM32:
            net = SLNet32()
        else:
            net = SLNet()
        net.load_state_dict(torch.load(path, weights_only=True))
        net.to(self.device)
        self.net = net

    def plot_loss_arr(self):
        plt.plot(self.loss_arr)
        plt.title(
            f"Learning rate = {self.lr}, num epochs = {self.num_epochs}, batch size = {self.batch_size}"
        )
        plt.show()

    def train_and_test(self, test_size, path):
        self.setup_dataloaders(test_size)
        self.train_nn(path)
        self.test_nn()

    def compute_acc(self, preds, labels):
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        total = 0
        correct = 0

        for v in range(len(preds)):
            if labels[v] == 0:
                if preds[v] == 0:
                    true_neg += 1
                else:
                    false_pos += 1

            elif labels[v] == 1:
                if preds[v] == 1:
                    true_pos += 1
                else:
                    false_neg += 1
            if labels[v] == preds[v]:
                correct += 1

            total += 1

        precision = (true_pos) / (true_pos + false_pos)
        recall = (true_pos) / (true_pos + false_neg)

        f1_score = (true_pos) / (true_pos + 0.5 * (false_pos + false_neg))

        if true_pos == 0 or true_neg == 0:
            auc = 0
        else:
            auc = roc_auc_score(labels, preds)

        aupr = average_precision_score(labels, preds)

        print(
            f"Precision: {precision}, Recall: {recall} \nF1 score: {f1_score}, AUC = {auc}, Accuracy = {correct / total}"
        )
        print(
            f"accuracy,precision,recall,f1 score, auc, aupr\n{correct / total}\n{precision}\n{recall}\n{f1_score}\n{auc}\n{aupr}\n"
        )
        return (precision, recall, f1_score, auc, correct / total, aupr)
