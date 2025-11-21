import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

print("Torch imports completed")

import pickle as pkl
import sys
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append("/work/magroup/kaileyhu/synthetic_lethality/")

from prediction.sCilantro.nn_helpers.dataloader import dataset, dataset_unlabeled
from prediction.sCilantro.nn_helpers.net_layers import FiLMNet, FiLMNetTwoRepr

from utils.extract_df_info import *

class via_film:
    """
    The viability embedder (via_film) class is a neural network that takes in knockout embeddings
    of a gene from a cell (done by geneformer) and conditions on the protein embedding of the
    knocked out gene.

    We output the second to last layer of the neural network to get an embedding
    informed by the viability score (extract_nn_second_last)
    """

    def __init__(self, df, input_length = 768):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Found device {device}")

        self.device = device

        self.df = df

        self.batch_size = None
        self.lr = None
        self.num_epochs = None

        self.net = None

        self.criterion = nn.MSELoss().to(device)
        self.dl = None
        self.dl_test = None

        self.loss_arr = None
        self.input_length = input_length
        self.two_genes = False

        ensembl_path = "/work/magroup/kaileyhu/Geneformer/geneformer/ensembl_mapping_dict_gc95M.pkl"

        with open(ensembl_path, "rb") as f:
            id_gene_dict = pkl.load(f)

        self.id_gene_dict = id_gene_dict

    def init_regression(self, batch_size, lr, num_epochs):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

    def train_nn(self, path=None):
        if not self.two_genes:
            net = FiLMNet(protein_size = self.input_length - 512)
        else:
            print(f"Using 2 genes")
            net = FiLMNetTwoRepr()
        print(f"Protein size is {self.input_length - 512}")

        net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        num_samples = len(self.dl)

        print(f"Starting to train, saving result at {path}")

        ESM_param_arr = []
        gene2vec_param_arr = []
        loss_arr = []
        for epoch in tqdm(range(self.num_epochs)):
            running_loss = 0.0
            for d in self.dl:
                inputs, labels = d

                # 512 cell emb, 256 gene emb
                gene_embs = inputs[:, 512:]
                inputs = inputs[:, :512]
                inputs = inputs.float().to(self.device)
                gene_embs = gene_embs.float().to(self.device)
                labels = labels.float().to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, _ = net(inputs, gene_embs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch: {epoch} / {self.num_epochs}, Loss: {running_loss / num_samples}")
            loss_arr.append(running_loss / num_samples)

            if self.two_genes:
                with torch.no_grad():
                    xb = inputs.to(net.l1.weight.device).float()
                    # replicate the pre-gate context exactly as in forward:
                    x_ctx = F.relu(net.l1(xb))
                    if hasattr(net, "norm_x"):      # if you added pre-FiLM LayerNorm
                        x_ctx = net.norm_x(x_ctx)
                
                    gE = torch.sigmoid(net.linear_ESM2_gate(x_ctx))      # [B,256] or [B,1]
                    gG = torch.sigmoid(net.linear_gene2vec_gate(x_ctx))  # [B,256] or [B,1]
                
                    # collapse feature dim, then batch:
                    g_esm_mean = gE.mean(dim=-1).mean().item()
                    g_g2v_mean = gG.mean(dim=-1).mean().item()

                    ESM_param_arr.append(g_esm_mean)
                    gene2vec_param_arr.append(g_g2v_mean)

        print("Finished Training")
        self.loss_arr = loss_arr
        self.net = net

        if path is not None:
            print(f"Saving model at {path}")
            torch.save(net.state_dict(), path)
        else:
            print(f"No path supplied, model not saved")

        if self.two_genes:
            with open ("/work/magroup/kaileyhu/res/TEMP_METRICS_ARR.pkl", "wb") as f:
                pkl.dump([loss_arr, ESM_param_arr, gene2vec_param_arr], f)

    def test_nn(self):
        """
        Tests nn associated with the model, returns if no nn exists
        @pre test data contains viability scores

        return: predicted values
        """

        if self.net is None:
            print(f"Error: no net exists")
            return

        correct = 0
        total = 0

        vals = []
        y_true = []

        mse = 0

        with torch.no_grad():
            for d in tqdm(self.dl_test):
                inputs, labels = d
                gene_embs = inputs[:, 512:]
                inputs = inputs[:, :512]

                inputs = inputs.float().to(self.device)
                gene_embs = gene_embs.float().to(self.device)
                labels = labels.float().to(self.device)

                outputs, _ = self.net(inputs, gene_embs)
                predicted = outputs.data

                total += 1

                correct += abs(predicted.item() - labels.item()) <= 0.1
                mse += (predicted.item() - labels.item()) ** 2
                vals.append(predicted.item())
                y_true.append(labels.item())

        print(f"Done testing, MSE: {mse / total}")
        return vals, y_true

    def extract_nn_second_last(self, test, path=None):
        """
        @pre test data has no viability scores, and that the index is the relevant names
        (ex. (patient, gene))
        """
        if self.net is None:
            print(f"Error: no net exists")
            return

        dataloader = dataset_unlabeled(test)

        patient_to_pred = {}
        patient_to_layer = {}

        with torch.no_grad():
            for d in tqdm(dataloader):
                inputs, name = d
                # 512 cell emb, 256 gene emb
                gene_embs = inputs[512:]
                inputs = inputs[:512]

                inputs = inputs.float().to(self.device)
                gene_embs = gene_embs.float().to(self.device)

                outputs, second_last = self.net(inputs, gene_embs)
                # predicted = outputs.data

                patient_to_pred[name] = outputs
                patient_to_layer[name] = second_last.to("cpu")

        if path is not None:
            res = pd.DataFrame.from_dict(patient_to_layer).T
            res.to_hdf(path, key="table")

        return patient_to_pred, patient_to_layer

    def setup_dataloaders(self, test_size, seed=None):
        print(f"Setting up dataloaders with input length {self.input_length}")
        if test_size == 0.0:
            train_data = dataset(self.df, self.input_length)
            self.dl = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
            return

        if seed is not None:
            train, test = train_test_split(self.df, test_size=test_size, random_state=seed)
        else:
            train, test = train_test_split(self.df, test_size=test_size)
        train_data = dataset(train, self.input_length)
        test_data = dataset(test, self.input_length)
        self.dl = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        self.dl_test = DataLoader(test_data, batch_size=1, shuffle=False)

    def setup_cv(self, folds, get_sets = False):
        """
        Sets up cross validation dataloaders
        """

        all_test = []
        all_train = []

        df2 = self.df.sample(frac=1)
        df_splits = np.array_split(df2, folds)

        test_sets = []
        train_sets = []

        for i in range(folds):
            test = df_splits[i]
            train = pd.concat(df_splits[:i] + df_splits[i + 1 :])

            test_sets.append(test)
            train_sets.append(train)

            train_data = dataset(train, self.input_length)
            test_data = dataset(test, self.input_length)

            print(f"Test dataset shuffle is false")
            all_train.append(DataLoader(train_data, batch_size=self.batch_size, shuffle=True))
            all_test.append(DataLoader(test_data, batch_size=1, shuffle=False))

        if get_sets:
            return all_test, all_train, test_sets, train_sets
        return all_test, all_train

    def train_and_test(self, test_size, path):
        """
        Sets up dataloaders, then trains & tests the model
        """

        self.setup_dataloaders(test_size)
        self.train_nn(path)

        if test_size != 0.0:
            return self.test_nn()

    def load_net(self, path):
        """
        Load already trained viability net from disk
        """

        net = FiLMNet(protein_size = self.input_length - 512)
        
        net.load_state_dict(torch.load(path, weights_only=True))
        net.to(self.device)
        self.net = net

    def plot_loss_arr(self):
        plt.plot(self.loss_arr)
        plt.title(
            f"Learning rate = {self.lr}, num epochs = {self.num_epochs}, batch size = {self.batch_size}"
        )
        plt.show()

    def create_SL_embs(self, pair_list, layers_path, out_path, emb_length=32, custom_func = None):
        df_list = []
        for pair in tqdm(pair_list):
            df_list.append([pair[0], pair[1], pair_list[pair]])

        def get_id(gene):
            if gene in self.id_gene_dict:
                return f"gene_{self.id_gene_dict[gene]}"
            return None

        d = pd.DataFrame(df_list, columns=["gene 1", "gene 2", "SL"])

        d["gene 1 ensembl"] = d["gene 1"].parallel_apply(get_id)
        d["gene 2 ensembl"] = d["gene 2"].parallel_apply(get_id)

        d = d.dropna()
        print(
            f"Missing {len(df_list) - len(d)} / {len(df_list)} pairs due to missing genes in the ensembl mapping dictionary."
        )

        if "viability score" in self.df.columns:
            df2 = self.df.drop(columns="viability score")
        else:
            df2 = self.df
        df2["name"] = df2.index
        df2["gene"] = df2["name"].parallel_apply(get_genes_from_index)
        df2["patient"] = df2["name"].parallel_apply(get_patient_name_from_index)

        g_to_p_mapping = df2.groupby("gene")["patient"].apply(set).to_dict()

        d["patients"] = d.parallel_apply(
            lambda row: g_to_p_mapping.get(
                row['gene 1 ensembl'], set()
            ).intersection(g_to_p_mapping.get(row['gene 2 ensembl'], set())),
            axis=1,
        )
        d = d.explode("patients").dropna(subset=["patients"])
        print(f"After exploding, {len(d)} patient/gene pairs remain.")

        layer_df = pd.read_hdf(layers_path, "table")

        def lookup_emb(row):
            name1 = (row["patients"], row['gene 1 ensembl'])
            name2 = (row["patients"], row['gene 2 ensembl'])
            emb1 = layer_df.loc[str(name1)]
            emb2 = layer_df.loc[str(name2)]
            
            if custom_func is not None:
                return [custom_func(a, b) for a, b in zip(emb1[:emb_length], emb2[:emb_length])]
            
            return list(emb1)[:emb_length] + list(emb2)[:emb_length]

        d["second_to_last"] = d.parallel_apply(lambda row: lookup_emb(row), axis=1)

        d.rename(columns={'patients': 'patient'}, inplace = True)
        c = ["patient", "gene 1", "gene 2", "gene 1 ensembl", "gene 2 ensembl"]

        d_expanded = pd.DataFrame(d["second_to_last"].tolist(), index=d.index)

        if custom_func is not None:
            d_expanded.columns = [i for i in range(emb_length)]
        else:
            d_expanded.columns = [i for i in range(2 * emb_length)]
        final_df = pd.concat([d[c], d_expanded, d["SL"]], axis=1)

        final_df.index = final_df.parallel_apply(lambda row : f"({row['patient']}, {row['gene 1 ensembl']}, {row['gene 2 ensembl']})", axis = 1)

        final_df.to_hdf(out_path, key="table")
        print(f"Saved resulting dataframe to {out_path}.")
