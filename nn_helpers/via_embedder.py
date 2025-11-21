import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

print("Torch imports completed")

from sklearn.model_selection import train_test_split

import pandas as pd
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle as pkl
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

sys.path.append("/work/magroup/kaileyhu/synthetic_lethality/")

from utils.extract_df_info import get_patient_name_from_index
from utils.extract_df_info import get_genes_from_index

from prediction.sCilantro.nn_helpers.dataloader import dataset, dataset_unlabeled
from prediction.sCilantro.nn_helpers.net_layers import ViaNetVarLength

sys.path.append("/work/magroup/kaileyhu/synthetic_lethality/utils")


class via_embedder:
    """
    The viability embedder (via_embedder) class is a neural network that takes in knockout embeddings
    of a gene from a cell (done by geneformer) and trains it to predict viability scores. This is
    preprocessing so no test set is necessary. This is a regression problem

    We output the second to last layer of the neural network to get an embedding
    informed by the viability score (extract_nn_second_last)
    """

    def __init__(self, df, input_length = 768):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Found device {device}")
        print(f"Input length {input_length}")

        self.device = device
        self.input_length = input_length

        # df.set_index("Unnamed: 0", inplace = True)

        self.df = df

        self.batch_size = None
        self.lr = None
        self.num_epochs = None

        self.net = None

        self.criterion = nn.MSELoss().to(device)
        self.dl = None
        self.dl_test = None

        self.loss_arr = None

        ensembl_path = "/work/magroup/kaileyhu/Geneformer/geneformer/ensembl_mapping_dict_gc95M.pkl"

        with open(ensembl_path, "rb") as f:
            id_gene_dict = pkl.load(f)

        self.id_gene_dict = id_gene_dict

    def init_regression(self, batch_size, lr, num_epochs):
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

    def train_nn(self, path=None):
        net = ViaNetVarLength(self.input_length)
        net.to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=self.lr)
        num_samples = len(self.dl)

        print(f"Starting to train, saving result at {path}")

        loss_arr = []
        for epoch in tqdm(range(self.num_epochs)):
            running_loss = 0.0
            for d in self.dl:
                inputs, labels = d
                inputs = inputs.float().to(self.device)
                labels = labels.float().to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs, _ = net(inputs)

                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch: {epoch} / {self.num_epochs}, Loss: {running_loss / num_samples}")
            loss_arr.append(running_loss / num_samples)

        print("Finished Training")
        self.loss_arr = loss_arr
        self.net = net

        if path is not None:
            print(f"Saving model at {path}")
            torch.save(net.state_dict(), path)
        else:
            print(f"No path supplied, model not saved")

    def test_nn(self):
        """
        Tests nn associated with the model, returns if no nn exists
        @pre test data contains viability scores
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

                labels = labels.long().to(self.device)
                inputs = inputs.float().to(self.device)

                outputs, second_last = self.net(inputs)
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
                inputs = inputs.float().to(self.device)

                outputs, second_last = self.net(inputs)
                # predicted = outputs.data

                patient_to_pred[name] = outputs
                patient_to_layer[name] = second_last.to("cpu")

        if path is not None:
            res = pd.DataFrame.from_dict(patient_to_layer).T
            res.index = list(map(lambda x : x.replace('"', "'"), list(res.index)))
            res.to_hdf(path, key="table")

        return patient_to_pred, patient_to_layer

    def setup_dataloaders(self, test_size, seed=None):

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

    def train_and_test(self, test_size, path):
        """
        Sets up dataloaders, then trains & tests the model
        """
        self.setup_dataloaders(test_size)
        self.train_nn(path)

        if test_size != 0.0:
            self.test_nn()

    def load_net(self, path):
        """
        Load already trained viability net from disk
        """
        net = ViaNetVarLength(self.input_length)
        net.load_state_dict(torch.load(path, weights_only=True))
        net.to(self.device)
        self.net = net

    def plot_loss_arr(self):
        plt.plot(self.loss_arr)
        plt.title(
            f"Learning rate = {self.lr}, num epochs = {self.num_epochs}, batch size = {self.batch_size}"
        )
        plt.show()
        

    def create_SL_embs(self, pair_list, layers_path, out_path, emb_length=32, hadamard=False, custom_func = None):
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

        df2 = self.df.drop(columns="viability score")
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
            
            if hadamard:
                return [a * b for a, b in zip(emb1[:emb_length], emb2[:emb_length])]
            
            return list(emb1)[:emb_length] + list(emb2)[:emb_length]

        d["second_to_last"] = d.parallel_apply(lambda row: lookup_emb(row), axis=1)

        d.rename(columns={'patients': 'patient'}, inplace = True)
        c = ["patient", "gene 1", "gene 2", "gene 1 ensembl", "gene 2 ensembl"]

        d_expanded = pd.DataFrame(d["second_to_last"].tolist(), index=d.index)

        if hadamard:
            d_expanded.columns = [i for i in range(emb_length)]
        else:
            d_expanded.columns = [i for i in range(2 * emb_length)]
        final_df = pd.concat([d[c], d_expanded, d["SL"]], axis=1)

        final_df.index = final_df.parallel_apply(lambda row : f"({row['patient']}, {row['gene 1 ensembl']}, {row['gene 2 ensembl']})", axis = 1)

        final_df.to_hdf(out_path, key="table")
        print(f"Saved resulting dataframe to {out_path}.")

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
