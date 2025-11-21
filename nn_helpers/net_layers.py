import torch.nn as nn
import torch.nn.functional as F
import torch
from prediction.sCilantro.nn_helpers import film

### PAIR CLASSIFICATION NETS


class SLNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)
    
class SLNet32(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class ESMNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class GeneformerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


### VIABILITY EMBEDDER NETS


class FiLMNet(nn.Module):
    def __init__(self, protein_size = 256):
        super().__init__()

        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(512, 256)
        self.protein_size = protein_size

        # gene emb = 256, protein emb dim = 256
        self.film_layer = film.FiLMLayer(primary_in_features=256, cond_in_features=protein_size)

        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 1)

    def forward(self, x, protein_emb):
        x = F.relu(self.l1(x))
        x = self.film_layer(x, protein_emb)
        x = F.relu(self.l2(x))

        # TODO: check if we want to ReLU before this
        x_temp = self.l3(x)
        x = F.relu(x_temp)
        x = self.l4(x)
        return (x, x_temp)
    
class FiLMNet128(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(512, 256)

        # cell emb = 256, protein emb dim = 128
        self.film_layer = film.FiLMLayer(primary_in_features=256, cond_in_features=128)

        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 1)

    def forward(self, x, protein_emb):
        x = F.relu(self.l1(x))
        x = self.film_layer(x, protein_emb)
        x = F.relu(self.l2(x))

        # TODO: check if we want to ReLU before this
        x_temp = self.l3(x)
        x = F.relu(x_temp)
        x = self.l4(x)
        return (x, x_temp)
    
class FiLMNet512(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(512, 256)

        # cell emb = 256, protein emb dim = 512
        self.film_layer = film.FiLMLayer(primary_in_features=256, cond_in_features=512)

        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 1)

    def forward(self, x, protein_emb):
        x = F.relu(self.l1(x))
        x = self.film_layer(x, protein_emb)
        x = F.relu(self.l2(x))

        # TODO: check if we want to ReLU before this
        x_temp = self.l3(x)
        x = F.relu(x_temp)
        x = self.l4(x)
        return (x, x_temp)

# FiLM net conditioning on both ESM2 and gene2vec
class FiLMNetTwoRepr(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(512, 256)
        self.norm_x = nn.LayerNorm(256)

        # cell emb = 256, protein emb dim = 128
        self.film_layer_ESM2 = film.FiLMLayer(primary_in_features=256, cond_in_features=128)

        # cell emb = 256, protein emb dim = 128
        self.film_layer_gene2vec = film.FiLMLayer(primary_in_features=256, cond_in_features=128)

        self.layer_norm_ESM2 = nn.LayerNorm(128)
        self.layer_norm_gene2vec = nn.LayerNorm(128)

        self.linear_ESM2_gate = nn.Linear(256, 256)
        self.linear_gene2vec_gate = nn.Linear(256, 256)

        nn.init.constant_(self.linear_ESM2_gate.weight, 0.0001)
        nn.init.constant_(self.linear_ESM2_gate.bias, -4)

        nn.init.constant_(self.linear_gene2vec_gate.weight, 0.0001)
        nn.init.constant_(self.linear_gene2vec_gate.bias, -2)

        self.sigmoid_ESM2 = nn.Sigmoid()
        self.sigmoid_gene2vec = nn.Sigmoid()

        # self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 32)
        self.l6 = nn.Linear(32, 1)

    def forward(self, x, full_gene_emb):
        x = self.norm_x(F.relu(self.l1(x)))
        if len(x.shape) == 1:
            ESM2_emb = full_gene_emb[:128]
            gene2vec_emb = full_gene_emb[128:]
        else:
            ESM2_emb = full_gene_emb[:, :128]
            gene2vec_emb = full_gene_emb[:, 128:]

        ESM2_emb = self.layer_norm_ESM2(ESM2_emb)
        gene2vec_emb = self.layer_norm_gene2vec(gene2vec_emb)

        x_ESM2 = self.film_layer_ESM2(x, ESM2_emb)
        x_gene2vec = self.film_layer_gene2vec(x, gene2vec_emb)

        ESM2_gate = self.sigmoid_ESM2(self.linear_ESM2_gate(x))
        gene2vec_gate = self.sigmoid_gene2vec(self.linear_gene2vec_gate(x))

        x = x + ESM2_gate * x_ESM2 + gene2vec_gate * x_gene2vec

        # x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))

        x_temp = self.l5(x)
        x = F.relu(x_temp)
        x = self.l6(x)
        return (x, x_temp)
    
class ViaNetVarLength(nn.Module):
    def __init__(self, input_length):
        super().__init__()

        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(input_length, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 1)

        self.linear_relu_stack_768 = nn.Sequential(
            nn.Linear(input_length, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        # TODO: check if we want to ReLU before this
        x_temp = self.l3(x)
        x = F.relu(x_temp)
        x = self.l4(x)
        return (x, x_temp)

class ViabilityNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(768, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 1)

        self.linear_relu_stack_768 = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        # TODO: check if we want to ReLU before this
        x_temp = self.l3(x)
        x = F.relu(x_temp)
        x = self.l4(x)
        return (x, x_temp)


class GeneformerNetVia(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.l1 = nn.Linear(512, 256)
        self.l2 = nn.Linear(256, 64)
        self.l3 = nn.Linear(64, 32)
        self.l4 = nn.Linear(32, 1)

        self.linear_relu_stack_512 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        # TODO: check if we want to ReLU before this
        x_temp = self.l3(x)
        x = F.relu(x_temp)
        x = self.l4(x)
        return (x, x_temp)
