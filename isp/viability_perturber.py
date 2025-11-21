# class to generate knockout and subtracted embeddings using the pretrained cancer model

import sys
import os

sys.path.append('/work/magroup/kaileyhu/Geneformer')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd 
import numpy as np
import anndata as ad
from tqdm import tqdm
import pickle
import torch
import math

from datasets import load_from_disk, concatenate_datasets

from geneformer import InSilicoPerturber
from geneformer import InSilicoPerturberStats
from geneformer import EmbExtractor
from geneformer import perturber_utils as pu

from transformers.utils import logging
logging.set_verbosity_error() 

TOKEN_DICTIONARY_FILE = "/work/magroup/kaileyhu/Geneformer/geneformer/token_dictionary_gc95M.pkl"
ENSEMBL_MAPPING_FILE = "/work/magroup/kaileyhu/Geneformer/geneformer/ensembl_mapping_dict_gc95M.pkl"

class via_perturber:
    def __init__(self, dataset, model, n_classes, model_type, embs_prefix, emb_mode, hvg_viable_file, 
                 batch_size = 50, 
                 start_at = 0, 
                 regenerate = False, 
                 temp_files = "/work/magroup/kaileyhu/outputs/single_knockout/torch_batches",
                 out_dir = "/work/magroup/kaileyhu/outputs/single_knockout/perturb_all/"):
        
        torch.cuda.empty_cache()
        self.temp_files = temp_files
        print(f"Initializing viability perturber... clearing all files in {self.temp_files}")
        
        if os.path.exists(temp_files):
            os.system(f"rm -rf {self.temp_files}/*")
        else:
            print(f"\n\nMaking temp file directory at {self.temp_files}\n\n")
            os.makedirs(self.temp_files)
            
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Found device {device}")
        
        self.ds_name = dataset
        self.ds = load_from_disk(dataset)
        self.n_classes = n_classes
        self.model = model
        self.model_type = model_type
        self.emb_mode = emb_mode
        self.batch_size = batch_size

        self.embs_prefix = embs_prefix
        self.start_at = start_at
        self.regenerate = regenerate

        self.valid_tokens = []
        for patient in self.ds['input_ids']:
            self.valid_tokens += patient

        if not os.path.exists(embs_prefix):
            print(f"Creating output directories for subtracted & perturbed embs at {embs_prefix}\n")
            os.makedirs(embs_prefix)
            os.makedirs(f"{embs_prefix}subtracted_embs/")
            os.makedirs(f"{embs_prefix}perturbed_embs/")

        self.out_dir = out_dir
        self.prefix = "out"

        self.res_dir = "/work/magroup/kaileyhu/res/"
        self.res_prefix = "original_embs"
        self.perturb_prefix = "perturb_embs"

        self.batch_name = f"{self.temp_files}/original_batch.dataset"
        self.batch_perturb_name = f"{self.temp_files}/perturbation_full_batch.dataset"
        
        with open(TOKEN_DICTIONARY_FILE, "rb") as td:
            self.gene_token_dict = pickle.load(td)

        with open (ENSEMBL_MAPPING_FILE, "rb") as td:
            self.ensembl_map = pickle.load(td)

        new_value_mapping = {}

        self.total_cells = 1479

        omics_expr = pd.read_csv("/work/magroup/kaileyhu/datasets/depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv")
        viability_mat = pd.read_csv("/work/magroup/kaileyhu/datasets/depmap/CRISPRGeneEffectProcessed.csv")
        
        omics_expr.set_index("Unnamed: 0", inplace = True)
        viability_mat.set_index("Unnamed: 0", inplace = True)
        
        for col in viability_mat.columns:
            split = col.split(" ")[0]
            if split in self.ensembl_map:
                new_value_mapping[col] = self.ensembl_map[split]

        self.viability_mat = viability_mat.rename(columns=new_value_mapping)

        with open(hvg_viable_file, 'rb') as hvg_file: # NEW for updated tokenization scheme
            self.hvg_viable = pickle.load(hvg_file)

    def test_if_empty(self, gene_tokens):
        try:
            pu.filter_data_by_tokens_and_log(
                self.ds, gene_tokens, 16, "genes_to_perturb"
            )
            return False
        except:
            print("Error: no genes to perturb - dataset is empty after filtering")
            return True

    def make_isp(self, to_perturb):
        isp = InSilicoPerturber(perturb_type="delete",
                    perturb_rank_shift=None,
                    genes_to_perturb=to_perturb,
                    combos=0,
                    anchor_gene=None,
                    model_type=self.model_type,
                    num_classes=self.n_classes,
                    emb_mode=self.emb_mode,
                    cell_emb_style="mean_pool",
                    emb_layer=-1,
                    forward_batch_size=self.batch_size,
                    nproc=16,
                    torch_batch_dir=self.temp_files)

        return isp

    def make_emb_extractor(self):
        embex = EmbExtractor(model_type=self.model_type,
                     num_classes=self.n_classes,
                     emb_mode=self.emb_mode,
                     emb_layer=-1,
                     emb_label=["patient_id"],
                     forward_batch_size=self.batch_size,
                     max_ncells=self.total_cells,
                     nproc=16)

        return embex

    def concat_files(self):
        files = os.listdir(self.temp_files)
        if (files[0] == 'original_batch.dataset'):
            files = files[1:]

        full_batch = load_from_disk(f"{self.temp_files}/{files[0]}")
        for i in range(1, len(files)):
            if (files[i] != 'original_batch.dataset'):
                batch = load_from_disk(f"{self.temp_files}/{files[i]}")
                full_batch = concatenate_datasets([full_batch, batch])
            
        full_batch.save_to_disk(self.batch_perturb_name)

    def perturb_gene(self, hvg):
        os.system(f"rm -rf {self.temp_files}/*")
        
        if hvg not in self.viability_mat.columns or hvg not in self.gene_token_dict:
            print("Error: hvg not in viability mat, continuing")
            return

        to_perturb = [hvg]

        embs = None
        embs2 = None
        sub = None
        isp = self.make_isp(to_perturb)

        res_original_df, perturb_full = isp.perturb_data(self.model, self.ds_name, self.out_dir, self.prefix, data = self.ds)

        embex = self.make_emb_extractor()

        embs = embex.extract_embs(self.model, self.batch_name, self.res_dir, self.res_prefix, data = res_original_df)

        embs2 = embex.extract_embs(self.model, self.batch_perturb_name, self.res_dir, self.perturb_prefix, data = perturb_full)

        embs.set_index("patient_id", inplace=True)
        embs2.set_index("patient_id", inplace=True)

        sub = embs - embs2
        sub.insert(0, 'viability', self.viability_mat.loc[:, hvg].to_frame())

        sub.to_csv(f"{self.embs_prefix}subtracted_embs/" + str(hvg) + ".csv")
        embs2.to_csv(f"{self.embs_prefix}perturbed_embs/" + str(hvg) + ".csv")
        print(f"Finished assembling {hvg}")

    def perturb_gene_set(self, genes):
        os.system(f"rm -rf {self.temp_files}/*")

        filename = ""

        for hvg in genes:
            if hvg not in self.viability_mat.columns or hvg not in self.gene_token_dict:
                print("Error: hvg not in viability mat, continuing")
                return
            filename += str(hvg)
            filename += " "

        to_perturb = genes

        gene_tokens = []
        for hvg in genes:
            gene_tokens.append(self.gene_token_dict[hvg])

        if (self.test_if_empty(gene_tokens)):
            return

        embs = None
        embs2 = None
        sub = None
        isp = self.make_isp(to_perturb)

        res_original_df, perturb_full = isp.perturb_data(self.model, self.ds_name, self.out_dir, self.prefix, data = self.ds)

        embex = self.make_emb_extractor()

        embs = embex.extract_embs(self.model, self.batch_name, self.res_dir, self.res_prefix, data = res_original_df)

        embs2 = embex.extract_embs(self.model, self.batch_perturb_name, self.res_dir, self.perturb_prefix, data = perturb_full)

        embs.set_index("patient_id", inplace=True)
        embs2.set_index("patient_id", inplace=True)

        sub = embs - embs2
        sub.insert(0, 'viability', self.viability_mat.loc[:, hvg].to_frame())

        sub.to_csv(f"{self.embs_prefix}subtracted_embs/{filename}.csv")
        embs2.to_csv(f"{self.embs_prefix}perturbed_embs/{filename}.csv")

    def perturb_all_genes(self):
        num_genes = len(self.hvg_viable)
        for i in tqdm(range(self.start_at, num_genes)):
            hvg = self.hvg_viable[i]

            if (not self.regenerate):
                if (os.path.isfile(f"{self.embs_prefix}subtracted_embs/" + str(hvg) + ".csv")):
                    print(f"Gene {hvg} has already been perturbed, continuing...")
                    continue
            
            if (hvg not in self.gene_token_dict or self.gene_token_dict[hvg] not in self.valid_tokens):
                print(f"Gene {hvg} not valid, continuing...")
                continue

            files = os.listdir(self.temp_files)
            print(f"\n\nRunning code on hvg {hvg}, currently on iteration {i}/{num_genes}")
            print(f"Current files in torch batches: {files}")
            self.perturb_gene(hvg)
        print(f"Finished perturbing all genes (for single knockout)!")

    def perturb_double_knockout(self):
        all_hvg = self.hvg_viable
        num_genes = len(all_hvg)
        
        for i in tqdm(range(self.start_at, num_genes)):
            gene_1 = all_hvg[i]
            for j in tqdm(range (i+1, num_genes)):
                gene_2 = all_hvg[j]
                
                print(f"\n\nComputing double knockout on on {gene_1} and {gene_2}, currently on iteration {i}/{len(all_hvg)}, {j}/{len(all_hvg)}")
                
                files = os.listdir(self.temp_files)
                print(f"Current files in torch batches: {files}\n\n")
                self.perturb_gene_set([gene_1, gene_2])

        print(f"Finished perturbing all genes (for double knockout)!")

    def perturb_pair_list(self):
        all_hvg = self.hvg_viable
        num_genes = len(all_hvg)
        
        for i in tqdm(range(self.start_at, num_genes)):
            gene_pair = all_hvg[i]
            gene_1 = gene_pair[0]
            gene_2 = gene_pair[1]

            filename = f"{gene_1} {gene_2} "


            if (os.path.isfile(f"{self.embs_prefix}subtracted_embs/{filename}.csv")):
                print(f"Gene {gene_1} and {gene_2} has already been perturbed, continuing...")
                continue

            print(f"Computing double knockout on on {gene_1} and {gene_2}, currently on iteration {i}/{len(all_hvg)}")
            
            files = os.listdir(self.temp_files)
            self.perturb_gene_set([gene_1, gene_2])

        print(f"Finished perturbing all genes (for double knockout from a pair list)!")



