"""
data_generator class to take the .csv outputs from the viability perturber and convert it into one dataframe to be used as input training data for the classifier.
    is_hvg: True if dataframe knockouts were done on only contains the top 500 HVG's, False otherwise
    ko_file_dir: path to directory containing knockout .csv files
    sub_file_dir: path to directory containing subtracted .csv files
    output_dir: path to save final dataframe
    model_path: path to Geneformer model used to produce embeddings
    model_type: "Pretrained" or "CellClassifier" depending on model type
    emb_mode: "cell" or "cls" depending on type of model (35M vs 95M)
    emb_dim: dimension of embedding (should always be 512)
    n_classes: number of classes model was trained to classify: 0 for Pretrained, 71 for ClCancer
    dataset: path to dataset object used for extracting embeddings

saves file in output_dir/gene_patient_emb_mat.csv
"""


import sys
import os

sys.path.append('/work/magroup/kaileyhu/Geneformer')

import torch
import pandas as pd 
import numpy as np
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import pickle as pkl
import ast

from geneformer import EmbExtractor

pd.options.mode.chained_assignment = None

class data_generator:
    def __init__(self, is_hvg, ko_file_dir, sub_file_dir, output_dir, model_path, model_type, emb_mode, emb_dim, n_classes, dataset):
        self.is_hvg = is_hvg
        self.ko_file_dir = ko_file_dir
        self.sub_file_dir = sub_file_dir
        self.output_dir = output_dir
        self.emb_mode = emb_mode
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.model_type = model_type
        self.model_path = model_path
        self.dataset = dataset

        if not os.path.exists(output_dir):
            print(f"Creating output directory at {output_dir}\n")
            os.makedirs(output_dir)

        self.adata = None
        if (is_hvg):
            self.adata = sc.read_h5ad("/work/magroup/kaileyhu/datasets/depmap/processed/hvg/omics_expr_hvg_500.h5ad")
        else:
            self.adata = sc.read_h5ad("/work/magroup/kaileyhu/datasets/depmap/processed/omics_expr.h5ad")

        self.n_cells = len(self.adata.X)
        self.n_genes = len(self.adata.X[0]) 

    def match_adata(self, df, is_sub):
        df.columns = [int(s) for s in df.columns]
        for patient in df.index:
            if patient not in self.adata.obs_names:
                print(f"Patient {patient} is missing")
                df.drop(patient, axis = 1)
                
        df = df[~df.index.duplicated(keep='first')]
        
        for patient in self.adata.obs_names:
            if patient not in df.index:
                if is_sub:  
                    df.loc[patient] = 0
                    
                else:
                    a = self.adata.obsm["orig_embedding"].loc[patient].copy()
                    df.loc[patient] = a
            
            if (is_sub):
                df.loc[patient] = df.loc[patient].fillna(0)
        
        df = df.reindex(self.adata.obs_names)
        return df

    def get_original_embs(self):
        print(f"Extracting original embeddings from dataset {self.dataset}")
        # embex = EmbExtractor(model_type=self.model_type,
        #              num_classes=self.n_classes,
        #              emb_mode=self.emb_mode,
        #              emb_layer=-1,
        #              emb_label=["patient_id"],
        #              forward_batch_size=100,
        #              max_ncells=self.n_cells,
        #              nproc=16)

        
        # embs = embex.extract_embs(self.model_path,
        #                           self.dataset,
        #                           "/work/magroup/kaileyhu/res/via_classifier/",
        #                           "orig_embedding_pretrained")
        # return embs
        embs = pd.read_csv("/work/magroup/kaileyhu/res/via_classifier/orig_embedding_pretrained.csv")
        return embs

    def remove_nan(self):
        embs = self.get_original_embs()
        print(f"Removing nan values from original embedding")
        embs.set_index("patient_id", inplace=True)

        if ("Unnamed: 0" in embs.columns):
            embs.drop(columns = ["Unnamed: 0"], inplace=True)
        # embs = self.match_adata(embs, False)
        embs.dropna(axis = 0, how = 'any', inplace = True)

        print("Original embedding has NAN values?", embs.isna().any().any())
        self.adata.obsm["orig_embedding"] = embs

    def process_ko(self):
        ko_files = os.listdir(self.ko_file_dir)
        print("Processing knockout files, total:", len(ko_files))

        ko_embs = {}
        for file in tqdm(ko_files):
            df_file = pd.read_csv(self.ko_file_dir+file)
            df_file.set_index("patient_id", inplace=True)

            df_file = self.dropna(axis = 0, how = 'any', inplace = True)
            ad_name = "gene_"+file.split('.')[0]
        
            if (df_file.isna().any().any()):
                print("Error: dataframe for", ad_name, "contains NAN values")
                print(df_file)
                break
                
            ko_embs[ad_name] = df_file
            
        self.adata.uns["knockout_embeddings"] = ko_embs

    def process_sub(self):
        sub_files = os.listdir(self.sub_file_dir) # get all perturbed csv's
        print("Processing subtracted files, total:", len(sub_files))
        
        delta_embs = {}
        
        for file in tqdm(sub_files):
            df_file = pd.read_csv(self.sub_file_dir+file)
            df_file.set_index("patient_id", inplace=True)
        
            df_file.dropna(axis = 0, how = 'any', inplace = True)
            ad_name = "gene_"+file.split('.')[0]
        
            if (df_file.isna().any().any()):
                print("Error: dataframe for", ad_name, "contains NAN values")
                print(df_file)
                break
                
            delta_embs[ad_name] = df_file
            
        self.adata.uns["embedding_differences"] = delta_embs

    def process_hadamard(self):
        ko_embs = self.adata.uns["knockout_embeddings"]
        orig_embs = self.adata.obsm["orig_embedding"]

        def multiply_lists(l1, l2):
            res = []
            for i in range(len(l1)):
                res.append(l1[i] * l2[i])
            return res

        hadamard = {}
        
        for gene in ko_embs:
            df_file = ko_embs[gene].copy()
            for patient in df_file.index:
                patient_ko = list(df_file.loc[patient])
                patient_orig = list(orig_embs.loc[patient])
                df_file.loc[patient] = multiply_lists(patient_ko, patient_orig)
            hadamard[gene] = df_file
        
        sub_files = os.listdir(self.sub_file_dir) # get all perturbed csv's
        print("Processing subtracted files, total:", len(sub_files))
        
        delta_embs = {}
        via_dict = {}
        
        for file in tqdm(sub_files):
            df_file = pd.read_csv(self.sub_file_dir+file)
            df_file.set_index("patient_id", inplace=True)
        
            via_scores = df_file['viability']
            via_scores = via_scores[~via_scores.index.duplicated(keep='first')]
        
            df_file = df_file.drop('viability', axis=1)
        
            original_idx = df_file.index
            
            df_file = self.match_adata(df_file, True)
            ad_name = "gene_"+file.split('.')[0]
        
            
            for patient in self.adata.obs_names:
                if (patient in original_idx):
                    if (not np.isnan(via_scores.loc[patient])):
                        via_dict[(patient, ad_name)] = via_scores.loc[patient]
            
            if (df_file.isna().any().any()):
                print("Error: dataframe for", ad_name, "contains NAN values")
                print(df_file)
                break
                
            delta_embs[ad_name] = df_file
            
        self.adata.uns["embedding_differences"] = hadamard
        self.adata.uns["viability_dict"] = via_dict

    def process_sub_no_via(self):
        sub_files = os.listdir(self.sub_file_dir) # get all perturbed csv's
        print("Processing subtracted files, total:", len(sub_files))
        
        delta_embs = {}
        
        for file in tqdm(sub_files):
            df_file = pd.read_csv(self.sub_file_dir+file)
            df_file.set_index("patient_id", inplace=True)
            df_file.drop(columns = ["viability"], inplace=True)
            
            df_file = self.match_adata(df_file, True)
            ad_name = "gene_"+file.split('.')[0]
            
            if (df_file.isna().any().any()):
                print("Error: dataframe for", ad_name, "contains NAN values")
                print(df_file)
                break
                
            delta_embs[ad_name] = df_file
            
        self.adata.uns["embedding_differences"] = delta_embs

    def save_df(self, save_via = True):
        res_file = f"{self.output_dir}/subtracted_embs_only.csv"

        print(f"Saving dataframe at {res_file}")
        delta_embs = self.adata.uns['embedding_differences']

        df = pd.DataFrame.from_dict(delta_embs, orient='index')

        if "512" in df.columns:
            df.rename(columns = {512 : "viability score"}, inplace = True)

        if (save_via):
            df.to_csv(res_file)
        else:
            df.drop(columns='viability score', inplace = True)
            df.to_csv(res_file)

    def save_df_knockout_sub(self):
        res_file = f"{self.output_dir}/knockout_sub_concat.csv"

        print(f"Saving dataframe at {res_file}")
        ko_embs = self.adata.uns['knockout_embeddings']
        delta_embs = self.adata.uns['embedding_differences']
        viability = self.adata.uns['viability_dict']

        diff_dict = {}
        ko_dict = {}

        for gene in tqdm(delta_embs):
            df_diff = delta_embs[gene]
            df_ko = ko_embs[gene]
            for patient in df_diff.index:
                if (patient, gene) in viability:
                    temp = (df_diff.loc[patient].tolist())
                    temp.append(viability[(patient, gene)])
                    diff_dict[(patient, gene)] = temp

            for patient in df_ko.index:
                if (patient, gene) in viability:
                    temp_ko = (df_ko.loc[patient].tolist())
                    ko_dict[(patient, gene)] = temp_ko

        df = pd.DataFrame.from_dict(diff_dict, orient='index')
        df_ko = pd.DataFrame.from_dict(ko_dict, orient='index')
        df_ko.columns = [f"knockout {s}" for s in df_ko.columns]
        
        df.rename(columns = {512 : "viability score"}, inplace = True)

        df_no_via = df.drop(columns='viability score')
        via = df['viability score']
        df_concat = pd.concat([df_no_via, df_ko, via], axis = 1)
        
        df_concat.to_csv(res_file)

    def save_df_orig_perturb(self):
        res_file = f"{self.output_dir}/orig_knockout_concat.csv"

        print(f"Saving dataframe at {res_file}")
        ko_embs = self.adata.uns['knockout_embeddings']
        orig_embs = self.adata.obsm["orig_embedding"]
        viability = self.adata.uns['viability_dict']

        orig_dict = {}
        ko_dict = {}

        for gene in tqdm(ko_embs):
            df_ko = ko_embs[gene]
            df_orig = orig_embs

            for patient in df_ko.index:
                if (patient, gene) in viability:
                    temp_ko = (df_ko.loc[patient].tolist())
                    temp_ko.append(viability[(patient, gene)])
                    ko_dict[(patient, gene)] = temp_ko

            for patient in df_orig.index:
                if (patient, gene) in viability:
                    temp_orig = (df_orig.loc[patient].tolist())
                    orig_dict[(patient, gene)] = temp_orig

        df_orig = pd.DataFrame.from_dict(orig_dict, orient='index')
        df_ko = pd.DataFrame.from_dict(ko_dict, orient='index')
        
        df_orig.columns = [f"original {s}" for s in df_orig.columns]
        df_ko.columns = [f"knockout {s}" for s in df_ko.columns]
        
        df_ko.rename(columns = {"knockout 512" : "viability score"}, inplace = True)

        df_no_via = df_ko.drop(columns='viability score')
        via = df_ko['viability score']
        df_concat = pd.concat([df_orig, df_no_via, via], axis = 1)
        
        df_concat.to_csv(res_file)

    # gets subtracted embeddings
    def proc_input(self):
        print(f"\nStarting input processing for viability perturber...\n")
        self.remove_nan()
        self.process_ko()
        self.process_sub()
        self.save_df()
        print("Input processing completed!")

    # gets hadamard product of embeddings
    def proc_input_hadamard(self):
        print(f"\nStarting input processing for viability perturber...\n")
        self.remove_nan()
        self.process_ko()
        self.process_hadamard()
        self.remove_via_col()
        self.save_df()
        print("Input processing completed!")

    def save_df_perturbed(self):
        res_file = f"{self.output_dir}/perturbed_embs_only.csv"

        print(f"Saving dataframe at {res_file}")
        ko_embs = self.adata.uns['knockout_embeddings']

        viability = self.adata.uns['viability_dict']

        ko_dict = {}

        for gene in tqdm(ko_embs):
            df_ko = ko_embs[gene]
            for patient in df_ko.index:
                if (patient, gene) in viability:
                    temp = (df_ko.loc[patient].tolist())
                    temp.append(viability[(patient, gene)])
                    df_ko[(patient, gene)] = temp

        df = pd.DataFrame.from_dict(ko_dict, orient='index')

        df.rename(columns = {512 : "viability score"}, inplace = True)
        
        df.to_csv(res_file)

    # gets perturbed embeddings
    def proc_input_perturbs(self):
        print(f"\nStarting input processing for viability perturber...\n")
        self.remove_nan()
        self.process_ko()
        self.process_sub()
        self.remove_via_col()
        self.save_df_perturbed()
        print("Input processing completed!")

    # gets perturbed embeddings
    def proc_input_subs(self):
        print(f"\nStarting input processing for viability perturber...\n")
        # self.remove_nan()
        # # self.process_ko()
        # self.process_sub()
        self.process_sub()

        print(f"Saving pkl file\n")
        with open ("/work/magroup/kaileyhu/res/perturbed/gf_12L_30M_i2048_SL/generated_df/delta_embs.pkl", "wb") as f:
            pkl.dump(self.adata.uns['embedding_differences'], f)

        print(f"Done saving pkl file\n")

        self.remove_via_col()
        self.save_df_sub()
        print("Input processing completed!")

    def proc_input_no_via(self):
        print(f"\nStarting input processing for viability perturber... not accounting for viability scores\n")
        self.remove_nan()
        self.process_ko()
        self.process_sub()
        self.save_df(save_via = False)
        print("Input processing completed!")

    def concat_perturb_sub(self):
        print(f"\nStarting input processing for viability perturber...\n")
        print(f"Will output a concatenation of the perturbed embedding and the subtracted embedding...\n")
        self.remove_nan()
        self.process_ko()
        self.process_sub()
        # self.remove_via_col()
        self.save_df_knockout_sub()
        print("Input processing completed!")

    def concat_orig_perturb(self):
        print(f"\nStarting input processing for viability perturber...\n")
        print(f"Will output a concatenation of the original embedding and the perturbed embedding...\n")
        self.remove_nan()
        self.process_ko()
        self.process_sub()
        self.remove_via_col()
        self.save_df_orig_perturb()
        print("Input processing completed!")

    def concat_gene_embs(self, gene_emb_dir, df_dir = None, file_name = None):
        if file_name is None:
            res_file = f"{self.output_dir}/emb_mat_with_gene_embs.csv"
        else:
            res_file = f"{self.output_dir}/{file_name}.csv"
        if (df_dir is None):
            self.proc_input()
            df = pd.read_csv(f"{self.output_dir}/gene_patient_emb_mat.csv")
        else:
            df = pd.read_csv(df_dir)

        df.set_index("Unnamed: 0", inplace = True)
        gene_embs = pd.read_csv(gene_emb_dir)
        gene_embs.set_index("Unnamed: 0", inplace = True)
        gene_dict = {}

        df_no_via = df.drop(columns='viability score')
        via = df['viability score']

        for entry in tqdm(df.index):
            e = ast.literal_eval(entry)
            gene = e[1][5:]
            try:
                gene_repr = gene_embs.loc[gene]
                # print(f"DF ENTRY: {df_no_via.loc[entry]}\nGENE REPR: {gene_repr}\nVIABILITY: {via.loc[entry]}")
                gene_dict[entry] = list(df_no_via.loc[entry]) + list(gene_repr) + [via.loc[entry]]
                # print(f"{gene_dict[entry]}")
            # return
            except:
                print(f"\n\n gene {gene} does not exist in the gene embedding representation, skipping")

        df_gene_columns = [f"gene {s}" for s in range(len(gene_repr))]
        df_columns = [f"{s}" for s in df_no_via.columns]

        print(f"Df gene columns: {df_gene_columns}\n")
        print(f"DF columns: {df_columns}\n")

        df_gene = pd.DataFrame(gene_dict).transpose()
        df_gene.columns = (df_columns + df_gene_columns + ["viability score"])
        
        if 'gene viability score' in df_gene.columns:
            df_gene.drop(columns='gene viability score', inplace = True)
            
        # df_no_via = df.drop(columns='viability score')
        # via = df['viability score']
        # df_concat = pd.concat([df_no_via, df_gene, via], axis = 1)
        df_gene.to_csv(res_file)

    def double_concat_gene_embs(self, gene_emb_dir, df_dir = None, file_name = None):
        if file_name is None:
            res_file = f"{self.output_dir}/emb_mat_with_TWO_gene_embs.csv"
        else:
            res_file = f"{self.output_dir}/{file_name}.csv"
            
        if (df_dir is None):
            self.proc_input()
            df = pd.read_csv(f"{self.output_dir}/gene_patient_emb_mat.csv")
        else:
            df = pd.read_csv(df_dir)

        df.set_index("Unnamed: 0", inplace = True)
        gene_embs = pd.read_csv(gene_emb_dir)
        gene_embs.set_index("Unnamed: 0", inplace = True)
        
        gene_dict = {}

        df_no_via = df.drop(columns='viability score')
        via = df['viability score']

        for entry in tqdm(df.index):
            e = ast.literal_eval(entry)
            gene = e[1][5:]
            try:
                gene_repr = gene_embs.loc[gene]
                gene_dict[entry] = list(df_no_via.loc[entry]) + list(gene_repr) + list(gene_repr) + [via.loc[entry]]
            # return
            except:
                print(f"\n\n gene {gene} does not exist in the gene embedding representation, skipping")

        df_gene_columns = [f"gene {s}" for s in range(len(gene_repr))]
        df_columns = [f"{s}" for s in df_no_via.columns]

        print(f"Df gene columns: {df_gene_columns}\n")
        print(f"DF columns: {df_columns}\n")

        df_gene = pd.DataFrame(gene_dict).transpose()
        df_gene.columns = (df_columns + df_gene_columns + ["viability score"])
        
        if 'gene viability score' in df_gene.columns:
            df_gene.drop(columns='gene viability score', inplace = True)

        df_gene.to_csv(res_file)

    def concat_esm_embs(self, start = 0, end = 0, df_dir = None):
        res_file = f"{self.output_dir}/emb_mat_with_ESM_embs.csv"

        if (df_dir is None):
            self.proc_input()
            df = pd.read_csv(f"{self.output_dir}/gene_patient_emb_mat.csv")
        else:
            df = pd.read_csv(df_dir)

        df.set_index("Unnamed: 0", inplace = True)

        if start != 0 or end != 0:
            df_subset = df[start : end]
            res_file = f"{self.output_dir}/emb_mat_with_ESM_embs_{start}_to_{end}.csv"
        else:
            df_subset = df

        gene_embs = torch.load('/work/magroup/shared/Heimdall/data/pretrained_embeddings/ESM2/protein_map_human_ensembl.pt')

        gene_dict = {}

        for entry in tqdm(df_subset.index):
            e = ast.literal_eval(entry)
            gene = e[1][5:]
            try:
                gene_repr = gene_embs[gene].tolist()
                gene_dict[entry] = gene_repr
            except:
                print(f"\n\n gene {gene} does not exist in the gene embedding representation, skipping")

        df_gene = pd.DataFrame(gene_dict).transpose()

        col_names = list(df_gene.columns)

        df_gene.columns = [f"gene {s}" for s in col_names]
        df_no_via = df_subset.drop(columns='viability score')
        via = df_subset['viability score']
        df_concat = pd.concat([df_no_via, df_gene, via], axis = 1)
        df_concat.to_csv(res_file)
