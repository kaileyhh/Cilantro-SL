import sys
import os
import torch
import pandas as pd 
import numpy as np
import scanpy as sc
from tqdm import tqdm
import pickle as pkl

# Ensure Geneformer is in path
sys.path.append('/work/magroup/kaileyhu/Geneformer')

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

        if is_hvg:
            self.adata = sc.read_h5ad("/work/magroup/kaileyhu/datasets/depmap/processed/hvg/omics_expr_hvg_500.h5ad")
        else:
            self.adata = sc.read_h5ad("/work/magroup/kaileyhu/datasets/depmap/processed/hvg/SL/omics_expr_hvg_w_SL.h5ad")

        self.n_cells = len(self.adata.X)

    def get_original_embs(self):
        # Using the pre-extracted CSV as per your implementation
        embs = pd.read_csv("/work/magroup/kaileyhu/res/via_classifier/orig_embedding_pretrained.csv")
        return embs

    def remove_nan(self):
        embs = self.get_original_embs()
        print(f"Removing nan values from original embedding")
        embs.set_index("patient_id", inplace=True)

        if "Unnamed: 0" in embs.columns:
            embs.drop(columns=["Unnamed: 0"], inplace=True)
        
        embs.dropna(axis=0, how='any', inplace=True)
        print("Original embedding has NAN values?", embs.isna().any().any())
        self.adata.obsm["orig_embedding"] = embs

    def process_ko(self):
        ko_files = os.listdir(self.ko_file_dir)
        print("Processing knockout files, total:", len(ko_files))

        ko_embs = {}
        for file in tqdm(ko_files):
            df_file = pd.read_csv(os.path.join(self.ko_file_dir, file))
            df_file.set_index("patient_id", inplace=True)
            df_file.dropna(axis=0, how='any', inplace=True)
            
            ad_name = "gene_" + file.split('.')[0]
            ko_embs[ad_name] = df_file
            
        self.adata.uns["knockout_embeddings"] = ko_embs

    def process_sub(self):
        sub_files = os.listdir(self.sub_file_dir)
        print("Processing subtracted files, total:", len(sub_files))

        sub_embs = {}
        via_dict = {}
        
        for file in tqdm(sub_files):
            df_file = pd.read_csv(os.path.join(self.sub_file_dir, file))
            df_file.set_index("patient_id", inplace=True)
            df_file.dropna(axis=0, how='any', inplace=True)

            via_scores = df_file['viability']
            via_scores = via_scores[~via_scores.index.duplicated(keep='first')]
            
            ad_name = "gene_" + file.split('.')[0]
            original_idx = df_file.index

            for patient in self.adata.obs_names:
                if patient in original_idx:
                    if not np.isnan(via_scores.loc[patient]):
                        via_dict[(patient, ad_name)] = via_scores.loc[patient]
            
            # Drop viability before storing embedding diffs
            df_file_embs = df_file.drop(columns=['viability'], errors='ignore')
            sub_embs[ad_name] = df_file_embs
            
        self.adata.uns["embedding_differences"] = sub_embs
        self.adata.uns["viability_dict"] = via_dict

    def save_df_sub(self):
        """Fixes the ValueError by aggregating into a dictionary before DF creation"""
        res_file = f"{self.output_dir}/sub_embs_only.csv"
        print(f"Saving dataframe at {res_file}")
        
        sub_embs_dict = self.adata.uns['embedding_differences']
        viability = self.adata.uns['viability_dict']
        final_dict = {}

        for gene, df_sub in tqdm(sub_embs_dict.items()):
            for patient in df_sub.index:
                if (patient, gene) in viability:
                    row = df_sub.loc[patient].tolist()
                    row.append(viability[(patient, gene)])
                    final_dict[(patient, gene)] = row

        df_final = pd.DataFrame.from_dict(final_dict, orient='index')
        # Assign column names dynamically
        num_dims = df_final.shape[1] - 1
        df_final.columns = [f"dim_{i}" for i in range(num_dims)] + ["viability score"]
        df_final.to_csv(res_file)

    def save_df_perturbed(self):
        """Fixes the ValueError for knockout embeddings"""
        res_file = f"{self.output_dir}/perturbed_embs_only.csv"
        print(f"Saving dataframe at {res_file}")
        
        ko_embs = self.adata.uns['knockout_embeddings']
        viability = self.adata.uns['viability_dict']
        final_dict = {}

        for gene, df_ko in tqdm(ko_embs.items()):
            for patient in df_ko.index:
                if (patient, gene) in viability:
                    row = df_ko.loc[patient].tolist()
                    row.append(viability[(patient, gene)])
                    final_dict[(patient, gene)] = row

        df_final = pd.DataFrame.from_dict(final_dict, orient='index')
        num_dims = df_final.shape[1] - 1
        df_final.columns = [f"dim_{i}" for i in range(num_dims)] + ["viability score"]
        df_final.to_csv(res_file)

    def proc_input_subs(self):
        print(f"\nStarting input processing for viability perturber (Subtractions)...\n")
        self.remove_nan()
        self.process_sub()

        print(f"Saving pkl file\n")
        with open(f"{self.output_dir}/sub_embs.pkl", "wb") as f:
            pkl.dump(self.adata.uns['embedding_differences'], f)

        self.save_df_sub()
        print("Input processing completed!")

    def proc_input_perturbs(self):
        print(f"\nStarting input processing for viability perturber (Knockouts)...\n")
        self.remove_nan()
        self.process_ko()
        self.process_sub() # Required to populate viability_dict
        self.save_df_perturbed()
        print("Input processing completed!")