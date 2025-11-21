import pandas as pd
import pickle as pkl
import sys

sys.path.append("/work/magroup/kaileyhu/synthetic_lethality")

import nn_helpers.via_film as via_film
import nn_helpers.pair_classifier as pc_module
import nn_helpers.training_framework as training_framework
import nn_helpers.via_embedder as via_embedder

path = "/work/magroup/kaileyhu/res/perturbed/gf_12L_30M_i2048_SL/gene2vec_df"
name = "gene2vec_full_metrics"
cv = 1
metrics_dict_path = f"/work/magroup/kaileyhu/res/ablations/cv{cv}/{name}.pkl"

with open("data/SL_pairs_sampled.pkl", "rb") as f:
    pair_list = pkl.load(f)

print(f"Running general model without FiLM pretraining using EXP5 CV{cv}")

print("Viability embedding generation")
df = pd.read_hdf(f"{path}/gene2vec_emb_mat.h5", "table")
print(f"df shape is {df.shape}")

ve = via_embedder.via_embedder(df, input_length = 640)
framework = training_framework.Framework(ve, None, metrics_path=metrics_dict_path)
framework.df_to_SL_embs(
   f"{path}/no_film_gene2vec_all.h5",
  "/work/magroup/kaileyhu/res/IMPORTANT/via_embs/no_film_gene2vec_all_double.h5",
  batch_size=512,
  lr=0.001,
  num_epochs=100,
  name=name,
)

# ve.create_SL_embs(
#    pair_list,
#    f"{path}/perturbed_ESM_film_all.h5",
#     "/work/magroup/kaileyhu/res/IMPORTANT/via_embs/perturbed_ESM_film_all_double.h5",
# )

d = pd.read_hdf(
    "/work/magroup/kaileyhu/res/IMPORTANT/via_embs/gene2vec_film_all_double.h5", "table"
)

pc = pc_module.pair_classifier(d, model_type=pc_module.ModelType.COMBINED)
all_test, all_train = pc.setup_cv(5, cv=cv)

framework = training_framework.Framework(None, pc, metrics_path=metrics_dict_path)

nn_save_paths = [f"net{i}.pth" for i in range(5)]
framework.all_test = all_test
framework.all_train = all_train

framework.run_cv(all_test, all_train, cv=cv, nn_save_paths=nn_save_paths)

framework.folds = 5

framework.uncertainty_quantification(
    training_framework.UQ.MONDRIAN_CONFORMAL,
    net=nn_save_paths,
    mondrian_class_dict=None
)