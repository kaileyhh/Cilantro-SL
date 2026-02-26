import pandas as pd
import pickle as pkl
import sys

sys.path.append("/work/magroup/kaileyhu/synthetic_lethality")

import nn_helpers.via_film as via_film
import nn_helpers.pair_classifier as pc_module
import nn_helpers.training_framework as training_framework

path = "data/"
name = "gene2vec_1x_2026_restricted"
cv = 1
metrics_dict_path = f"/work/magroup/kaileyhu/res/ablations/cv{cv}/{name}.pkl"

with open("data/sl_pairs_5x_sampling.pkl", "rb") as f:
    pair_list = pkl.load(f)

print(f"Running general model with FiLM pretraining using EXP5 CV{cv}")

print("Viability embedding generation")
df = pd.read_hdf(f"{path}/gene2vec_emb_mat.h5", "table")
print(f"df shape is {df.shape}")

ve = via_film.via_film(df, input_length = 640)
framework = training_framework.Framework(ve, None, metrics_path=metrics_dict_path)
framework.df_to_SL_embs(
   f"{path}/gene2vec_5x_film_all.h5",
  f"{path}/gene2vec_5x_film_all_double.h5",
  batch_size=512,
  lr=0.001,
  num_epochs=100,
  name=name,
)

# ve.create_SL_embs(
#    pair_list,
#    f"{path}/gene2vec_1x_film_all_restricted.h5",
#     "/work/magroup/kaileyhu/res/IMPORTANT/via_embs/2026/gene2vec_1x_film_all_double_restricted.h5",
# )

d = pd.read_hdf(
    f"{path}/gene2vec_5x_film_all_double.h5", "table"
)

pc = pc_module.pair_classifier(d, model_type=pc_module.ModelType.COMBINED)
framework = training_framework.Framework(None, pc, metrics_path=metrics_dict_path)


nn_save_paths = [f"5x_cv1_{i}.pth" for i in range(5)]

test, train = pc.setup_cv(5, cv=cv)
framework.all_test = test
framework.all_train = train

framework.run_cv(test, train, cv=cv, nn_save_paths=nn_save_paths)

framework.folds = 5

framework.uncertainty_quantification(
    training_framework.UQ.MONDRIAN_CONFORMAL,
    net=nn_save_paths,
    mondrian_class_dict=None
)