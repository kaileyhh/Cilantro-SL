# generate input for dual knockout with specific SL (BRCA & PARP)
# NO GENE EMBEDDINGS

import sys

sys.path.append('/work/magroup/kaileyhu/synthetic_lethality/classifier/training/input_setup/')

from data_generator import data_generator

dg = data_generator(
        is_hvg = True,
        ko_file_dir = "/work/magroup/kaileyhu/res/perturbed/DUAL_BRCA_PARP/perturbed_embs/",
        sub_file_dir = "/work/magroup/kaileyhu/res/perturbed/DUAL_BRCA_PARP/subtracted_embs/",
        output_dir = "/work/magroup/kaileyhu/res/perturbed/DUAL_BRCA_PARP/generated_df",
        model_path = "/work/magroup/kaileyhu/Geneformer/gf-12L-30M-i2048/",
        model_type = "Pretrained",
        emb_mode = "cell",
        emb_dim = 512,
        n_classes = 0,
        dataset = "/work/magroup/kaileyhu/res/hvg_500_w_SL_tokenized_2048.dataset")

dg.proc_input_no_via()