import sys

sys.path.append('/work/magroup/kaileyhu/synthetic_lethality/prediction/input_setup/')


from data_generator import data_generator

dg = data_generator(
        is_hvg = True,
        ko_file_dir = "/work/magroup/kaileyhu/res/perturbed/gf_12L_30M_i2048_SL/perturbed_embs/",
        sub_file_dir = "/work/magroup/kaileyhu/res/perturbed/gf_12L_30M_i2048_SL/subtracted_embs/",
        output_dir = "/work/magroup/kaileyhu/res/perturbed/gf_12L_30M_i2048_SL/hadamard_df",
        model_path = "/work/magroup/kaileyhu/Geneformer/gf-12L-30M-i2048/",
        model_type = "Pretrained",
        emb_mode = "cell",
        emb_dim = 512,
        n_classes = 0,
        dataset = "/work/magroup/kaileyhu/res/hvg_500_w_SL_tokenized_2048.dataset")

dg.proc_input_hadamard()