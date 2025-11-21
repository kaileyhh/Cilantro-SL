from data_generator import data_generator

dg = data_generator(
        is_hvg = True,
        ko_file_dir = "/work/magroup/kaileyhu/res/gf_12L_30M_i2048_HVG/perturbed_embs/",
        sub_file_dir = "/work/magroup/kaileyhu/res/gf_12L_30M_i2048_HVG/subtracted_embs/",
        output_dir = "/work/magroup/kaileyhu/res/gf_12L_30M_i2048_HVG/generated_df",
        model_path = "/work/magroup/kaileyhu/Geneformer/gf-12L-30M-i2048/",
        model_type = "Pretrained",
        emb_mode = "cell",
        emb_dim = 512,
        n_classes = 0,
        dataset = "/work/magroup/kaileyhu/res/hvg_500_tokenized_2048.dataset")

dg.concat_orig_perturb()