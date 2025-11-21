from viability_perturber import via_perturber

vp = via_perturber("/work/magroup/kaileyhu/res/hvg_500_w_SL_tokenized_2048.dataset", 
                   "/work/magroup/kaileyhu/Geneformer/gf-12L-30M-i2048/", 
                   0, 
                   "Pretrained", 
                   "/work/magroup/kaileyhu/res/gf_12L_30M_i2048_SL/", 
                   "cell",
                   temp_files = "/work/magroup/kaileyhu/outputs/dual_knockout/torch_batches",
                   hvg_viable_file = "/work/magroup/kaileyhu/res/viable_files/Human_SL_genes.pkl")

vp.perturb_all_genes()
