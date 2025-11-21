 Regular pre-trained model:

**SAVES EVERYTHING IN VIA_CLASSIFIER FOLDER IN RES**
- Take in knockout and subtrated embeddings and turn it into an adata object using `./ko_to_adata.ipynb`
- Use `./dict_to_df.ipynb` to turn these adata objects into a dataframe
   - Saves it as a csv in `/work/magroup/kaileyhu/res/via_classifier/gene_patient_emb_mat.csv`
- Merge for new 95M model is in `./pretrained_input_setup.ipynb`
   - Saves it as a csv in `/work/magroup/kaileyhu/res/via_classifier/pretrained/gene_patient_emb_mat.csv`
 
CLCancer pre-trained model:
- Entire pipeline lies in `./cancer_input_setup.ipynb`
- Saves it as a csv in `/work/magroup/kaileyhu/res/via_classifier/cancer/gene_patient_emb_mat.csv`