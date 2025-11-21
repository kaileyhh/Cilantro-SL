# get gene names from df

from ast import literal_eval

def get_genes_from_index(x):
    tup = literal_eval(x)
    return tup[1]

def get_gene_1_from_index(x):
    tup = literal_eval(x)
    return tup[1]

def get_gene_2_from_index(x):
    tup = literal_eval(x)
    return tup[2]

def get_patient_name_from_index(x):
    tup = literal_eval(x)
    return tup[0]

# discretize function...

def discrete_via(x, threshold):
    if (x <= threshold):
        return 1
    else:
        return 0