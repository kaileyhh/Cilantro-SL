# Create an intelligent test-train split

import random
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None

def dict_add(d, k):
    if k not in d:
        d[k] = 1
    else:
        d[k] += 1
    return d

def dict_ap(d, k1, k2):
    return dict_add(dict_add(d, k1), k2)

def compute_degree(genes1, genes2):
    if len(genes1) != len(genes2):
        print (f"Error: length mismatch in gene pairs, {len(genes1)} and {len(genes2)}")
        return None

    deg_dict = {}
    for i in range(len(genes1)):
        deg_dict = dict_ap(deg_dict, genes1[i], genes2[i])

    return deg_dict

def get_min_fold(fold_sizes):
    return np.argmin(fold_sizes)

def test_train_split_CV1_pair_list(pair_list, num_folds):
    """
    genes1 : string list
    | a list of first element in the gene pairs included in the SL and non-SL pairs

    genes2 : string list
    | a list of second element in the gene pairs included in the SL and non-SL pairs

    num_folds : int
    | number of iterations to run cross validation for; test set will be approximately 1/num_folds of the gene pairs

    d : dataframe

    returns: (all_test, all_train) : (string list list) * (string list list) 
    | list of [num_folds] train sets and test sets
    """

    sl_pairs = list(pair_list.keys())
    random.shuffle(sl_pairs)

    test_pairs = [sl_pairs[i::num_folds] for i in range(num_folds)]
    train_pairs = [list(set(sl_pairs) - set(test_pairs[i])) for i in range(num_folds)]

    print(f"length of each test set: {[len(test_pairs[i]) for i in range(num_folds)]}")

    all_test = [[(g1, g2, pair_list[(g1, g2)]) for (g1, g2) in i] for i in test_pairs]
    all_train = [[(g1, g2, pair_list[(g1, g2)]) for (g1, g2) in i] for i in train_pairs]
    return all_test, all_train

    
# CV1 test train split - split by SL pairs
def test_train_split_CV1(genes1, genes2, d, num_folds, genes_only = False):
    """
    genes1 : string list
    | a list of first element in the gene pairs included in the SL and non-SL pairs

    genes2 : string list
    | a list of second element in the gene pairs included in the SL and non-SL pairs

    num_folds : int
    | number of iterations to run cross validation for; test set will be approximately 1/num_folds of the gene pairs

    d : dataframe

    returns: (all_test, all_train) : (string list list) * (string list list) 
    | list of [num_folds] train sets and test sets
    """

    sl_pairs = list(set(zip(genes1, genes2)))
    random.shuffle(sl_pairs)

    test_pairs = [sl_pairs[i::num_folds] for i in range(num_folds)]

    print(f"length of each test set: {[len(test_pairs[i]) for i in range(num_folds)]}")

    all_test = []
    all_train = []

    if genes_only:
        all_test = test_pairs 
        all_train = [list(set(sl_pairs) - set(test_pairs[i])) for i in range(num_folds)]
        return all_test, all_train

    d['pairs'] = list(zip(d['gene 1'], d['gene 2']))
    for i in range(num_folds):
        temp = d[d['pairs'].isin(test_pairs[i])]
        all_test.append(temp.drop(columns=['pairs']))

        temp = d[~d['pairs'].isin(test_pairs[i])]
        all_train.append(temp.drop(columns=['pairs']))

    return all_test, all_train

# CV3 test train split
def test_train_split(genes1, genes2, d, num_folds):
    """
    genes1 : string list
    | a list of first element in the gene pairs included in the SL and non-SL pairs

    genes2 : string list
    | a list of second element in the gene pairs included in the SL and non-SL pairs

    num_folds : int
    | number of iterations to run cross validation for; test set will be approximately 1/num_folds of the gene pairs

    d : dataframe

    returns: (all_test, all_train) : (string list list) * (string list list) 
    | list of [num_folds] train sets and test sets
    """

    single_genes = set(genes1) | set(genes2)

    # sorted in reverse order
    deg_dict = dict(sorted(compute_degree(genes1, genes2).items(), key=lambda item: item[1], reverse=True))

    folds = [set() for _ in range(num_folds)]
    fold_sizes = [0 for _ in range(num_folds)]

    for k in deg_dict:
        chosen_fold = get_min_fold(fold_sizes)
        folds[chosen_fold].add(k)
        fold_sizes[chosen_fold] += deg_dict[k]

    folds_comp = [(single_genes - folds[i]) for i in range(num_folds)]

    all_test = []
    all_train = []
    
    for i in range(num_folds):
        has_both = d[d['gene 1'].isin(folds[i]) & d['gene 2'].isin(folds[i])]
        has_none = d[(d['gene 1'].isin(folds_comp[i])) & (d['gene 2'].isin(folds_comp[i]))]
    
        test = has_both
        train = has_none 
    
        all_test.append(test)
        all_train.append(train)
    return all_test, all_train

def split_validation(train_df, val_frac=0.1):
    """
    Splits a training dataframe into training and validation dataframes.

    train_df : dataframe
    | dataframe containing training data

    val_frac : float
    | fraction of training data to use as validation data

    returns: (train_split, val_split) : (dataframe, dataframe)
    | training and validation dataframes
    """

    val_size = int(len(train_df) * val_frac)
    val_indices = set(random.sample(range(len(train_df)), val_size))

    val_split = train_df.iloc[list(val_indices)]
    train_split = train_df.drop(train_df.index[list(val_indices)])

    return train_split, val_split
    

    