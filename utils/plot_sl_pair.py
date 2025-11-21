import matplotlib.pyplot as plt
import numpy as np

def calc_sl_metrics(df, gene1, gene2):
    dead_blank = df.loc[df[gene1] == 1]
    alive_blank = df.loc[df[gene1] == 0]
    
    dead_dead = dead_blank.loc[dead_blank[gene2] == 1]["SL"]
    dead_alive = dead_blank.loc[dead_blank[gene2] == 0]["SL"]
    alive_dead = alive_blank.loc[alive_blank[gene2] == 1]["SL"]
    alive_alive = alive_blank.loc[alive_blank[gene2] == 0]["SL"]
    
    num_dead_from_da = list(dead_alive).count(1)
    num_alive_from_da = list(dead_alive).count(0)
    num_dead_from_dd = list(dead_dead).count(1)
    num_alive_from_dd = list(dead_dead).count(0)
    
    num_dead_from_aa = list(alive_alive).count(1)
    num_alive_from_aa = list(alive_alive).count(0)
    num_dead_from_ad = list(dead_alive).count(1)
    num_alive_from_ad = list(dead_alive).count(0)

    return ([num_dead_from_da, num_dead_from_aa, num_dead_from_dd, num_dead_from_ad], 
            [num_alive_from_da, num_alive_from_aa, num_alive_from_dd, num_alive_from_ad], 
            [f"{gene1} dead, {gene2} alive", 
             f"{gene1} alive, {gene2} alive", 
             f"{gene1} dead, {gene2} dead", 
             f"{gene1} alive, {gene2} dead"])
    

def plot_sl_pair(dead_list, alive_list, x_names, title):
    plt.figure(figsize=(16,6))
    plt.title(title)

    plt.bar(x_names,
            dead_list, 
            color='crimson')

    plt.bar(x_names,
            alive_list, 
            bottom = dead_list,
            color='cornflowerblue')

def plot_with_color_dim(y_axis, color_vals, size = 0.1, color_label = "predicted viability score", x_label="test data", y_label = "true viability score", title = "viability score predictions on the test set"):
    x = np.random.rand(len(y_axis))

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x, y_axis.tolist(), c=color_vals, cmap='viridis', s = size)
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_label)
    
    # Labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

def plot_viability_buckets(data, minval, maxval, step = 0.01):
    buckets = np.arange(minval, maxval, step)
    scores_rounded = list(map((lambda x : round(x, 2)), data))

    to_plot = []
    for b in buckets:
        to_plot.append(scores_rounded.count(round(b, 2)))

    plt.bar(buckets, to_plot)