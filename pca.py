import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

from data_common import DAA

def print_pca(df, features, labels, xlim=None, ylim=None):
    n = 2
    pca = PCA(n_components=n, svd_solver='full').fit_transform(df[features])
    print_decomposition(pca, df, labels, xlim, ylim)

def print_nmf(df, features, labels, xlim=None, ylim=None, max_iter=1000):
    n = 2
    nmf = NMF(n_components=n, max_iter=max_iter).fit_transform(df[features])
    print_decomposition(nmf, df, labels, xlim, ylim)

def print_decomposition(decomp, df, labels, xlim, ylim):
    cond = df.loc[:,DAA] == 1
    daa_loc = df[cond].index
    non_daa_loc = df[~cond].index

    legend_handles = []
    legend_labels = []

    fig = plt.figure()
    ax = fig.add_subplot()

    if xlim:
        ax.set_xlim((-4, 4))
    if ylim:
        ax.set_ylim((-4, 4))

    colors = ['b', 'c', 'y', 'm', 'r', 'k', 'g']
    unique_labels = np.unique(labels)
    for k in range(len(unique_labels)):
        k_label = unique_labels[k]
        k_indices = [i for i, label in enumerate(labels) if label == k_label]

        k_daa = decomp[list(set(k_indices).intersection(daa_loc)), :]
        k_non_daa = decomp[list(set(k_indices).intersection(non_daa_loc)), :]

        daa_handler = ax.scatter(k_daa[:,0], k_daa[:,1], marker='x', color=colors[k])
        non_daa_handler = ax.scatter(k_non_daa[:,0], k_non_daa[:,1], marker='o', color=colors[k])

        legend_handles.append(daa_handler)
        legend_handles.append(non_daa_handler)

        if len(unique_labels) == 1:
            legend_labels.append(f'DAA')
            legend_labels.append(f'Not-DAA')
        else:
            legend_labels.append(f'{k_label}-DAA')
            legend_labels.append(f'{k_label}-Not-DAA')

        p = len(k_daa)
        q = len(k_non_daa)
        t = len(k_indices)
        print(f'Cluster {k}: DAA[{p}({100*p/t:.2f}%)] Not-DAA[{q}({100*q/t:.2f}%)] Total[{t}]')

    ax.legend(handles=legend_handles, labels=legend_labels, loc='best')
    plt.show()