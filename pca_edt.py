import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

from data_common import DAA

def print_df_pca(df, title, col_x, col_y, col_lbl, xlim=None, ylim=None):
    legend_handles = []
    legend_labels = []

    fig = plt.figure()
    ax = fig.add_subplot()

    if xlim:
        ax.set_xlim((-4, 4))

    if ylim:
        ax.set_ylim((-4, 4))

    colors = ['b', 'c', 'y', 'm', 'r', 'k', 'g']

    labels = df[col_lbl]
    labels_unique = np.unique(labels)

    for k in range(len(labels_unique)):
        k_label = labels_unique[k]

        k_df = df[df[col_lbl] == k]

        k_nda = k_df[k_df[DAA] == 0]
        k_daa = k_df[k_df[DAA] == 1]

        daa_handler = ax.scatter(k_daa.loc[:,col_x], k_daa.loc[:,col_y], marker='x', color=colors[k])
        nda_handler = ax.scatter(k_nda.loc[:,col_x], k_nda.loc[:,col_y], marker='o', color=colors[k])

        legend_handles.append(daa_handler)
        legend_handles.append(nda_handler)

        legend_labels.append(f'{k_label}-DAA')
        legend_labels.append(f'{k_label}-Not-DAA')

        p = len(k_daa)
        q = len(k_nda)
        t = len(k_df)
        print(f'Cluster {k}: DAA[{p}({100*p/t:.2f}%)] Not-DAA[{q}({100*q/t:.2f}%)] Total[{t}]')

    ax.legend(handles=legend_handles, labels=legend_labels, title=title, loc='best')
    plt.show()