import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from data_common import DAA

def print_pca(df, km):
    cond = df.loc[:,DAA] == 1
    daa_loc = df[cond].index
    non_daa_loc = df[~cond].index

    handles = []
    labels = []

    pca = PCA(2).fit_transform(km.df)

    colors = ['b', 'c', 'y', 'm', 'r', 'k', 'g']
    for k in range(km.k):
        label_k = [i for i, label in enumerate(km.labels) if label == k]

        pca_k_daa = pca[list(set(label_k).intersection(daa_loc)), :]
        pca_k_non_daa = pca[list(set(label_k).intersection(non_daa_loc)), :]

        daa_handler = plt.scatter(pca_k_daa[:,0], pca_k_daa[:,1], marker='x', color=colors[k])
        non_daa_handler = plt.scatter(pca_k_non_daa[:,0], pca_k_non_daa[:,1], marker='o', color=colors[k])

        handles.append(daa_handler)
        handles.append(non_daa_handler)

        if km.k == 1:
            labels.append(f'DAA')
            labels.append(f'Not-DAA')
        else:
            labels.append(f'{k}-DAA')
            labels.append(f'{k}-Not-DAA')

        p = len(pca_k_daa)
        q = len(pca_k_non_daa)
        t = len(label_k)
        print(f'Cluster {k}: DAA[{p}({100*p/t:.2f}%)] Not-DAA[{q}({100*q/t:.2f}%)] Total[{t}]')

    plt.legend(handles=handles, labels=labels, loc='best')
    plt.show()