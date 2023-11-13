import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF

from data_common import DAA

def print_pca(df, km, n=2):
    if not 2 <= n <= 3:
        raise RuntimeError('Unsupported number of components')

    pca = PCA(n_components=n, svd_solver='full').fit_transform(km.df)
    __print_decomposition(pca, df, km)

def print_nmf(df, km, n=2):
    if not 2 <= n <= 3:
        raise RuntimeError('Unsupported number of components')

    nmf = NMF(n_components=n, max_iter=1000).fit_transform(km.df)
    __print_decomposition(nmf, df, km)

def __print_decomposition(decomp, df, km):
    cond = df.loc[:,DAA] == 1
    daa_loc = df[cond].index
    non_daa_loc = df[~cond].index

    handles = []
    labels = []

    is_3d = (decomp.shape[1] == 3)

    fig = plt.figure()
    if is_3d:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()

    colors = ['b', 'c', 'y', 'm', 'r', 'k', 'g']
    for k in range(km.k):
        label_k = [i for i, label in enumerate(km.labels) if label == k]

        k_daa = decomp[list(set(label_k).intersection(daa_loc)), :]
        k_non_daa = decomp[list(set(label_k).intersection(non_daa_loc)), :]

        if is_3d:
            daa_handler = ax.scatter(k_daa[:,0], k_daa[:,1], k_daa[:,2], marker='x', color=colors[k])
            non_daa_handler = ax.scatter(k_non_daa[:,0], k_non_daa[:,1], k_non_daa[:,2], marker='o', color=colors[k])
        else:
            daa_handler = ax.scatter(k_daa[:,0], k_daa[:,1], marker='x', color=colors[k])
            non_daa_handler = ax.scatter(k_non_daa[:,0], k_non_daa[:,1], marker='o', color=colors[k])

        handles.append(daa_handler)
        handles.append(non_daa_handler)

        if km.k == 1:
            labels.append(f'DAA')
            labels.append(f'Not-DAA')
        else:
            labels.append(f'{k}-DAA')
            labels.append(f'{k}-Not-DAA')

        p = len(k_daa)
        q = len(k_non_daa)
        t = len(label_k)
        print(f'Cluster {k}: DAA[{p}({100*p/t:.2f}%)] Not-DAA[{q}({100*q/t:.2f}%)] Total[{t}]')

    ax.legend(handles=handles, labels=labels, loc='best')
    plt.show()