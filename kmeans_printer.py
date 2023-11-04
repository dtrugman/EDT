import matplotlib.pyplot as plt

def print_per_game_kmeans(km, labels, print_labels=False, plot_graph=False):
    if print_labels:
        print('Labels:', km.labels, '\n')

    games_count = len(labels)
    labels_count = len(labels[0])

    for i in range(km.k):
        print(f'Cluster #{i}:')
        for g in range(games_count):
            print([f'{x:.3f}' for x in km.centroids[i, g * labels_count : (g+1) * labels_count]])

    if not plot_graph:
        return

    _, axs = plt.subplots(games_count, figsize=(10, 10))
    for g in range(games_count):
        graph_labels = labels[g]
        graph_centers = km.centroids[:, g * labels_count : (g+1) * labels_count]

        for k in range(km.k):
            axs[g].plot(graph_labels, graph_centers[k,:], label=f'Cluster #{k}')

        axs[g].set_ylim([1.5, 4])
        axs[g].legend(loc='upper right')