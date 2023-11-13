from sklearn.cluster import KMeans as SK_KMeans
from sklearn.metrics import silhouette_score

class KMeans:

    INIT = 'k-means++' # 'random'
    N_INIT = 'auto'
    RANDOM_STATE = None # Random
    DISTANCE_METRIC = 'euclidean'

    def __init__(self, df, k, labels=[], init=INIT, n_init=N_INIT, random_state=RANDOM_STATE):
        self.__df = df if not labels else df[labels]
        self.__k = k
        self.__clusterer = SK_KMeans(n_clusters=self.__k, init=init, n_init=n_init, random_state=random_state)
        self.__pred = self.__clusterer.fit_predict(self.__df)

    @property
    def df(self):
        return self.__df

    @property
    def k(self):
        return self.__k

    @property
    def labels(self):
        return self.__clusterer.labels_
    
    @property
    def centroids(self):
        return self.__clusterer.cluster_centers_
    
    @property
    def pred(self):
        return self.__pred

    @property
    def silhouette(self, metric=DISTANCE_METRIC):
        return silhouette_score(self.__df, self.pred, metric=metric)