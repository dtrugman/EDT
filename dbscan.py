from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

class DBScan:

    DISTANCE_METRIC = 'euclidean'

    def __init__(self, df, labels=[], eps=0.5, min_samples=5):
        self.__df = df if not labels else df[labels]
        self.__clusterer = DBSCAN()
        self.__pred = self.__clusterer.fit(self.__df)

    @property
    def k(self):
        return self.labels.max() - self.labels.min() + 1

    @property
    def df(self):
        return self.__df

    @property
    def labels(self):
        return self.__clusterer.labels_
    
    @property
    def pred(self):
        return self.__pred

    @property
    def silhouette(self, metric=DISTANCE_METRIC):
        return silhouette_score(self.__df, self.pred, metric=metric)