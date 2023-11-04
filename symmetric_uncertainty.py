from scipy.stats import entropy 
import numpy as np
import pandas as pd

class SUT:

    EPS = 1e-5

    def __init__(self, df, index=None, columns=None):
        self.__df = df

        self.__index = index if index is not None else sorted(df['DAA'].unique())
        self.__columns = columns if index is not None else sorted(df['Cluster'].unique())

        self.__df_counts = self.__build_matrix_of_counts(self.__df)
        self.__df_p = self.__build_matrix_of_p(self.__df_counts, self.__index, self.__columns)

        self.__daa_counts = self.__df_counts.sum(axis=1)
        self.__daa_p = self.__counts_to_p(self.__daa_counts)

        self.__cluster_counts = self.__df_counts.sum(axis=0)
        self.__cluster_p = self.__counts_to_p(self.__cluster_counts)

    @staticmethod
    def __build_matrix_of_counts(df):
        return pd.crosstab(index=df['DAA'], columns=df['Cluster'])

    @staticmethod
    def __build_matrix_of_p(df_count, index, columns):
        odds = df_count.div(df_count.sum(axis=1).sum(axis=0), axis=0)
        return odds.reindex(index=index, columns=columns, fill_value=0)

    @staticmethod
    def __counts_to_p(occurences):
        return occurences.div(occurences.sum())

    def __str__(self):
        return f'''
Samples:               {self.len}
DAA (X) total:         {self.__daa_counts.tolist()}
DAA (X) p:             {self.__daa_p.tolist()}
DAA (X) entropy:       {self.__daa_entropy}
Cluster (Y) total:     {self.__cluster_counts.tolist()}
Cluster (Y) p:         {self.__cluster_p.tolist()}
Cluster (Y) entropy:   {self.__cluster_entropy}
Joint entropy:         {self.__joint_entropy}
Mutual info:           {self.__mutual_information}
Symmetric uncertainty: {self.symmetric_uncertainty}
KL-divergence:         {self.kl_divergence}
Total var distance:    {self.total_variation_distance}
{self.__df_counts.to_string()}
{self.__df_p.to_string()}
'''

    @property
    def len(self):
        return len(self.__df)

    @property
    def index(self):
        return self.__index
    
    @property
    def columns(self):
        return self.__columns

    @property
    def df(self):
        return self.__df

    @property
    def df_counts(self):
        return self.__df_counts

    @property
    def df_p(self):
        return self.__df_p

    @property
    def __cluster_p_for_non_daa(self):
        return self.__df_p.loc[0]

    @property
    def __cluster_p_for_daa(self):
        return self.__df_p.loc[1]

    @property
    def __daa_entropy(self):
        return entropy(self.__daa_p + self.EPS, base=2)

    @property
    def __cluster_entropy(self):
        return entropy(self.__cluster_p + self.EPS, base=2)

    @property
    def __joint_entropy(self):
        joint_p = np.concatenate([self.__cluster_p_for_daa, self.__cluster_p_for_non_daa])
        return entropy(joint_p + self.EPS, base=2)

    @property
    def __mutual_information(self):
        return self.__daa_entropy + self.__cluster_entropy - self.__joint_entropy

    @property
    def symmetric_uncertainty(self):
        return (2 * self.__mutual_information) / (self.__daa_entropy + self.__cluster_entropy)

    @property
    def total_variation_distance(self):
        return 0.5 * sum(abs(self.__cluster_p_for_daa - self.__cluster_p_for_non_daa))

    @property
    def kl_divergence(self):
        pk = self.__cluster_p_for_daa + self.EPS
        qk = self.__cluster_p_for_non_daa + self.EPS
        pq = entropy(pk, qk=qk, base=2)
        qp = entropy(qk, qk=pk, base=2)
        return sum([pq, qp]) / 2

    @property
    def metric(self):
        return self.symmetric_uncertainty