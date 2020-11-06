"""
Imports
"""
import pandas as pd
import numpy as np
import sklearn.preprocessing as skprep
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

"""
Data Setup Functions
"""


def data_import(filename: str, range1: list, range2: list, sep: str = '\t', data_label: str = 'data', label_label:
str = 'labels'):
    """
    :param filename: string with file location to be read by pandas.read_csv
    :param range1: a list of columns that contain the data to extract
    :param range2: a list of columns that contain the data labels
    :param sep: separator character. Default is tab.
    :param data_label: name for saved numpy array for data
    :param label_label: name for saved numpy array for labels
    :return: full dataframe and two numpy arrays in the order data, labels
    """
    # read file into dataframe
    # the seeds_dataset.txt has no headers
    df = pd.read_csv(filename, sep=sep, header=None)

    # create numpy arrays for the data and the labels
    data = df[range1].to_numpy()
    label = df[range2].to_numpy()

    return df, data, label


"""
Data Preprocessing
"""


# MinMax Pipeline
def min_max_knn(x, y, k, weight, algorithm, metric):
    scalar = skprep.MinMaxScaler()
    pipeline = make_pipeline(scalar, KNeighborsClassifier(n_neighbors=k, weights=weight, algorithm=algorithm, metric=metric))
    pipeline.fit(x, y)

    return pipeline


# Z-score Pipeline
def z_score_knn(x, y, k, weight, algorithm, metric):
    scalar = skprep.StandardScaler()
    pipeline = make_pipeline(scalar, KNeighborsClassifier(k, weight, algorithm, metric))
    pipeline.fit(x, y)

    return pipeline

"""
Data Validation
"""
