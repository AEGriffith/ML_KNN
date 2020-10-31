"""
Imports
"""
import pandas as pd
import numpy as np
import sklearn.preprocessing as skprep

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
    :return: two numpy arrays in the order data, labels
    """
    # read file into dataframe
    # the seeds_dataset.txt has no headers
    df = pd.read_csv(filename, sep=sep, header=None)

    # create numpy arrays for the data and the labels
    data = df[range1].to_numpy()
    label = df[range2].to_numpy()

    # save the numpy arrays
    np.save('files/' + data_label + '.npy', data)
    np.save('files/' + label_label + '.npy', label)

    # return numpy arrays
    return data, label


"""
Data Preprocessing
"""


# MinMaxNormalization
def min_max_norm(data):
    scalar = skprep.MinMaxScaler()
    normalized_data = scalar.fit_transform(data)

    return normalized_data


# Z-score Normalization
def z_score_norm(data):
    scalar = skprep.StandardScaler()
    normalized_data = scalar.fit_transform(data)

    return normalized_data

