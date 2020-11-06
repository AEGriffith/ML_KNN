"""
Author: Amanda Griffith
Course: EEL5840 - Fundamentals of Machine Learning
Assignment: 03
Purpose: Choose a dataset and create a K-NN Classifier
Due: November 10, 2020
"""
from sklearn import metrics
from sklearn.model_selection import cross_val_score

"""
Imports
"""
import pandas as pd
import processing as prep
import plot_data as plot
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import seaborn as sns


"""
Variables
"""
# Variables for seed data
seed_filename = 'files/seeds_dataset.txt'
seed_data_range = [0, 1, 2, 3, 4, 5, 6]
seed_label_range = [7]

# Variables for KNN parameters
knn_k = [1, 3, 5, 7, 9]
knn_weights = ['uniform', 'distance']
knn_algorithms = ['ball_tree', 'kd_tree', 'brute', 'auto']
knn_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
parameters = {'k': knn_k, 'weight': knn_weights, 'algorithm': knn_algorithms, 'metric': knn_metrics}

"""
Data Import
"""
# Import Data from seed data text file
seeds_df, seed_data, seed_labels = prep.data_import(seed_filename, seed_data_range, seed_label_range)


"""
Data Visualization
"""
# Plot features
fig, axes = plt.subplots(ncols=3, nrows=3)
axes = axes.flatten()
for i in seed_data_range:
    sns.histplot(seeds_df, x=seeds_df[i], hue=seeds_df[7], multiple="stack", ax=axes[i], legend=False)
plt.show()

"""
Data Visualization
"""


