"""
Author: Amanda Griffith
Course: EEL5840 - Fundamentals of Machine Learning
Assignment: 03
Purpose: Choose a dataset and create a K-NN Classifier
Due: November 10, 2020
"""

"""
Imports
"""
import processing as prep
import plot_data as plot

"""
Variables
"""
# Variables for seed data
seed_filename = 'files/seeds_dataset.txt'
seed_data_range = [0, 1, 2, 3, 4, 5, 6]
seed_label_range = [7]

"""
Data Import and Processing
"""
# Import Data from seed data text file and store as numpy arrays
seed_data, seed_labels = prep.data_import(seed_filename, seed_data_range, seed_label_range)

# MinMax Data Normalization
mm_seed = prep.min_max_norm(seed_data)

# Z-score Data Normalization
z_seed = prep.z_score_norm(seed_data)

# Plot Normalizations
# plot.hist(mm_seed[:,0], mm_seed[:,1], mm_seed[:,2], mm_seed[:,3], mm_seed[:,4], mm_seed[:,5],
#              mm_seed[:, 6])
# plot.hist(z_seed[:,0], z_seed[:,1], z_seed[:,2], z_seed[:,3], z_seed[:,4], z_seed[:,5],
#              z_seed[:, 6], False)

