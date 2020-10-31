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
