# Load libraries
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn.preprocessing import OrdinalEncoder


col_names = ['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y']
# load dataset
test_data = pd.read_csv(r"C:\Users\Jaden\OneDrive\Desktop\CS\DM\train.csv", header=None, names=col_names)
# print(test_data.head())
# Identify columns with mixed data types
mixed_type_columns = [col for col in test_data.columns if test_data[col].apply(lambda x: type(x) == str).any()]

# Convert mixed data type columns to a consistent data type (e.g., str)
for col in mixed_type_columns:
    test_data[col] = test_data[col].astype(str)

enc = OrdinalEncoder()
#encoded_data = pd.DataFrame(enc.fit_transform(test_data))
#print(encoded_data.head())
enc.fit(test_data[["Dates","Category","Descript","DayOfWeek","PdDistrict","Resolution","Address","X","Y"]])
test_data[["Dates","Category","Descript","DayOfWeek","PdDistrict","Resolution","Address","X","Y"]] = enc.fit_transform(
    test_data[["Dates","Category","Descript","DayOfWeek","PdDistrict","Resolution","Address","X","Y"]])

# print(test_data)


# Define the values of k
k_values = [3, 5, 10]

# Loop through different values of k
for k in k_values:
    # Create a KMeans instance with the specified number of clusters (k)
    kmeans = KMeans(n_clusters=k, random_state=0)
    
    # Fit the model to the training data
    kmeans.fit(test_data)
    
    # Get the centroids of the clusters
    centroids = kmeans.cluster_centers_
    
    # Print the centroids for the current value of k
    print(f"Centroids for k={k}:")
    print(centroids)
