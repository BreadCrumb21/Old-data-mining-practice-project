import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix
import dask.dataframe as dd

from numpy import savetxt

# Import the data required from a CSV file named 'adultDataCorrected.csv' into a Pandas DataFrame
data = pd.read_csv('train.csv/train.csv')

# Convert categorical columns to category data type and create new numerical columns
# These new numerical columns will contain the category codes of the original categorical data
data['Dates'] = data['Dates'].astype('category')
data['Category'] = data['Category'].astype('category')
data['Descript'] = data['Descript'].astype('category')
data['DayOfWeek'] = data['DayOfWeek'].astype('category')
data['PdDistrict'] = data['PdDistrict'].astype('category')
data['Resolution'] = data['Resolution'].astype('category')
data['Address'] = data['Address'].astype('category')
data['X'] = data['X'].astype('category')
data['Y'] = data['Y'].astype('category')

data['Dates_new'] = data['Dates'].cat.codes
data['Category_new'] = data['Category'].cat.codes
data['Descript_new'] = data['Descript'].cat.codes
data['DayOfWeek_new'] = data['DayOfWeek'].cat.codes
data['PdDistrict_new'] = data['PdDistrict'].cat.codes
data['Resolution_new'] = data['Resolution'].cat.codes
data['Address_new'] = data['Address'].cat.codes
data['X_new'] = data['X'].cat.codes
data['Y_new'] = data['Y'].cat.codes

# Create an instance of the OneHotEncoder
encoder = OneHotEncoder()

# Perform one-hot encoding on selected columns and convert the result to a DataFrame
enc_data_sparse = csr_matrix(encoder.fit_transform(data[['Dates_new', 'Category_new',
                                                         'Descript_new', 'DayOfWeek_new',
                                                         'PdDistrict_new', 'Resolution_new', 'Address_new',
                                                         'X_new', 'Y_new']]))

# Convert Pandas DataFrame to Dask DataFrame
dask_data = dd.from_pandas(data, npartitions=10)

# Apply one-hot encoding using Dask
dask_encoded = dask_data[['Dates_new', 'Category_new', 'Descript_new', 'DayOfWeek_new',
                          'PdDistrict_new', 'Resolution_new', 'Address_new', 'X_new', 'Y_new']].categorize().compute()

# Save the result to a CSV file
dask_encoded.to_csv('hotData.csv', index=False)
