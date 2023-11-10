import array
import string
import pandas as pd


def construct_cat_value(csv_file: string, categories: array):
    df = pd.read_csv(csv_file)
    for category in categories:
        df[category] = df[category].astype('category')
        df[category] = df[category].cat.codes

    return df
