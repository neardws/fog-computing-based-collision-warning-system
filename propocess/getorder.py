import numpy as np
import pandas as pd


def get_order(filename):
    read_csv(filename)


def read_csv(fliename):
    df = pd.read_csv(fliename)
    print("Data example:")
    print(df.head(10))
    print("Data shape:")
    print(df.shape)
    print("Data describe")
    print(df.describe())
    groups = df.drop_duplicates().sort_values(by=['time']).groupby('vehicleID')
    for group in groups:
        pass
