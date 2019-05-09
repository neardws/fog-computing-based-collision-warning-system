import numpy as np
import pandas as pd


def read_csv(fliename):
    df = pd.read_csv(fliename, error_bad_lines=False)
    print("Data example:")
    print(df.head(10))
    print("Data shape:")
    print(df.shape)
    print("Data describe")
    print(df.describe())
    groups = df.drop_duplicates().sort_values(by=['time']).groupby('vehicleID')
    for group in groups:
        pass


def main():
    CSV_FILE_NAME = '../../koln.tr/data_process.csv'
    read_csv(CSV_FILE_NAME)


if __name__ == '__main__':
    main()