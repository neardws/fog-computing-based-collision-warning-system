import numpy as np
import pandas as pd


def read_csv(fliename):
    chunksize = 6 ** 10
    for chunk in pd.read_csv(fliename, error_bad_lines=False, chunksize=chunksize):
        print("Data example:")
        print(chunk.head(10))
        print("Data shape:")
        print(chunk.shape)
        print("Data describe")
        print(chunk.describe())
        groups = chunk.drop_duplicates().sort_values(by=['time']).groupby('vehicleID')
        i = 1
        for group in groups:
            if i <= 1:
             with open('../../koln.tr/data_process.csv', 'a+') as f:
                 f.write(group)


def main():
    CSV_FILE_NAME = '../../koln.tr/withspeed.csv'
    read_csv(CSV_FILE_NAME)


if __name__ == '__main__':
    main()