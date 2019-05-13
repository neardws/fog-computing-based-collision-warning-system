import numpy as np
import pandas as pd

CSV_FILE_NAME = '/mnt/d/koln/withspeed/withspeed.csv'
TRACE_FILE = '/mnt/d/koln/trace.csv'

f = open(TRACE_FILE, 'a+', encoding='utf-8')
f.write('traceID,time,x,y')
f.write('\n')


def read_csv(fliename):
    chunksize = 5 ** 10
    trace_num = 1
    for chunk in pd.read_csv(fliename, error_bad_lines=False, chunksize=chunksize):
        # print("Data example:")
        # print(chunk.head(10))
        # print("Data shape:")
        # print(chunk.shape)
        # print("Data describe")
        # print(chunk.describe())
        # groups = chunk.drop_duplicates().sort_values(by=['time']).groupby('vehicleID')
        # i = 1
        # for group in groups:
        #     if i <= 1:
        #         print(len(groups))
        #         print(group)
        #         print(group[0])
        #         i += 1
        #     else:
        #         break
             # with open('../../koln.tr/data_process.csv', 'a+') as f:
             #     f.write(group)
        i = 1
        if i <= 1:
            vehicleID = chunk['vehicleID'].drop_duplicates()
            print(len(vehicleID))
            i += 1
            for id in vehicleID:
                trace = chunk[chunk['vehicleID'] == id].sort_values(by=['time'])
                if len(trace) >= 180:
                    x = trace['x_coordinates']
                    y = trace['y_coordinates']
                    time = trace['time']
                    for i in range(len(trace)):
                        f.write(str(trace_num) + ',' + str(time.values[i]) + ',' + str(x.values[i]) + ',' + str(y.values[i]))
                        f.write('\n')
                    trace_num += 1


def main():

    read_csv(CSV_FILE_NAME)


if __name__ == '__main__':
    main()