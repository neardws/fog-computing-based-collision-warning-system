import numpy as np
import pandas as pd
import multiprocessing as mp

CSV_FILE_NAME = r'E:\NearXu\withspeed.csv'
TRACE_FILE = r'E:\NearXu\trace\trace'


# def read_csv(fliename):
#     chunksize = 10 ** 10
#     trace_num = 1
#     for chunk in pd.read_csv(fliename, error_bad_lines=False, chunksize=chunksize):
#         # print("Data example:")
#         # print(chunk.head(10))
#         # print("Data shape:")
#         # print(chunk.shape)
#         # print("Data describe")
#         # print(chunk.describe())
#         # groups = chunk.drop_duplicates().sort_values(by=['time']).groupby('vehicleID')
#         # i = 1
#         # for group in groups:
#         #     if i <= 1:
#         #         print(len(groups))
#         #         print(group)
#         #         print(group[0])
#         #         i += 1
#         #     else:
#         #         break
#              # with open('../../koln.tr/data_process.csv', 'a+') as f:
#              #     f.write(group)
#         vehicleID = chunk['vehicleID'].drop_duplicates()
#         print(len(vehicleID))
#         for id in vehicleID:
#             trace = chunk[chunk['vehicleID'] == id].sort_values(by=['time'])
#             if len(trace) >= 180:
#                 x = trace['x_coordinates']
#                 y = trace['y_coordinates']
#                 time = trace['time']
#                 for i in range(len(trace)):
#                     f.write(
#                         str(trace_num) + ',' + str(time.values[i]) + ',' + str(x.values[i]) + ',' + str(y.values[i]))
#                     f.write('\n')
#                 trace_num += 1


def process_wrapper(chunk, chunk_num):
    trace_num = 1
    with open(TRACE_FILE + '_' + str(chunk_num) + '.csv', 'a+', encoding='utf-8') as f:
        f.write('traceID,time,x,y')
        f.write('\n')
        vehicleID = chunk['vehicleID'].drop_duplicates()
        print(len(vehicleID))
        for id in vehicleID:
            trace = chunk[chunk['vehicleID'] == id].sort_values(by=['time'])
            if len(trace) >= 60:
                x = trace['x_coordinates']
                y = trace['y_coordinates']
                time = trace['time']
                for i in range(len(trace)):
                    if i + 1 < len(trace): # is not over bound
                        if time.values[i+1] == time.values[i] + 1:
                            f.write(str(trace_num) + ',' + str(time.values[i]) + ',' + str(x.values[i]) + ',' + str(y.values[i]))
                            f.write('\n')
                        elif time.values[i+1] - time.values[i] >= 30:
                            trace_num += 1
                            # f.write(str(trace_num) + ',' + str(time.values[i]) + ',' + str(x.values[i]) + ',' + str(y.values[i]))
                            # f.write('\n')
                        else:
                            pass
                trace_num += 1
        f.close()

def main():
    # init objects
    pool = mp.Pool(processes=20)
    jobs = []

    chunk_size = 5 ** 10
    # loop = True
    # chunks = []
    # reader = pd.read_csv(CSV_FILE_NAME, iterator=True)
    # while loop:
    #     try:
    #         chunk = reader.get_chunk(chunk_size)
    #         chunks.append(chunk)
    #     except StopIteration:
    #         loop = False
    #         print('Iteration is stopped.')
    # df = pd.concat(chunks, ignore_index=True)
    # print(df.isnull())

    chunk_num = 0
    for chunk in pd.read_csv(CSV_FILE_NAME, error_bad_lines=False, chunksize=chunk_size):
        jobs.append(pool.apply_async(process_wrapper, (chunk, chunk_num)))
        chunk_num += 1

    # wait for all jobs to finish
    for job in jobs:
        job.get()

    # clean up
    pool.close()


if __name__ == '__main__':
    main()