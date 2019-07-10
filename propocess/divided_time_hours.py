import numpy as np
import pandas as pd
import multiprocessing as mp

TRACE_FILE = r'E:\NearXu\trace\trace_'

seconds = 3600

def process_wrapper(i, trace_file):
    chunk_size = 1000000
    for chunk in pd.read_csv(trace_file, error_bad_lines=False, chunksize=chunk_size):
        print('Max')
        print(chunk['time'].max())
        print('Min')
        print(chunk['time'].min())


    # with open(TRACE_FILE + '_' + str(chunk_num) + '.csv', 'a+', encoding='utf-8') as f:
    #     f.write('traceID,time,x,y')
    #     f.write('\n')
    #     vehicleID = chunk['vehicleID'].drop_duplicates()
    #     print(len(vehicleID))
    #     for id in vehicleID:
    #         trace = chunk[chunk['vehicleID'] == id].sort_values(by=['time'])
    #         if len(trace) >= 60:
    #             x = trace['x_coordinates']
    #             y = trace['y_coordinates']
    #             time = trace['time']
    #             for i in range(len(trace)):
    #                 if i + 1 < len(trace): # is not over bound
    #                     if time.values[i+1] == time.values[i] + 1:
    #                         f.write(str(trace_num) + ',' + str(time.values[i]) + ',' + str(x.values[i]) + ',' + str(y.values[i]))
    #                         f.write('\n')
    #                     elif time.values[i+1] - time.values[i] >= 30:
    #                         trace_num += 1
    #                         # f.write(str(trace_num) + ',' + str(time.values[i]) + ',' + str(x.values[i]) + ',' + str(y.values[i]))
    #                         # f.write('\n')
    #                     else:
    #                         pass
    #             trace_num += 1
    #     f.close()

def main():
    # init objects
    pool = mp.Pool(processes=1)
    jobs = []

    for i in range(29):
        trace_file = TRACE_FILE + str(i) + '.csv'
        jobs.append(pool.apply_async(process_wrapper, (i, trace_file)))

    # wait for all jobs to finish
    for job in jobs:
        job.get()

    # clean up
    pool.close()


if __name__ == '__main__':
    main()
