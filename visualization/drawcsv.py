import pandas as pd
import matplotlib.pyplot as plot
import multiprocessing as mp

CSV_FILE = r'E:\NearXu\trace\trace_0.csv'
X_MIN = 11000
X_MAX = 13000
Y_MIN = 11000
Y_MAX = 13000

#
# def process_wrapper(chunk, chunk_num):
#     colors = ['r', 'g', 'c', 'b', 'y', 'k']
#     trace_id = chunk['traceID'].drop_duplicates()
#     for id in trace_id:
#         trace = chunk[(chunk['traceID'] == id) & (chunk['x'] >= 10000) & (chunk['x'] <= 16000) & (chunk['y'] >= 10000) & (chunk['y'] <= 20000)]
#         if len(trace):
#             x = trace['x']
#             y = trace['y']
#             plot.scatter(x, y, 0.1, colors[id % 6])
#             print(id)


def main():
    df = pd.read_csv(CSV_FILE)
    x_max = df['x'].max()
    y_max = df['y'].max()
    plot.xlim(0, x_max)
    plot.ylim(0, y_max)
    # pool = mp.Pool(processes=20)
    # jobs = []

    chunk_size = 100000

    chunk_num = 1
    for chunk in pd.read_csv(CSV_FILE, error_bad_lines=False, chunksize=chunk_size):
        colors = ['r', 'g', 'c', 'b', 'y', 'k']
        trace_id = chunk['traceID'].drop_duplicates()
        for id in trace_id:
            trace = chunk[
                (chunk['traceID'] == id)]
            if len(trace):
                x = trace['x']
                y = trace['y']
                plot.scatter(x, y, 0.1, colors[id % 6])
                print(id)

    # wait for all jobs to finish
    # for job in jobs:
    #     job.get()
    plot.show()
    # clean up
    # pool.close()


if __name__ == '__main__':
    main()