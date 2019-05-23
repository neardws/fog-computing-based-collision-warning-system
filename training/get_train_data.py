import pandas as pd
import multiprocessing as mp
CSV_FILE_PATH = r'E:\NearXu\trace\trace_'
TRAIN_DATA = r'E:\NearXu\train_data\train_'


def get_trace(num):
    print(num)
    with open(TRAIN_DATA + str(int(num)) + '.txt', 'a+', encoding='utf-8') as file:
        csv_file = CSV_FILE_PATH + str(num) + '.csv'
        df = pd.read_csv(csv_file, encoding='utf-8')
        trace_id = df['traceID']
        for id in range(len(trace_id)):
            print(id+1)
            trace = df[df['traceID'] == (id + 1)]
            x = trace['x']
            y = trace['y']
            for i in range(len(trace)-1):
                add_x = x.values[i+1] - x.values[i]
                add_y = y.values[i+1] - y.values[i]
                print(str(add_x) + ' ' + str(add_y))
                file.write(str(add_x) + ',' + str(add_y) + ' ')
            file.write('\n')


def main():
    # init objects
    pool = mp.Pool(processes=20)
    jobs = []
    for i in range(20,29):
        trace_id = []
        trace_id.append(i)
        jobs.append(pool.apply_async(get_trace, trace_id))
    for job in jobs:
        job.get()
    pool.close()


if __name__ == '__main__':
    main()