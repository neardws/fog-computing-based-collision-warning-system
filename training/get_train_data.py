import pandas as pd
import multiprocessing as mp
CSV_FILE_PATH = r'E:\NearXu\trace\trace_'
TRAIN_DATA = r'E:\NearXu\train_data\train_'

# global train_file
# train_file = open(TRAIN_DATA, 'a+', encoding='utf-8')


def get_trace(num):
    print(num)
    with open(TRAIN_DATA + str(int(num)) + '.txt', 'a+', encoding='utf-8') as file:
        csv_file = CSV_FILE_PATH + str(num) + '.csv'
        df = pd.read_csv(csv_file, encoding='utf-8')
        trace_id = df['traceID'].drop_duplicates()
        for id in trace_id:
            # print(id)
            trace = df[df['traceID'] == id]
            x = trace['x']
            y = trace['y']
            line = ''
            # if id % 500 == 0:
            #     print(id / length)
            for i in range(len(trace)-1):
                add_x = x.values[i+1] - x.values[i]
                add_y = y.values[i+1] - y.values[i]
                if (int(add_x) < -30) | (int(add_x) > 30) | (int(add_y) < -30) | (int(add_y) > 30):
                    # print('break')
                    break
                else:
                    line = line + str(add_x) + ',' + str(add_y) + ' '
            file.writelines(line + '\n')
        file.close()


def main():
    # init objects
    pool = mp.Pool(processes=10)
    jobs = []
    for i in range(29):
        trace_id = []
        trace_id.append(i)
        jobs.append(pool.apply_async(get_trace, trace_id))
    for job in jobs:
        job.get()
    pool.close()


if __name__ == '__main__':
    main()