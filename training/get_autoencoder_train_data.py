import numpy as np
import multiprocessing as mp
TRAIN_DATA = r'E:\NearXu\train_data\train_'
AUTOENCODER_TRAIN_PATH_CSV = r'E:\NearXu\autoencoder2\train_'
AUTOENCODER_TRAIN_CSV = r'E:\NearXu\autoencoder2\train.csv'
AUTOENCODER_TEST_CSV = r'E:\NearXu\autoencoder2\test.csv'


global train_csv_file
train_csv_file = open(AUTOENCODER_TRAIN_CSV, 'a+', encoding='utf-8')
global test_csv_file
test_csv_file = open(AUTOENCODER_TEST_CSV, 'a+', encoding='utf-8')


def process(trace_id):
    #print(trace_id)
    with open(AUTOENCODER_TRAIN_PATH_CSV + str(int(trace_id)) + '.csv', 'a+', encoding='utf-8') as csv_file:
        with open(TRAIN_DATA + str(int(trace_id)) + '.txt', 'r+', encoding='utf-8') as file:
            for line in file:
                points = line.split(' ')
                print(len(points))
                random_number = np.random.randint(len(points))
                # print(random_number)
                # print(points[random_number])
                xy = points[random_number].split(',')
                # print(xy)
                # print(len(xy))
                if (len(xy) == 2):
                    x = xy[0]
                    y = xy[1]
                    try:
                        float(x)
                        float(y)
                        csv_str = str(x) + ',' + str(y) + '\n'
                        if csv_str.count(',') > 1:
                            print('count > 1')
                        else:
                            train_csv_file.writelines(csv_str)
                    except ValueError:
                        print("ValueError")


def main():
    pool = mp.Pool(processes=1)
    jobs = []
    for i in range(29):
        trace_id = []
        trace_id.append(i)
        jobs.append(pool.apply_async(process, trace_id))
    for job in jobs:
        job.get()
    pool.close()



if __name__ == '__main__':
    main()
