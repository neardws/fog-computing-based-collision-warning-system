import pandas as pd
import multiprocessing as mp
import random
TRAIN_DATA = r'E:\NearXu\train_data\train_'
RANDOM_TRAIN_SET = r'E:\NearXu\random_set\train.txt'
RANDOM_TEST_SET = r'E:\NearXu\random_set\test.txt'

global random_train_file
global random_test_file
random_train_file = open(RANDOM_TRAIN_SET, 'a+', encoding='utf-8')
random_test_file = open(RANDOM_TEST_SET, 'a+', encoding='utf-8')


def get_trace(num):
    print(num)
    with open(TRAIN_DATA + str(int(num)) + '.txt', 'r+', encoding='utf-8') as file:
        lines = file.readlines()
        line_number = len(lines)
        # print(lines)
        print(line_number)
        random_list = range(0, line_number)
        random_train_id = random.sample(random_list, int(0.3 * line_number))
        random_test_id = random.sample(random_list, int(0.1 * line_number))
        print(random_train_id)
        for id in random_train_id:
            random_train_file.writelines(lines[id])
        for id in random_test_id:
            random_test_file.writelines(lines[id])


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