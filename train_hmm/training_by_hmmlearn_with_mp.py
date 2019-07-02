''''
多进程
'''''
from hmmlearn import hmm
import numpy as np
import multiprocessing as mp
import linecache
import pickle
TRAIN_DATA = r'E:\NearXu\hmm_train_data\train_'
MODEL_PATH = r'E:\NearXu\model\model_'

"""
hmmlearn 有三种隐马尔可夫模型：
GaussianHMM：观测状态是连续状态，且符合高斯分布
GMMHMM：观测状态是连续状态，且符合混合高斯分布
MultinomialHMM：观测状态是离散的
"""
train_file = TRAIN_DATA + '0' + '.txt'
chunk_lines = 1000
be_big = 10000


def read_distributed(*lines):
    print('mp started')
    the_x = np.array([])
    the_x_len = np.array([])
    for line in lines:
        status = line.split()
        len_traj = len(status)
        if len_traj >= 10:
            x_status = np.array([])
            x_status = np.hstack((x_status, status))
            the_x = np.append(the_x,x_status)
            the_x_len = np.append(the_x_len, len_traj)
    return the_x, the_x_len


def main():
    x = np.array([])
    x_len = np.array([])

    line_cache = linecache.getlines(train_file)
    count = len(line_cache)
    number = int(count / chunk_lines)
    print(count)
    print(number)

    pool = mp.Pool(processes=10)
    jobs = []
    for i in range(10):
        jobs.append(pool.apply_async(read_distributed, line_cache[i * chunk_lines : i * chunk_lines + chunk_lines]))
    # jobs.append(pool.apply_async(read_distributed, line_cache[number * chunk_lines : count]))
    for job in jobs:
        x = np.append(x, job.get()[0])
        x_len = np.append(x_len, job.get()[1])
        print(x)
        print(len(x))
        print(x_len)
    pool.close()

    x = x[:, np.newaxis]
    x = x.astype(np.float64)
    x = x * be_big
    print(x)
    print(len(x))
    print(x_len)
    number_of_status = 100
    print('￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥')
    print('Start Training')
    model = hmm.GaussianHMM(n_components=number_of_status, n_iter=10, tol=0.001, covariance_type='diag', verbose=True)
    model.fit(x, x_len)
    # print(model.score(x,x_len))
    print('**************************************')
    print(model.transmat_)
    model_name = MODEL_PATH +'.pkl'
    with open(model_name, 'wb') as model_file:
        pickle.dump(model, model_file)


if __name__ == '__main__':
    main()