''''
多线程与多进程共享变量
改用多线程
'''''
from hmmlearn import hmm
import numpy as np
import multiprocessing as mp
import threading
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
x = np.array([])
x_len = np.array([])
train_file = TRAIN_DATA + '0' + '.txt'
chunk_lines = 5000


def read_distributed(*lines):
    print('thread is started')
    global x, x_len
    print(x)
    print(x_len)
    the_x_len = np.array([])
    the_x = np.array([])
    for line in lines:
        status = line.split()
        len_traj = len(status)
        if len_traj >= 10:
            x_status = np.array([])
            x_status = np.hstack((x_status, status))
            the_x = np.append(the_x,x_status)
            the_x_len = np.append(the_x_len, len_traj)
    x = np.append(x, the_x)
    x_len = np.append(x_len, the_x_len)
    print(x)
    print(x_len)


def main():
    line_cache = linecache.getlines(train_file)
    count = len(line_cache)
    number = int(count / chunk_lines)
    print(count)
    print(number)
    threads = []
    for i in range(1):
        threads.append(threading.Thread(target=read_distributed, args=line_cache[i * chunk_lines : i * chunk_lines + chunk_lines]))
    print(threads)
    for thread in threads:
        thread.setDaemon(True)
        thread.start()
    # print(x)
    # print(x_len)

    # pool = mp.Pool(processes=1)
    # jobs = []
    # for i in range(1):
    #     jobs.append(pool.apply_async(read_distributed, line_cache[i * chunk_lines : i * chunk_lines + chunk_lines]))
    # for job in jobs:
    #     job.get()
    # pool.close()

    # x = x[:, np.newaxis]
    # x = x.astype(np.float64)
    # print(x)
    # print(x_len)
    # number_of_status = 372
    # print('￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥')
    # print('Start Training')
    # model = hmm.GaussianHMM(n_components=number_of_status, n_iter=1, tol=0.001, covariance_type="tied")
    # model.fit(x, x_len)
    # print(model.score(x,x_len))
    # print('**************************************')
    # print(model.transmat_)
    # model_name = MODEL_PATH +'.pkl'
    # with open(model_name, 'wb') as model_file:
    #     pickle.dump(model, model_file)


if __name__ == '__main__':
    main()