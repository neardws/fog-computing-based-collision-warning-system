''''
多进程
'''''
from sklearn import preprocessing
from hmmlearn import hmm
import numpy as np
import multiprocessing as mp
import linecache
import pickle
import time
TRAIN_DATA = r'E:\NearXu\hmm_train_data\train_'
MODEL_PATH = r'E:\NearXu\model\model_statue_372_number_10000_'

"""
hmmlearn 有三种隐马尔可夫模型：
GaussianHMM：观测状态是连续状态，且符合高斯分布
GMMHMM：观测状态是连续状态，且符合混合高斯分布
MultinomialHMM：观测状态是离散的
"""
train_file = TRAIN_DATA + '2dim' + '.txt'
chunk_lines = 1000


def read_distributed(*lines):
    print('mp started')
    the_x = np.array([])
    the_x_len = np.array([])
    line_number = len(lines)
    print(line_number)
    now_line_number = 0
    for line in lines:
        now_line_number += 1
        status = line.split()
        status_num = 0
        for xys in status:
            xy = xys.split(',')
            if len(xy) == 2:
                status_num += 1
                sta = np.array(xy).astype('float32').astype('int32')
                new_sta = int(sta[0]) * 61 + int(sta[1])
                the_x = np.hstack((the_x, new_sta))
        len_traj = status_num
        if len_traj == 0:
            pass
        else:
            the_x_len = np.append(the_x_len, len_traj)
        if now_line_number % 100 == 0:
            print('Processed '+ str(now_line_number / line_number * 100) + ' %')
    return the_x, the_x_len


def main():
    le = preprocessing.LabelEncoder()
    x = np.array([])
    x_len = np.array([])

    line_cache = linecache.getlines(train_file)
    count = len(line_cache)
    number = int(count / chunk_lines)
    print(count)
    print(number)

    t()
    pool = mp.Pool(processes=10)
    jobs = []
    for i in range(10):
        jobs.append(pool.apply_async(read_distributed, line_cache[i * chunk_lines : i * chunk_lines + chunk_lines]))
    # jobs.append(pool.apply_async(read_distributed, line_cache[number * chunk_lines : count]))
    for job in jobs:
        x = np.append(x, job.get()[0])
        x_len = np.append(x_len, job.get()[1])
    pool.close()

    labels = []
    for number in x:
        if number in labels:
            pass
        else:
            labels.append(number)

    # print(labels)
    le.fit(labels)

    print('**************************************')
    t()
    print(le.classes_)
    model_le_name = MODEL_PATH + 'le.pkl'
    with open(model_le_name, 'wb') as model_file:
        pickle.dump(le, model_file)
    print("le saved")

    x = x[:, np.newaxis]

    new_x = le.transform(x)
    X = np.array(new_x).astype('int32')
    # X = X[:, np.newaxis]
    X = X.reshape(-1, 1)
    # print(X.shape)
    # print(X.dtype)
    #
    print(X)
    print(len(X))
    #
    # print(x_len.shape)
    # print(x_len.dtype)
    X_len = np.array(x_len).astype('int32')

    # print(X_len.shape)
    # print(X_len.dtype)
    print(sum(X_len))

    number_of_status = 372
    print('￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥')
    t()
    print('Start Training')
    model = hmm.MultinomialHMM(n_components=number_of_status, n_iter=10000, tol=0.01, verbose=True)
    model.fit(X, X_len)
    # print(model.score(x,x_len))
    print('**************************************')
    print(model.transmat_)
    model_name = MODEL_PATH +'hmm.pkl'
    with open(model_name, 'wb') as model_file:
        pickle.dump(model, model_file)
    print("hmm saved")


def t():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


if __name__ == '__main__':
    main()