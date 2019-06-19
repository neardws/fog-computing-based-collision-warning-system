''''
多进程
'''''
from hmmlearn import hmm
import numpy as np
import multiprocessing as mp
import linecache
import pickle
TRAIN_DATA = r'E:\NearXu\hmm_train_data\train_'
MODEL_PATH = r'E:\NearXu\model\model_2dim'

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
    first_line = 0
    line_number = len(lines)
    print(line_number)
    now_line_number = 0
    for line in lines:
        now_line_number += 1
        if first_line == 0:
            status = line.split()
            x_status = np.array([])
            first_status = 0
            for xys in status:
                xy = xys.split(',')
                if len(xy) == 2:
                    if first_status == 0:
                        x_status = np.hstack((x_status, xy))
                        x_status = np.expand_dims(x_status, axis=0)
                        the_x = np.hstack((the_x, xy))
                        the_x = np.expand_dims(the_x, axis=0)
                        first_status += 1
                    else:
                        x_status = np.vstack((x_status, xy))
                        the_x = np.vstack((the_x, xy))
            len_traj = len(x_status)
            if len_traj == 0:
                pass
            else:
                the_x_len = np.append(the_x_len, len_traj)
            first_line += 1
        else:
            status = line.split()
            x_status = np.array([])
            first_status = 0
            for xys in status:
                xy = xys.split(',')
                if len(xy) == 2:
                    if first_status == 0:
                        x_status = np.hstack((x_status, xy))
                        x_status = np.expand_dims(x_status, axis=0)
                        the_x = np.vstack((the_x, xy))
                        first_status += 1
                    else:
                        x_status = np.vstack((x_status, xy))
                        the_x = np.vstack((the_x, xy))
            len_traj = len(x_status)
            if len_traj == 0:
                pass
            else:
                the_x_len = np.append(the_x_len, len_traj)
        if now_line_number % 100 == 0:
            print('Processed '+ str(now_line_number / line_number * 100) + ' %')
    return the_x, the_x_len


def main():
    line_cache = linecache.getlines(train_file)
    count = len(line_cache)
    number = int(count / chunk_lines)
    print(count)
    print(number)

    pool = mp.Pool(processes=20)
    jobs = []
    for i in range(20):
        jobs.append(pool.apply_async(read_distributed, line_cache[i * chunk_lines : i * chunk_lines + chunk_lines]))
    # jobs.append(pool.apply_async(read_distributed, line_cache[number * chunk_lines : count]))'
    first = 0
    X = np.array([])
    X_len = np.array([])
    for job in jobs:
        if first == 0:
            X = np.array(job.get()[0])
            X_len = np.append(X_len, job.get()[1])
            first += 1
        else:
            X = np.vstack((X, job.get()[0]))
            X_len = np.append(X_len, job.get()[1])
    pool.close()
    X = X.astype('float32')
    X = (X + 30.0) / 60.
    print("X IS\n")
    print(X)
    print("X_len is\n")
    print(X_len)
    print(len(X))
    print(sum(X_len))

    number_of_status = 100
    print('￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥')
    print('Start Training')
    model = hmm.GaussianHMM(n_components=number_of_status, n_iter=10, tol=0.001, covariance_type='tied')
    model.fit(X, X_len)
    # print(model.score(x,x_len))
    print('**************************************')
    print(model.transmat_)
    model_name = MODEL_PATH +'.pkl'
    with open(model_name, 'wb') as model_file:
        pickle.dump(model, model_file)


def test():
    x_status = np.array([])
    x_status = np.hstack((x_status, np.array([1,2])))
    x_status = np.expand_dims(x_status, axis=0)
    x_status = np.vstack((x_status, np.array([1,2])))
    print(x_status)
    the_x = np.array([])
    the_x_len = np.array([])
    the_x = np.hstack((the_x, x_status))
    print(the_x)

if __name__ == '__main__':
    main()