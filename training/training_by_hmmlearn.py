from hmmlearn import hmm
import numpy as np
import pickle
TRAIN_DATA = r'E:\NearXu\train\train_'
MODEL_PATH = r'E:\NearXu\model\model_'

"""
hmmlearn 有三种隐马尔可夫模型：
GaussianHMM：观测状态是连续状态，且符合高斯分布
GMMHMM：观测状态是连续状态，且符合混合高斯分布
MultinomialHMM：观测状态是离散的
"""
def main():
    x_len = np.array([])
    x = np.array([])
    train_file = TRAIN_DATA + '0' + '.txt'
    count = len(open(train_file, 'r').readlines())
    print(count)
    with open(train_file, 'r+', encoding='utf-8') as file:
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        line_num = 0
        for line in file:
            line_num += 1
            print(line_num)
            status = line.split()
            # print(status)
            len_traj = len(status)
            # print(len_traj)
            x_len = np.append(x_len, len_traj)
            x_status = np.array([])
            for sta in status:
                x_status = np.hstack((x_status, sta))
                # print(x_status)
            # x_status = x_status[:, np.newaxis]
            print(x_status)
            x = np.append(x, x_status)
            print(x)
            # if line_num <= 500:
            #
            # else:
            #     break
        x = x[:, np.newaxis]
        x = x.astype(np.float64)
        x_len = x_len.astype(np.float64)
        print(x)
        print(x_len)
        number_of_status = 67*67
        print('￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥￥')
        print('Start Training')
        model = hmm.GaussianHMM(n_iter=5, tol=0.01, covariance_type="full")
        model.fit(x, x_len)
        print('**************************************')
        print(model.transmat_)
        model_name = MODEL_PATH + '0' +'.pkl'
        with open(model_name, 'wb') as model_file:
            pickle.dump(model, model_file)

if __name__ == '__main__':
    main()