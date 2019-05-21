from hmmlearn import hmm
import numpy as np
import pickle
TRAIN_DATA = r'E:\NearXu\train\train_'
MODEL_PATH = r'E:\NearXu\model\model_'


def main():
    x_len = []
    x = np.array([])
    train_file = TRAIN_DATA + '0' + '.txt'
    with open(train_file, 'r+', encoding='utf-8') as file:
        for line in file:
            status = line.split()
            len_traj = len(status)
            x_len.append(len_traj)
            x_status = np.array([])
            for sta in status:
                x_status = np.hstack((x_status, [sta]))
            x_status = x_status[:, np.newaxis]
            x = np.append(x, x_status)
        x = x[:, np.newaxis]
        print(x)
        print(x_len)
        number_of_status = 67*67
        model = hmm.GaussianHMM(n_components=number_of_status, n_iter=1000, tol=0.01, covariance_type="full")
        model.fit(x, x_len)
        print('**************************************')
        print(model.transmat_)
        model_name = MODEL_PATH + '0' +'.pkl'
        with open(model_name, 'wb') as model_file:
            pickle.dump(model, model_file)

if __name__ == '__main__':
    main()