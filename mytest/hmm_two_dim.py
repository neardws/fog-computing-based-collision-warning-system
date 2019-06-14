import pickle
import linecache
import numpy as np
from sklearn.utils import check_random_state
TRAIN_DATA = r'E:\NearXu\hmm_train_data\train_'
ADD_STRING = '2dim.pkl'
MODEL_PATH = r'E:\NearXu\model\model_' + ADD_STRING
TEST_FILE = r'E:\NearXu\test\test.txt'

np.set_printoptions(threshold=np.inf)

model_file = open(MODEL_PATH, 'rb')
test_file = TRAIN_DATA + '2dim' + '.txt'
model = pickle.load(model_file)

# print(model.get_stationary_distribution())
# print(len(model.get_stationary_distribution()))
# print(model.transmat_)

with open(TEST_FILE, 'a+', encoding='utf-8') as file:
    file.write(np.array2string(model.transmat_))

# print('HMM Score')

line_cache = linecache.getlines(test_file)

number = 0
for line in line_cache:
    number += 1
    x_status = np.array([])
    first_status = 0
    if number <= 100:
        status = line.split()
        print(status)
        for xys in status:
            xy = xys.split(',')
            if len(xy) == 2:
                if first_status == 0:
                    x_status = np.hstack((x_status, xy))
                    x_status = np.expand_dims(x_status, axis=0)
                    first_status += 1
                else:
                    x_status = np.vstack((x_status, xy))
        len_traj = len(x_status)


        print(x_status.shape)
        x_status = x_status.astype('float32')
        x_status = (x_status + 30.0) / 60.
        # print(x_status)
        status_sequence = model.predict(x_status)
        print(status_sequence)





        # x_status = np.array([])
        # status = line.split()
        # if len(status) >= 10:
        #     len_traj = 0
        #     #for sta in status:
        #     #    len_traj += 1
        #         # if len_traj <= 10:
        #     x_status = np.hstack((x_status, status))
        #     the_x = np.append(the_x, x_status)
        #     the_x = the_x[:, np.newaxis]
        #     the_x = the_x.astype(np.float64)
        #     # print(the_x)
        #     status_sequence = model.predict(the_x)
        #     print(status_sequence)
            # status_sequence = model.decode(the_x)
            # print(status_sequence)
            # print('HMM Score')
            # print(model.score(the_x))
            # print(the_x)

            # transmat_cdf = np.cumsum(model.transmat_, axis=1)
            # random_state = check_random_state(model.random_state)
            # next_state = (transmat_cdf[status_sequence[-1]] > random_state.rand()).argmax()
            # next_obs = model._generate_sample_from_state(status_sequence[-1], next_state)
            # print(next_state)
            # print(next_obs)

            # status_sequence = model.predict(the_x)
            # print(status_sequence)
            # transmat_cdf = np.cumsum(model.transmat_, axis=1)
            # random_state = check_random_state(model.random_state)
            # next_state = (transmat_cdf[status_sequence[-1]] > random_state.rand()).argmax()
            # next_obs = model._generate_sample_from_state(status_sequence[-1], next_state)
            # print(next_state)
            # print(next_obs)
