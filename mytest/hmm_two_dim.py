import pickle
import linecache
import numpy as np
from sklearn.utils import check_random_state
TRAIN_DATA = r'E:\NearXu\hmm_train_data\train_'
LE_STRING = 'le.pkl'
HMM_STRING = 'hmm.pkl'
MODEL_SETTING = 'statue_37_number_5000_'
LE_MODEL_PATH = r'E:\NearXu\model\model_' + MODEL_SETTING  + LE_STRING
HMM_MODEL_PATH = r'E:\NearXu\model\model_' + MODEL_SETTING  + HMM_STRING
TEST_FILE = r'E:\NearXu\test\test.txt'

np.set_printoptions(threshold=np.inf)

test_file = TRAIN_DATA + '2dim' + '.txt'

hmm_model_file = open(HMM_MODEL_PATH, 'rb')
hmm_model = pickle.load(hmm_model_file)

le_model_file = open(LE_MODEL_PATH, 'rb')
le_model = pickle.load(le_model_file)


with open(TEST_FILE, 'a+', encoding='utf-8') as file:
    file.write(np.array2string(hmm_model.transmat_))


line_cache = linecache.getlines(test_file)
line_number = 0
for line in line_cache:
    line_number += 1
    the_x = np.array([])
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
        x = the_x
        x = x[:, np.newaxis]
        new_x = le_model.transform(x)
        X = np.array(new_x).astype('int32')
        X = X.reshape(-1, 1)
        print(X)
        status_sequence = hmm_model.predict(X)
        print(status_sequence)
    if line_number >= 10:
        break

