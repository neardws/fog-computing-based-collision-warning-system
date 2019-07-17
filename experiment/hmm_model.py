import pickle
import linecache
import numpy as np
from sklearn.utils import check_random_state


def predict(type, line, le_model, hmm_model):
    X = process_state(type, line, le_model)
    print(X)
    status_sequence = hmm_model.predict(X)
    transmat_cdf = np.cumsum(hmm_model.transmat_, axis=1)
    random_state = check_random_state(hmm_model.random_state)
    next_state = (transmat_cdf[status_sequence[-1]] > random_state.rand()).argmax()
    next_obs1 = hmm_model._generate_sample_from_state(next_state, random_state)
    emission_cdf = np.cumsum(hmm_model.emissionprob_, axis=1)
    next_obs2 = (emission_cdf[next_state] > random_state.rand()).argmax()
    xy_increment = get_origin_xy_increment(type=type, obs= next_obs1)
    origin_xy = get_origin_xy(line)
    prediction_x = origin_xy[0] + xy_increment[0]
    prediction_y = origin_xy[1] + xy_increment[1]
    print('*' * 64)
    print(prediction_x)
    print(prediction_y)


def process_state(type, line, le_model):
    X = None
    the_x = np.array([])
    if type == 'discrete':
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
    elif type == 'continuous':
        '''
        TODO
        '''
        print('Not defined')
    else:
        print('Type Error')
    return X


def get_origin_xy_increment(type, obs, le_model):
    x_increment = None
    y_increment = None
    if type == 'discrete':
        origin_obs = le_model.inverse_transform(obs)
        y_increment = origin_obs % 61
        x_increment = (origin_obs - y_increment) / 61
        if x_increment < -30:
            x_increment += 61
            y_increment -= 1
        if x_increment > 30:
            x_increment -= 61
            y_increment += 1
    elif type == 'continuous':
        '''
        TODO
        '''
        print('Not defined.')
    else:
        print('Type error')
    return x_increment, y_increment


def get_origin_xy(line):
    status = line.split()
    last_status = status[-1]
    origin_xy = last_status.split(',')
    return origin_xy[0], origin_xy[1]