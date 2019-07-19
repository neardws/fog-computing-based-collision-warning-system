import pickle
import linecache
import numpy as np
from sklearn.utils import check_random_state

class hmm_model:
    def __init__(self, type, le_model, hmm_model):
        self.type = type
        self.le_model = le_model
        self.hmm_model = hmm_model
        self.origin_trace = None
        self.prediction_trace = None

    def set_origin_trace(self, trace):
        self.origin_trace = trace

    def get_prediction_trace(self):
        prediction_trace = None
        return prediction_trace

    def predict(self, trace):
        X = self.process_state(trace)
        print(X)
        status_sequence = hmm_model.predict(X)
        transmat_cdf = np.cumsum(hmm_model.transmat_, axis=1)
        random_state = check_random_state(hmm_model.random_state)
        next_state = (transmat_cdf[status_sequence[-1]] > random_state.rand()).argmax()
        next_obs1 = hmm_model._generate_sample_from_state(next_state, random_state)
        # emission_cdf = np.cumsum(hmm_model.emissionprob_, axis=1)
        # next_obs2 = (emission_cdf[next_state] > random_state.rand()).argmax()
        xy_increment = self.get_origin_xy_increment(next_obs1)
        origin_xy = self.get_origin_xy(trace)
        prediction_x = origin_xy[0] + xy_increment[0]
        prediction_y = origin_xy[1] + xy_increment[1]
        # print('*' * 64)
        # print(prediction_x)
        # print(prediction_y)
        return prediction_x, prediction_y

    def process_state(self, trace):
        status = []
        for i in range(len(trace) - 1):
            x_add = trace[i+1].location_x - trace[i].location_x
            y_add = trace[i+1].location_y - trace[i].location_y
            status.append([x_add, y_add])
        X = None
        the_x = np.array([])
        if type == 'discrete':
            status_num = 0
            for xys in status:
                status_num += 1
                sta_x = np.array(xys[0]).astype('float32').astype('int32')
                sta_y = np.array(xys[1]).astype('float32').astype('int32')
                new_sta = int(sta_x) * 61 + int(sta_y)
                the_x = np.hstack((the_x, new_sta))
            len_traj = status_num
            if len_traj == 0:
                pass
            else:
                x = the_x
                x = x[:, np.newaxis]
                new_x = self.le_model.transform(x)
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

    def get_origin_xy_increment(self, obs):
        x_increment = None
        y_increment = None
        if type == 'discrete':
            origin_obs = self.le_model.inverse_transform(obs)
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

    def get_origin_xy(self, trace):
        origin_x = trace[-1].location_x
        origin_y = trace[-1].location_y
        return origin_x, origin_y
