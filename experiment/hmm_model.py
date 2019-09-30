import numpy as np
from vehicle import vehicle
from sklearn.utils import check_random_state

'''
HMM
used: my_model = hmm_model(type='discrete'. le_model=le_model, hmm_model=hmm_model)
      my_model.set_origin_trace(vehicles)
      my_model.set_prediction_seconds(10)
      prediction_trace = my_model.get_prediction_trace()
'''
class hmm_model:
    def __init__(self, type, le_model, hmm_model, packet_loss_rate):
        self.type = type
        self.le_model = le_model
        self.hmm_model = hmm_model
        self.packet_loss_rate = packet_loss_rate
        self.origin_trace = None
        self.prediction_trace = None
        self.prediction_seconds = None

    '''
    Input
    '''
    def set_origin_trace(self, trace):
        self.origin_trace = trace

    def set_prediction_seconds(self, seconds):
        self.prediction_seconds = seconds

    '''
    Output
    get prediction trace in the prediction time 
    '''
    def get_prediction_trace(self, saver):
        prediction_trace = None
        trace = self.origin_trace
        origin_time = trace[-1].time
        origin_id = trace[-1].vehicleID
        for i in range(self.prediction_seconds):
            prediction_location = self.predict(trace, saver)
            if prediction_location is not None:
                v = vehicle(self.packet_loss_rate)
                v.set_vehicleID(origin_id)
                v.set_location(prediction_location)
                v.set_time(origin_time + i + 1)
                trace.append(v)
            else:
                # print("prediction location is none in HMM_MODEL get prediction trace")
                break
        prediction_trace = trace
        return prediction_trace

    '''
    get the prediction location according to the given trace
    '''
    def predict(self, trace, saver):
        X = self.process_state(trace, saver)
        if X is not None:
            # print(X)
            # print("X is not None")
            status_sequence = self.hmm_model.predict(X)
            saver.write("status_sequence")
            saver.write(str(status_sequence))
            transmat_cdf = np.cumsum(self.hmm_model.transmat_, axis=1)
            random_state = check_random_state(self.hmm_model.random_state)
            next_state = (transmat_cdf[status_sequence[-1]] > random_state.rand()).argmax()
            next_obs1 = self.hmm_model._generate_sample_from_state(next_state, random_state)
            xy_increment = self.get_origin_xy_increment(next_obs1, saver)
            prediction_x = trace[-1].location_x + xy_increment[0]
            prediction_y = trace[-1].location_y + xy_increment[1]

            saver.write("prediction_x")
            saver.write(str(prediction_x))
            saver.write("prediction_y")
            saver.write(str(prediction_y))
            # print('*' * 64)
            # print(prediction_x)
            # print(prediction_y)
            prediction_xy = []
            prediction_xy.append(prediction_x)
            prediction_xy.append(prediction_y)
            return prediction_xy
        else:
            # print("X is none in HMM MODEL predict")
            return None

    '''
    process the trace into status, which contains x and y increment
    '''
    def process_state(self, trace, saver):
        status = []
        X = None
        for i in range(len(trace) - 1):
            # print(i)
            # print(trace[i + 1].location_x)
            # print(trace[i].location_x)
            x_add = trace[i + 1].location_x - trace[i].location_x
            y_add = trace[i + 1].location_y - trace[i].location_y
            status.append([int(x_add), int(y_add)])

        saver.write("()"*64)
        saver.write("status")
        saver.write(str(status))

        the_x = np.array([])
        # print('status is')
        # print(status)
        if self.type == 'discrete':
            # print('TYPE MATCH')
            status_num = 0
            for xys in status:
                status_num += 1
                new_sta = xys[0] * 61 + xys[1]
                the_x = np.hstack((the_x, new_sta))
            len_traj = status_num
            # print("The x is")
            # print(the_x)
            if len_traj == 0:
                pass
            else:
                x = the_x
                x = x[:, np.newaxis]
                saver.write("x")
                saver.write(str(x))
                try:
                    new_x = self.le_model.transform(x)
                except ValueError:
                    # print("Value Error")
                    return None
                X = np.array(new_x).astype('int32')
                X = X.reshape(-1, 1)
                saver.write("X")
                saver.write(str(X))
        elif self.type == 'continuous':
            '''
            TODO
            '''
            print('Not defined')
        else:
            print('Type Error')
        # print(X)
        return X

    '''
    get the origin x, y increment by observation
    '''
    def get_origin_xy_increment(self, obs, saver):
        x_increment = None
        y_increment = None
        if self.type == 'discrete':
            origin_obs = self.le_model.inverse_transform(obs)
            saver.write("origin obs")
            saver.write(str(origin_obs))
            y_increment = origin_obs % 61
            x_increment = (origin_obs - y_increment) / 61
            saver.write("y_increment")
            saver.write(str(y_increment))
            saver.write("x_increment")
            saver.write(str(x_increment))
            if x_increment < -30:
                x_increment += 61
                y_increment -= 1
            if x_increment > 30:
                x_increment -= 61
                y_increment += 1
            if y_increment > 30:
                x_increment += 1
                y_increment -= 61
            if y_increment < -30:
                x_increment -= 1
                y_increment += 61
            saver.write("y_increment")
            saver.write(str(y_increment))
            saver.write("x_increment")
            saver.write(str(x_increment))
        elif self.type == 'continuous':
            '''
            TODO
            '''
            print('Not defined.')
        else:
            print('Type error')
        return x_increment, y_increment
