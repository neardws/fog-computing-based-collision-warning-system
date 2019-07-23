from data_ready import data_ready
from fog_node import fog_node
from hmm_model import hmm_model
import pickle
import time
import warnings
warnings.filterwarnings('ignore')
'''
experiment setup
some parameters are listed as below
    start_time: experiment start time, like '9am', means experiment starts in 9 AM, range from 1AM to 11PM
    during_time: how long the experiment last, like 600, means 10min
    scenario: choose different intersection to do experiment, like '1', means scenario one, range from 1 to 9
              which is also marked the location of fog node
    scenario_range: different range of selected intersection, like 500
                    which is also 
    packet_loss_rate: packet loss rate in packet transmission, like 3.00, means 3.00% packet will lost
    collision_distance: the distance to judge collision, like 5, means 5 meters
    hmm_type: the hmm model type, like 'discrete' or 'continuous', means MultinomialHMM or GaussianHMM
    hmm_model: the hmm model which is predicted the vehicle trace, from the model file, like hmm_model = pickle.load(hmm_model_file)
    le_model: if the hmm model type is discrete, then le model should be given, from the model file, like le_model = pickle.load(le_model_file)
    prediction_time: how long the prediction
    headway: the threshold of collision happen condition 

Two evaluating indicators:
    precision: TP/(TP+FP)
    recall: TP/(TP+FN)
where TP is True Positive, FP is False Positive, FN is False Negative
True/False means the classify is right or wrong
Positive/Negative means the classify category 
    
Three difference changeable parameters:
    scenario
    packet_loss_rate
    headway
'''
class experiment:
    def __init__(self, start_time, during_time, scenario_range, collision_distance,
                       hmm_model, prediction_time):
        self.start_time = start_time
        self.during_time = during_time
        self.scenario_range = scenario_range
        self.collision_distance = collision_distance
        self.hmm_model = hmm_model
        self.prediction_time = prediction_time
        self.scenario = None
        self.packet_loss_rate = None
        self.headway = None

    def set_scenario(self, scenario):
        self.scenario = scenario

    def set_packet_loss_rate(self, packet_loss_rate):
        self.packet_loss_rate = packet_loss_rate

    def set_headway(self, headway):
        self.headway = headway

    def fog_node_with_real_time_view_experiment(self):
        evaluation_fog_with_real_time_view = []
        dr = data_ready(time=self.start_time, scenario=self.scenario, scenario_range=self.scenario_range,
                        during_time=self.during_time, packet_loss_rate=self.packet_loss_rate, collision_distance=self.collision_distance)
        dr.set_vehicle_traces_and_collision_time_matrix_and_vehicle_id_array()

        print("-"*64)
        print("Data is ready")
        show_time()
        dr.show_detail()

        fg = fog_node(scenario=self.scenario, range=self.scenario_range, hmm_model=self.hmm_model,
                      prediction_time=self.prediction_time, collision_distance=self.collision_distance)

        tp = 0  # true in collision warning message
        fp = 0  # false in collision waring message
        fn = 0  # true not in collision warning message
        for time in range(dr.get_start_time(self.start_time), (dr.get_start_time(self.start_time) + self.during_time)):
            print("-"*64)
            show_time()
            print("time is")
            print(time)

            send_packet = dr.get_send_packets(time=time)
            # print("send packet is ")
            # print(send_packet)
            fg.set_headway(self.headway)
            fg.set_unite_time(time + 1)
            # print(type(fg))
            # print(type(send_packet))
            fg.receive_packets(send_packet)
            fg.selected_packet_under_communication_range()
            fg.form_fog_real_time_view()
            fg.prediction()
            selected_vehicle_id = fg.get_selected_vehicle_id()
            collision_warning_message = fg.get_collision_warning_messages()
            true_collision_warning = self.get_true_collision_warning(dr.get_collision_time_matrix(),
                                                                     dr.get_vehicle_id_array(), time)

            print('$' * 64)
            print('1' * 64)
            print("Selected_vehicle_id")
            print(selected_vehicle_id)
            print('2' * 64)
            print("true_collision_warning")
            print(true_collision_warning)
            print('3' * 64)
            print("collision_waring_message")
            print(collision_warning_message)
            for id in selected_vehicle_id:
                if id in true_collision_warning:
                    if id in collision_warning_message:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if id in collision_warning_message:
                        fp += 1
                    else:
                        pass
        print('&' * 64)
        print(tp)
        print(fp)
        print(fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        experiment_result = {'time': self.start_time, 'precision': precision, 'recall': recall}
        evaluation_fog_with_real_time_view.append(experiment_result)
        return evaluation_fog_with_real_time_view

    def fog_node_without_real_time_view_experiment(self):
        evaluation_fog_without_real_time_view = []
        dr = data_ready(time=self.start_time, scenario=self.scenario, scenario_range=self.scenario_range,
                        during_time=self.during_time, packet_loss_rate=self.packet_loss_rate,
                        collision_distance=self.collision_distance)
        dr.set_vehicle_traces_and_collision_time_matrix_and_vehicle_id_array()

        fg = fog_node(scenario=self.scenario, range=self.scenario_range, hmm_model=self.hmm_model,
                      prediction_time=self.prediction_time, collision_distance=self.collision_distance)
        fg.set_headway(self.headway)

        for time in range(start=dr.get_start_time(self.start_time),
                          stop=(dr.get_start_time(self.start_time) + self.during_time)):
            send_packet = dr.get_send_packets(time=time)
            fg.set_unite_time(time + 1)
            fg.receive_packets(send_packet)
            fg.selected_packet_under_communication_range()
            fg.form_fog_not_real_time_view()
            fg.prediction()
            selected_vehicle_id = fg.get_selected_vehicle_id()
            collision_warning_message = fg.get_collision_warning_messages()
            true_collision_warning = self.get_true_collision_warning(dr.get_collision_time_matrix(),
                                                                     dr.get_vehicle_id_array(), time)
            tp = 0  # true in collision warning message
            fp = 0  # false in collision waring message
            fn = 0  # true not in collision warning message
            for id in selected_vehicle_id:
                if id in true_collision_warning:
                    if id in collision_warning_message:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if id in collision_warning_message:
                        fp += 1
                    else:
                        pass
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            experiment_result = {'time': time, 'precision': precision, 'recall': recall}
            evaluation_fog_without_real_time_view.append(experiment_result)
        return evaluation_fog_without_real_time_view


    def cloud_node_without_real_time_view_experiment(self):
        evaluation_cloud_without_real_time_view = []
        dr = data_ready(time=self.start_time, scenario=self.scenario, scenario_range=self.scenario_range,
                        during_time=self.during_time, packet_loss_rate=self.packet_loss_rate,
                        collision_distance=self.collision_distance)
        dr.set_vehicle_traces_and_collision_time_matrix_and_vehicle_id_array()

        fg = fog_node(scenario=self.scenario, range=self.scenario_range, hmm_model=self.hmm_model,
                      prediction_time=self.prediction_time, collision_distance=self.collision_distance)

        for time in range(start=dr.get_start_time(self.start_time),
                          stop=(dr.get_start_time(self.start_time) + self.during_time)):
            send_packet = dr.get_send_packets(time=time)
            fg.set_headway(self.headway)
            fg.set_unite_time(time + 1)
            fg.receive_packets(send_packet)
            fg.selected_packet_under_communication_range()
            fg.form_cloud_not_real_time_view()
            fg.prediction()
            selected_vehicle_id = fg.get_selected_vehicle_id()
            collision_warning_message = fg.get_collision_warning_messages()
            true_collision_warning = self.get_true_collision_warning(dr.get_collision_time_matrix(),
                                                                     dr.get_vehicle_id_array(), time)
            tp = 0  # true in collision warning message
            fp = 0  # false in collision waring message
            fn = 0  # true not in collision warning message
            for id in selected_vehicle_id:
                if id in true_collision_warning:
                    if id in collision_warning_message:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if id in collision_warning_message:
                        fp += 1
                    else:
                        pass
            print('&'*64)
            print(tp)
            print(fp)
            print(fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            experiment_result = {'time': time, 'precision': precision, 'recall': recall}
            evaluation_cloud_without_real_time_view.append(experiment_result)
        return evaluation_cloud_without_real_time_view


    def get_true_collision_warning(self, collision_time_matrix, vehicle_id_array, time):
        true_collision_warning = []
        for i in range(collision_time_matrix.shape[0]):
            for j in range(collision_time_matrix.shape[1]):
                if collision_time_matrix[i][j] == 0:
                    pass
                else:
                    the_headway = collision_time_matrix[i][j] - time
                    if the_headway < 0:
                        pass
                    elif the_headway < self.headway:
                        vehicle_id_one = vehicle_id_array[i]
                        vehicle_id_two = vehicle_id_array[j]
                        if vehicle_id_one in true_collision_warning:
                            pass
                        else:
                            true_collision_warning.append(int(vehicle_id_one))
                        if vehicle_id_two in true_collision_warning:
                            pass
                        else:
                            true_collision_warning.append(int(vehicle_id_two))
        return true_collision_warning

    def fog(self):
        evaluation_fog_with_real_time_view = []
        dr = data_ready(time=self.start_time, scenario=self.scenario, scenario_range=self.scenario_range,
                        during_time=self.during_time, packet_loss_rate=self.packet_loss_rate,
                        collision_distance=self.collision_distance)
        dr.set_vehicle_traces_and_collision_time_matrix_and_vehicle_id_array()

        print("-" * 64)
        print("Data is ready")
        show_time()
        dr.show_detail()

        fg = fog_node(scenario=self.scenario, range=self.scenario_range, hmm_model=self.hmm_model,
                      prediction_time=self.prediction_time, collision_distance=self.collision_distance)

        tp = 0  # true in collision warning message
        fp = 0  # false in collision waring message
        fn = 0  # true not in collision warning message
        true_collision_number = 0

        for time in range(43269, 43272):
            print("-" * 64)
            show_time()
            print("time is")
            print(time)

            send_packet = dr.get_send_packets(time=time)
            # print("send packet is ")
            # print(send_packet)
            fg.set_headway(self.headway)
            fg.set_unite_time(time + 1)
            # print(type(fg))
            # print(type(send_packet))
            fg.receive_packets(send_packet)
            fg.selected_packet_under_communication_range()
            fg.form_fog_real_time_view()
            fg.prediction()
            selected_vehicle_id = fg.get_selected_vehicle_id()
            collision_warning_message = fg.get_collision_warning_messages()
            true_collision_warning = self.get_true_collision_warning(dr.get_collision_time_matrix(),
                                                                     dr.get_vehicle_id_array(), time)
            true_collision_number += len(true_collision_warning)
            print('$' * 64)
            # print('1' * 64)
            # print("Selected_vehicle_id")
            # print(selected_vehicle_id)
            # print('2' * 64)
            # print("true_collision_warning")
            # print(true_collision_warning)
            # print('3' * 64)
            # print("collision_waring_message")
            # print(collision_warning_message)
            for id in selected_vehicle_id:
                if id in true_collision_warning:
                    if id in collision_warning_message:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if id in collision_warning_message:
                        fp += 1
                    else:
                        pass


        print('&' * 64)

        print(true_collision_number)

        print("TP")
        print(tp)
        print("FP")
        print(fp)
        print("FN")
        print(fn)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        experiment_result = {'time': self.start_time, 'precision': precision, 'recall': recall}
        evaluation_fog_with_real_time_view.append(experiment_result)
        print(experiment_result)
        return evaluation_fog_with_real_time_view

def show_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def start_experiment():
    print("-"*64)
    print("experiment start")
    show_time()

    hmm_type = 'discrete'
    status_number = 37
    train_number = 5000
    accuracy = 0.01
    le_model_file = open(get_le_model_file_path(status_number=status_number, train_number=train_number, accuracy=accuracy), 'rb')
    hmm_model_file = open(get_hmm_model_file_path(type=hmm_type, status_number=status_number, train_number=train_number, accuracy=accuracy), 'rb')
    my_hmm_model = hmm_model(type='discrete', le_model=pickle.load(le_model_file), hmm_model=pickle.load(hmm_model_file), packet_loss_rate=3.00)
    my_experiment = experiment(start_time='12am', during_time=300, scenario_range=500,
                               collision_distance=5.0, hmm_model=my_hmm_model, prediction_time=2)

    my_experiment.set_headway(2)
    my_experiment.set_packet_loss_rate(3.00)
    my_experiment.set_scenario('6')

    my_experiment.fog()

    # my_experiment.fog_node_with_real_time_view_experiment()
    # my_experiment.fog_node_without_real_time_view_experiment()
    # my_experiment.cloud_node_without_real_time_view_experiment()

'''
TODO: fix the file path bug
'''
def get_hmm_model_file_path(type, status_number, train_number, accuracy):
    file_path = r'E:\NearXu\model\model_'
    if type == 'discrete':
        file_path += 'mu_status_'
    elif type == 'continuous':
        file_path += 'gm_status_'
    else:
        pass
        file_path = file_path + str(status_number) + '_number_' + str(train_number) + '_' + str(accuracy) + '_hmm.pkl'
    return r'E:\NearXu\model\model_mu_statue_37_number_10000_0.01_hmm.pkl'

def get_le_model_file_path(status_number, train_number, accuracy):
    file_path = r'E:\NearXu\model\model_mu_status_' + str(status_number) + '_number_' + str(train_number) + '_' + str(accuracy) + '_le.pkl'
    return r'E:\NearXu\model\model_mu_statue_37_number_10000_0.01_le.pkl'

if __name__ == '__main__':
    start_experiment()

