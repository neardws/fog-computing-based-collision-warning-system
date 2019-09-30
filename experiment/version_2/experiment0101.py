from data_ready import data_ready
from node import node
from hmm_model import hmm_model
from result_save import result_save
import threading
import pickle
import pathlib
import time
import warnings
import multiprocessing as mp
from prettytable import PrettyTable

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
    def __init__(self, hmm_model):
        self.hmm_model = hmm_model
        self.headway = None
        self.node = None
        self.saver = None
        self.dr = None
        '''
        static type string 
        '''
        self.TYPE_CLOUD = "TYPE_CLOUD"
        self.TYPE_FOG = "TYPE_FOG"
        self.TYPE_FOG_RE = "TYPE_FOG_RE"
        self.TYPE_FOG_GVCW = "TYPE_FOG_GVCW"

    def set_headway(self, headway):
        self.headway = headway

    def set_node(self, node):
        self.node = node

    def set_saver(self, saver):
        self.saver = saver

    def set_dr(self, data_ready):
        self.dr = data_ready

    def get_hmm_model(self):
        return self.hmm_model

    ''''
    The base algorithm of cloud 
    '''
    def cloud_base_algorithm(self):

        recall_tp = 0
        recall_fn = 0

        precision_tp = 0
        precision_fp = 0

        true_collision_number = 0

        packets_in_seconds = self.dr.return_packet_in_seconds()
        self.saver.write(str(len(packets_in_seconds)))
        self.saver.write(str(packets_in_seconds))
        for packets in packets_in_seconds:
            self.saver.write("-" *64)
            self.saver.write("experiment time " + str(packets[-1].time))
            print("-" * 64)
            show_time()
            print("time is" + str(packets[-1].time))
            self.node.re_init()
            self.node.set_unite_time(packets[-1].time)
            self.node.receive_packets(packets)
            self.node.selected_packet_under_communication_range()
            self.node.form_cloud_not_real_time_view()
            self.node.prediction_by_tradition(self.saver)

            selected_vehicle_id = self.node.get_selected_vehicle_id()
            collision_warning_message = self.node.get_collision_warning_messages()
            true_collision_warning = self.get_true_collision_warning(self.node.get_selected_vehicle_id(),
                                                                     packets[-1].time,
                                                                     self.dr.get_collision_message())
            true_collision_number += len(true_collision_warning)

            self.saver.write('time ' + str(packets[-1].time))
            self.saver.write('selected_id ' + str(selected_vehicle_id))
            self.saver.write('true_collision ' + str(true_collision_warning))
            self.saver.write('collision_warning' + str(collision_warning_message))

            for id in collision_warning_message:
                if id in true_collision_warning:
                    precision_tp += 1
                else:
                    precision_fp += 1

            for id in true_collision_warning:
                if id in collision_warning_message:
                    recall_tp += 1
                else:
                    recall_fn += 1

        print('&' * 64)

        print(true_collision_number)

        print("recall_tp")
        print(recall_tp)
        print("precision_tp")
        print(precision_tp)
        print("FP")
        print(precision_fp)
        print("FN")
        print(recall_fn)

        tp = max(recall_tp, precision_tp)

        if (tp + precision_fp) == 0:
            precision = 1
        else:
            precision = tp / (tp + precision_fp)
        if (tp + recall_fn) == 0:
            recall = 1
        else:
            recall = tp / (tp + recall_fn)
        experiment_result = {'type': 'cloud_base_algorithm',
                             'time': self.dr.time,
                             'scenario': self.dr.scenario,
                             'packet loss rate': self.dr.packet_loss_rate,
                             'headway':self.headway,
                             'recall_tp': recall_tp,
                             'precision_tp': precision_tp,
                             'FP': precision_fp,
                             'FN': recall_fn,
                             'true collision number': true_collision_number,
                             'precision': precision,
                             'recall': recall}
        print(experiment_result)
        result_saver = result_save(type="result_threading_different_scenario_cloud", start_time=self.dr.time,
                                scenario=self.dr.scenario, packet_loss_rate=self.dr.packet_loss_rate,
                                headway=self.headway)
        result_saver.write(str(experiment_result))
        self.saver.write(str(experiment_result))
        print("cloud_base_algorithm result saved")

    def fog_base_algorithm(self):
        recall_tp = 0
        recall_fn = 0

        precision_tp = 0
        precision_fp = 0

        true_collision_number = 0

        packets_in_seconds = self.dr.return_packet_in_seconds()
        self.saver.write(str(len(packets_in_seconds)))
        self.saver.write(str(packets_in_seconds))
        for packets in packets_in_seconds:
            self.saver.write("-" * 64)
            self.saver.write("experiment time " + str(packets[-1].time))
            print("-" * 64)
            show_time()
            print("time is" + str(packets[-1].time))
            self.node.re_init()
            self.node.set_unite_time(packets[-1].time)
            self.node.receive_packets(packets)
            self.node.selected_packet_under_communication_range()
            self.node.form_fog_not_real_time_view()
            self.node.prediction_by_tradition(self.saver)

            selected_vehicle_id = self.node.get_selected_vehicle_id()
            collision_warning_message = self.node.get_collision_warning_messages()
            true_collision_warning = self.get_true_collision_warning(self.node.get_selected_vehicle_id(),
                                                                     packets[-1].time,
                                                                     self.dr.get_collision_message())
            true_collision_number += len(true_collision_warning)

            self.saver.write('time ' + str(packets[-1].time))
            self.saver.write('selected_id ' + str(selected_vehicle_id))
            self.saver.write('true_collision ' + str(true_collision_warning))
            self.saver.write('collision_warning' + str(collision_warning_message))

            for id in collision_warning_message:
                if id in true_collision_warning:
                    precision_tp += 1
                else:
                    precision_fp += 1

            for id in true_collision_warning:
                if id in collision_warning_message:
                    recall_tp += 1
                else:
                    recall_fn += 1

        print('&' * 64)

        print(true_collision_number)

        print("recall_tp")
        print(recall_tp)
        print("precision_tp")
        print(precision_tp)
        print("FP")
        print(precision_fp)
        print("FN")
        print(recall_fn)

        tp = max(recall_tp, precision_tp)

        if (tp + precision_fp) == 0:
            precision = 1
        else:
            precision = tp / (tp + precision_fp)
        if (tp + recall_fn) == 0:
            recall = 1
        else:
            recall = tp / (tp + recall_fn)
        experiment_result = {'type': 'fog_base_algorithm',
                             'time': self.dr.time,
                             'scenario': self.dr.scenario,
                             'packet loss rate': self.dr.packet_loss_rate,
                             'headway': self.headway,
                             'recall_tp': recall_tp,
                             'precision_tp': precision_tp,
                             'FP': precision_fp,
                             'FN': recall_fn,
                             'true collision number': true_collision_number,
                             'precision': precision,
                             'recall': recall}
        print(experiment_result)
        result_saver = result_save(type="result_threading_different_scenario_fog", start_time=self.dr.time,
                                scenario=self.dr.scenario, packet_loss_rate=self.dr.packet_loss_rate,
                                headway=self.headway)
        result_saver.write(str(experiment_result))
        self.saver.write(str(experiment_result))
        print("fog_base_algorithm result saved")

    def fog_correction_algorithm(self):
        recall_tp = 0
        recall_fn = 0

        precision_tp = 0
        precision_fp = 0

        true_collision_number = 0

        packets_in_seconds = self.dr.return_packet_in_seconds()
        self.saver.write(str(len(packets_in_seconds)))
        self.saver.write(str(packets_in_seconds))
        for packets in packets_in_seconds:
            self.saver.write("-" * 64)
            self.saver.write("experiment time " + str(packets[-1].time))
            print("-" * 64)
            show_time()
            print("time is" + str(packets[-1].time))
            self.node.re_init()
            self.node.set_unite_time(packets[-1].time)
            self.node.receive_packets(packets)
            self.node.selected_packet_under_communication_range()

            self.node.form_fog_real_time_view()
            self.node.prediction_by_tradition(self.saver)

            selected_vehicle_id = self.node.get_selected_vehicle_id()
            collision_warning_message = self.node.get_collision_warning_messages()
            true_collision_warning = self.get_true_collision_warning(self.node.get_selected_vehicle_id(),
                                                                     packets[-1].time,
                                                                     self.dr.get_collision_message())
            true_collision_number += len(true_collision_warning)

            self.saver.write('time ' + str(packets[-1].time))
            self.saver.write('selected_id ' + str(selected_vehicle_id))
            self.saver.write('true_collision ' + str(true_collision_warning))
            self.saver.write('collision_warning' + str(collision_warning_message))

            for id in collision_warning_message:
                if id in true_collision_warning:
                    precision_tp += 1
                else:
                    precision_fp += 1

            for id in true_collision_warning:
                if id in collision_warning_message:
                    recall_tp += 1
                else:
                    recall_fn += 1

        print('&' * 64)

        print(true_collision_number)

        print("recall_tp")
        print(recall_tp)
        print("precision_tp")
        print(precision_tp)
        print("FP")
        print(precision_fp)
        print("FN")
        print(recall_fn)

        tp = max(recall_tp, precision_tp)

        if (tp + precision_fp) == 0:
            precision = 1
        else:
            precision = tp / (tp + precision_fp)
        if (tp + recall_fn) == 0:
            recall = 1
        else:
            recall = tp / (tp + recall_fn)
        experiment_result = {'type': 'fog_correction_algorithm',
                             'time': self.dr.time,
                             'scenario': self.dr.scenario,
                             'packet loss rate': self.dr.packet_loss_rate,
                             'headway': self.headway,
                             'recall_tp': recall_tp,
                             'precision_tp': precision_tp,
                             'FP': precision_fp,
                             'FN': recall_fn,
                             'true collision number': true_collision_number,
                             'precision': precision,
                             'recall': recall}
        print(experiment_result)
        result_saver = result_save(type="result_threading_different_scenario_fog_re", start_time=self.dr.time,
                                scenario=self.dr.scenario, packet_loss_rate=self.dr.packet_loss_rate,
                                headway=self.headway)
        result_saver.write(str(experiment_result))
        self.saver.write(str(experiment_result))
        print("fog_correction_algorithm result saved")

    def fog_correction_hmm_algorithm(self):
        recall_tp = 0
        recall_fn = 0

        precision_tp = 0
        precision_fp = 0

        true_collision_number = 0

        packets_in_seconds = self.dr.return_packet_in_seconds()
        self.saver.write(str(len(packets_in_seconds)))
        self.saver.write(str(packets_in_seconds))
        if len(packets_in_seconds) == 0:
            print("%"*64)
            print("packets is null")
        for packets in packets_in_seconds:
            self.saver.write("-" * 64)
            self.saver.write("experiment time " + str(packets[-1].time))
            print("-" * 64)
            show_time()
            print("time is" + str(packets[-1].time))
            self.node.re_init()
            self.node.set_unite_time(packets[-1].time)
            self.node.receive_packets(packets)
            self.node.selected_packet_under_communication_range()
            self.node.form_fog_real_time_view()
            self.node.prediction_by_hmm(self.saver)

            selected_vehicle_id = self.node.get_selected_vehicle_id()
            collision_warning_message = self.node.get_collision_warning_messages()
            true_collision_warning = self.get_true_collision_warning(self.node.get_selected_vehicle_id(),
                                                                     packets[-1].time,
                                                                     self.dr.get_collision_message())
            true_collision_number += len(true_collision_warning)

            self.saver.write('time ' + str(packets[-1].time))
            self.saver.write('selected_id ' + str(selected_vehicle_id))
            self.saver.write('true_collision ' + str(true_collision_warning))
            self.saver.write('collision_warning' + str(collision_warning_message))

            for id in collision_warning_message:
                if id in true_collision_warning:
                    precision_tp += 1
                else:
                    precision_fp += 1

            for id in true_collision_warning:
                if id in collision_warning_message:
                    recall_tp += 1
                else:
                    recall_fn += 1

        print('&' * 64)

        print(true_collision_number)

        print("recall_tp")
        print(recall_tp)
        print("precision_tp")
        print(precision_tp)
        print("FP")
        print(precision_fp)
        print("FN")
        print(recall_fn)

        tp = max(recall_tp, precision_tp)

        if (tp + precision_fp) == 0:
            precision = 1
        else:
            precision = tp / (tp + precision_fp)
        if (tp + recall_fn) == 0:
            recall = 1
        else:
            recall = tp / (tp + recall_fn)
        experiment_result = {'type': 'fog with realtime view',
                             'time': self.dr.time,
                             'scenario': self.dr.scenario,
                             'packet loss rate': self.dr.packet_loss_rate,
                             'headway': self.headway,
                             'recall_tp': recall_tp,
                             'precision_tp': precision_tp,
                             'FP': precision_fp,
                             'FN': recall_fn,
                             'true collision number': true_collision_number,
                             'precision': precision,
                             'recall': recall}
        print(experiment_result)
        result_saver = result_save(type="result_threading_different_scenario_fog_gvcw", start_time=self.dr.time,
                                scenario=self.dr.scenario, packet_loss_rate=self.dr.packet_loss_rate,
                                headway=self.headway)
        result_saver.write(str(experiment_result))
        self.saver.write(str(experiment_result))
        print("fog_correction_hmm_algorithm result saved")

    def get_true_collision_warning(self, selected_id, time, collision_message):
        true_collision_warning = []

        for message in collision_message:
            vehicle_one_id = message['vehicleOne']
            vehicle_two_id = message['vehicleTwo']
            collision_time = message['collisionTime']
            if (collision_time - time) < 0:
                pass
            else:
                if (collision_time - time) <= self.headway:
                    if vehicle_one_id in selected_id:
                        if vehicle_two_id in selected_id:
                            true_collision_warning.append(int(vehicle_one_id))
                            true_collision_warning.append(int(vehicle_two_id))
                        else:
                            pass
                    else:
                        pass
        return true_collision_warning


def show_time():
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))


def show_dr_details(drs, saver):
    table = PrettyTable(['time','scenario', 'scen_range',  'during',  'coll_dis', 'traffic_density', 'vehicle_speed', 'vehicle_acceleration', 'collision_number'])
    for dr in drs:
        time = dr.time
        scenario = dr.scenario
        scenario_range = dr.scenario_range
        during_time = dr.during_time
        collision_distance = dr.collision_distance
        collision_message_number = len(dr.get_collision_message())
        features = dr.get_features_of_data_ready()
        traffic_density = features['traffic_density']
        vehicle_speed = features['vehicle_speed']
        vehicle_acceleration = features['vehicle_acceleration']
        row = []
        row.append(str(time))
        row.append(str(scenario))
        row.append(str(scenario_range))
        row.append(str(during_time))
        row.append(str(collision_distance))
        row.append(str(traffic_density))
        row.append(str(vehicle_speed))
        row.append(str(vehicle_acceleration))
        row.append(str(collision_message_number))
        table.add_row(row)

    print(table)
    saver.write(str(table))


def start_experiment(type, my_experiment, saver, dr, prediction_time, headway):
    print("start experiment")

    print(type)
    print(type == my_experiment.TYPE_CLOUD)
    print(type == my_experiment.TYPE_FOG)
    print(type == my_experiment.TYPE_FOG_RE)
    print(type == my_experiment.TYPE_FOG_GVCW)
    my_node = node(scenario=dr.scenario, headway=headway, range=dr.scenario_range,
                   hmm_model=my_experiment.get_hmm_model(), prediction_time=prediction_time,
                   collision_distance=dr.collision_distance)
    print("——" * 64)
    my_experiment.set_headway(headway)
    my_experiment.set_dr(dr)
    my_experiment.set_node(my_node)
    my_experiment.set_saver(saver)
    if type == my_experiment.TYPE_CLOUD:
        print("my_experiment.TYPE_CLOUD")
        my_experiment.cloud_base_algorithm()
    elif type == my_experiment.TYPE_FOG:
        print("my_experiment.TYPE_FOG")
        my_experiment.fog_base_algorithm()
    elif type == my_experiment.TYPE_FOG_RE:
        print("my_experiment.TYPE_FOG_RE")
        my_experiment.fog_correction_algorithm()
    elif type == my_experiment.TYPE_FOG_GVCW:
        print("my_experiment.TYPE_FOG_GVCW")
        my_experiment.fog_correction_hmm_algorithm()
    else:
        print("TYPE ERROR")



def start_experiment_by_threading(dr, prediction_time, headway):
    default_packet_loss_rate = 3
    default_headway = 3
    print("start threading")
    hmm_type = 'discrete'
    status_number = 37
    train_number = 5000
    accuracy = 0.01
    le_model_file = open(
        get_le_model_file_path(status_number=status_number, train_number=train_number, accuracy=accuracy), 'rb')
    hmm_model_file = open(get_hmm_model_file_path(type=hmm_type, status_number=status_number, train_number=train_number,
                                                  accuracy=accuracy), 'rb')
    my_hmm_model = hmm_model(type='discrete', le_model=pickle.load(le_model_file),
                             hmm_model=pickle.load(hmm_model_file))
    my_experiment = experiment(my_hmm_model)
    my_saver1 = result_save(type="experiment0101_01_threading_different_scenario_cloud", start_time=dr.time,
                           scenario=dr.scenario, packet_loss_rate=default_packet_loss_rate,
                           headway=default_headway)
    my_saver2 = result_save(type="experiment0101_01_threading_different_scenario_fog", start_time=dr.time,
                           scenario=dr.scenario, packet_loss_rate=default_packet_loss_rate,
                           headway=default_headway)
    my_saver3 = result_save(type="experiment0101_01_threading_different_scenario_fog_re", start_time=dr.time,
                           scenario=dr.scenario, packet_loss_rate=default_packet_loss_rate,
                           headway=default_headway)
    my_saver4 = result_save(type="experiment0101_01_threading_different_scenario_fog_gvcw", start_time=dr.time,
                           scenario=dr.scenario, packet_loss_rate=default_packet_loss_rate,
                           headway=default_headway)
    dr.set_vehicle_traces_and_collision_time_matrix_and_vehicle_id_array()
    print("dr is ready!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    cloud_thread = threading.Thread(target=start_experiment, args=("TYPE_CLOUD", my_experiment, my_saver1, dr, prediction_time, headway))
    cloud_thread.start()
    fog_thread = threading.Thread(target= start_experiment, args=("TYPE_FOG", my_experiment, my_saver2, dr, prediction_time, headway))
    fog_thread.start()
    fog_re_thread = threading.Thread(target=start_experiment, args=("TYPE_FOG_RE", my_experiment, my_saver3, dr, prediction_time, headway))
    fog_re_thread.start()
    fog_gvcw_thread = threading.Thread(target=start_experiment, args=("TYPE_FOG_GVCW", my_experiment, my_saver4, dr, prediction_time, headway))
    fog_gvcw_thread.start()


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
    file_path = r'E:\NearXu\model\model_mu_status_' + str(status_number) + '_number_' + str(train_number) + '_' + str(
        accuracy) + '_le.pkl'
    return r'E:\NearXu\model\model_mu_statue_37_number_10000_0.01_le.pkl'


def get_data_ready(start_time, scenario, scenario_range, during_time, packet_loss_rate, collision_distance):
    dr = data_ready(time=start_time, scenario=scenario, scenario_range=scenario_range,
                    during_time=during_time, packet_loss_rate=packet_loss_rate,collision_distance=collision_distance)
    dr.set_vehicle_traces_and_collision_time_matrix_and_vehicle_id_array()

    print("-" * 64)
    print("Data is ready")
    show_time()
    # saver.write(str(dr.show_detail()))
    return dr



'''
Experiment
Change the scenario 
'''
def main():

    ''''
    init value
    '''
    different_start_time = ['6am', '7am', '8am', '9am', '10am', '11am', '12am', '1pm', '2pm', '3pm', '4pm', '5pm',
                            '6pm', '7pm', '8pm', '9pm', '10pm', '11pm']
    different_scenario = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    different_headway = [1, 2, 3, 4, 5]
    different_packet_loss_rate = [0, 1.5, 3, 4.5, 6]

    '''
    default value
    '''
    during_time = 100
    scenario_range = 500
    collision_distance = 5.0
    prediction_time = 1
    default_packet_loss_rate = 3
    default_headway = 3

    '''
    change value
    selected_scenario
    | 7am  |    2     |    500     |  100   |   5.0    |       139       | 18.92823498469161  |   0.015845421133777758  |        18        |
    | 7am  |    6     |    500     |  100   |   5.0    |       120       | 13.062099727709821 |   -0.03689736923888576  |        40        |
    | 7am  |    9     |    500     |  100   |   5.0    |       121       | 23.024674146701795 |  -0.053098976624194924  |        15        |
    | 8am  |    2     |    500     |  100   |   5.0    |        80       | 17.850042710981093 |   -0.04453214370222366  |        13        |
    | 8am  |    6     |    500     |  100   |   5.0    |        81       | 12.942697656478494 |   0.007933901577578919  |        19        |
    | 9am  |    6     |    500     |  100   |   5.0    |        54       | 12.567781576115875 |    0.0909153339231181   |        11        |
    | 12am |    5     |    500     |  100   |   5.0    |        27       | 17.21671993750567  |   0.09291386434744327   |        15        |
    | 12am |    6     |    500     |  100   |   5.0    |       104       | 12.999320726828484 |   -0.1348130505809356   |        40        |
    | 1pm  |    6     |    500     |  100   |   5.0    |        67       | 12.762791295913285 |   0.028545183911414716  |        24        |
    | 4pm  |    1     |    500     |  100   |   5.0    |        54       | 14.380193869754692 |    0.2260140702499548   |        10        |
    | 4pm  |    2     |    500     |  100   |   5.0    |        85       | 19.22874385678008  |    0.1646757530570065   |        16        |
    | 4pm  |    7     |    500     |  100   |   5.0    |        72       | 17.277884038915342 |   -0.08825776920478888  |        22        |
    | 4pm  |    9     |    500     |  100   |   5.0    |        93       | 24.54092914894115  |  0.00046152994549642987 |        14        |
    | 5pm  |    1     |    500     |  100   |   5.0    |        54       | 14.474476050300543 | -0.00018715848820969588 |        14        |
    | 5pm  |    2     |    500     |  100   |   5.0    |       107       | 19.883287335270897 |   0.10988931866385934   |        32        |
    | 5pm  |    6     |    500     |  100   |   5.0    |       105       | 12.419208221977446 |   -0.09433623844682969  |        46        |
    | 5pm  |    7     |    500     |  100   |   5.0    |        82       | 17.231847300618824 |   0.03972400321151402   |        12        |
    | 5pm  |    9     |    500     |  100   |   5.0    |        98       | 23.928981914534443 |   -0.0358471618632544   |        19        |
    | 6pm  |    1     |    500     |  100   |   5.0    |        55       |  15.5429032483822  |   0.47265437448894154   |        11        |
    | 6pm  |    2     |    500     |  100   |   5.0    |       114       |  19.1815578754212  |   0.06047709119872064   |        27        |
    | 6pm  |    6     |    500     |  100   |   5.0    |       120       |  12.3646782906035  |  -0.050349146278667156  |        31        |
    | 6pm  |    7     |    500     |  100   |   5.0    |        62       | 17.638282520594082 |   0.023201055753263394  |        15        |
    | 6pm  |    8     |    500     |  100   |   5.0    |        56       | 16.779824864912797 |    0.1317913627434077   |        21        |
    | 6pm  |    9     |    500     |  100   |   5.0    |        93       | 24.091808199008295 |   0.05898514709501288   |        18        |
    | 7pm  |    6     |    500     |  100   |   5.0    |       106       | 10.598324438124479 |   0.07562112815649023   |       168        |
    | 8pm  |    6     |    500     |  100   |   5.0    |        76       | 13.065029159604101 |  -0.045240582963270534  |        12        |
    | 10pm |    6     |    500     |  100   |   5.0    |        54       | 14.015215311579345 |    0.2029108770185149   |        12        |
    '''

    drs = []
    file_path = "drs.pkl"

    path = pathlib.Path(file_path)
    if path.is_file():
        with open(file_path, 'rb') as drs_file:
            drs = pickle.load(drs_file)
            print("drs is load")
            pool = mp.Pool(processes=10)
            jobs = []
            number = 0
            for dr in drs:
                number += 1
                if number <= 6:
                    pass
                print("number " + str(number))

                default_packet_loss_rate = 3
                default_headway = 3
                print("start threading")
                hmm_type = 'discrete'
                status_number = 37
                train_number = 5000
                accuracy = 0.01
                le_model_file = open(
                    get_le_model_file_path(status_number=status_number, train_number=train_number, accuracy=accuracy),
                    'rb')
                hmm_model_file = open(
                    get_hmm_model_file_path(type=hmm_type, status_number=status_number, train_number=train_number,
                                            accuracy=accuracy), 'rb')
                my_hmm_model = hmm_model(type='discrete', le_model=pickle.load(le_model_file),
                                         hmm_model=pickle.load(hmm_model_file))
                my_experiment = experiment(my_hmm_model)
                my_saver1 = result_save(type="experiment0101_01_threading_different_scenario_cloud", start_time=dr.time,
                                        scenario=dr.scenario, packet_loss_rate=default_packet_loss_rate,
                                        headway=default_headway)
                my_saver2 = result_save(type="experiment0101_01_threading_different_scenario_fog", start_time=dr.time,
                                        scenario=dr.scenario, packet_loss_rate=default_packet_loss_rate,
                                        headway=default_headway)
                my_saver3 = result_save(type="experiment0101_01_threading_different_scenario_fog_re",
                                        start_time=dr.time,
                                        scenario=dr.scenario, packet_loss_rate=default_packet_loss_rate,
                                        headway=default_headway)
                my_saver4 = result_save(type="experiment0101_01_threading_different_scenario_fog_gvcw",
                                        start_time=dr.time,
                                        scenario=dr.scenario, packet_loss_rate=default_packet_loss_rate,
                                        headway=default_headway)
                dr.set_vehicle_traces_and_collision_time_matrix_and_vehicle_id_array()
                print("dr is ready!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                jobs.append(pool.apply_async(start_experiment,
                                             ("TYPE_CLOUD", my_experiment, my_saver1, dr, prediction_time, default_headway)))
                jobs.append(pool.apply_async(start_experiment,
                                             ("TYPE_FOG", my_experiment, my_saver2, dr, prediction_time, default_headway)))
                jobs.append(pool.apply_async(start_experiment,
                                             ("TYPE_FOG_RE", my_experiment, my_saver3, dr, prediction_time, default_headway)))
                jobs.append(pool.apply_async(start_experiment,
                                             ("TYPE_FOG_GVCW", my_experiment, my_saver4, dr, prediction_time, default_headway)))

            for job in jobs:
                job.get()
            pool.close()
    else:
        dr = get_data_ready(start_time='7am',
                            scenario='2',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)
        drs.append(dr)
        dr = get_data_ready(start_time='7am',
                            scenario='6',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)
        drs.append(dr)
        dr = get_data_ready(start_time='7am',
                            scenario='9',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)

        dr = get_data_ready(start_time='8am',
                            scenario='2',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='8am',
                            scenario='6',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)

        dr = get_data_ready(start_time='9am',
                            scenario='6',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)

        dr = get_data_ready(start_time='12am',
                            scenario='5',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='12am',
                            scenario='6',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)

        dr = get_data_ready(start_time='1pm',
                            scenario='6',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)

        dr = get_data_ready(start_time='4pm',
                            scenario='1',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='4pm',
                            scenario='2',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='4pm',
                            scenario='7',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='4pm',
                            scenario='9',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)

        dr = get_data_ready(start_time='5pm',
                            scenario='1',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='5pm',
                            scenario='2',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='5pm',
                            scenario='6',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='5pm',
                            scenario='7',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='5pm',
                            scenario='9',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)

        dr = get_data_ready(start_time='6pm',
                            scenario='1',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='6pm',
                            scenario='2',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='6pm',
                            scenario='6',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)
        drs.append(dr)
        dr = get_data_ready(start_time='6pm',
                            scenario='7',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)
        drs.append(dr)
        dr = get_data_ready(start_time='6pm',
                            scenario='8',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        dr = get_data_ready(start_time='6pm',
                            scenario='9',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)

        dr = get_data_ready(start_time='7pm',
                            scenario='6',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)

        dr = get_data_ready(start_time='8pm',
                            scenario='6',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)
        drs.append(dr)

        dr = get_data_ready(start_time='10pm',
                            scenario='6',
                            scenario_range=scenario_range,
                            during_time=during_time,
                            packet_loss_rate=default_packet_loss_rate,
                            collision_distance=collision_distance)

        drs.append(dr)
        with open(file_path, 'wb') as drs_file:
            pickle.dump(drs, drs_file)
            print("drs is saved")



if __name__ == '__main__':
    main()
