from data_ready import data_ready
from node import node
from hmm_model import hmm_model
from result_save import result_save
import pickle
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

    def set_headway(self, headway):
        self.headway = headway

    def set_node(self, node):
        self.node = node

    def set_server(self, server):
        self.server = server

    def set_dr(self, data_ready):
        self.dr = data_ready

    ''''
    The base algorithm of cloud 
    '''
    def cloud_base_algorithm(self):

        recall_tp = 0
        recall_fn = 0

        precision_tp = 0
        precision_fp = 0

        true_collision_number = 0

        packets_in_seconds = self.dr.get_packet_in_seconds()
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

            selected_vehicle_id = node.get_selected_vehicle_id()
            collision_warning_message = node.get_collision_warning_messages()
            true_collision_warning = self.get_true_collision_warning(node.get_selected_vehicle_id(),
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
                             'headway': self.dr.headway,
                             'recall_tp': recall_tp,
                             'precision_tp': precision_tp,
                             'FP': precision_fp,
                             'FN': recall_fn,
                             'true collision number': true_collision_number,
                             'precision': precision,
                             'recall': recall}
        print(experiment_result)
        self.saver.write(str(experiment_result))
        print("cloud_base_algorithm result saved")

    def fog_base_algorithm(self):
        recall_tp = 0
        recall_fn = 0

        precision_tp = 0
        precision_fp = 0

        true_collision_number = 0

        packets_in_seconds = self.dr.get_packet_in_seconds()
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
                             'headway': self.dr.headway,
                             'recall_tp': recall_tp,
                             'precision_tp': precision_tp,
                             'FP': precision_fp,
                             'FN': recall_fn,
                             'true collision number': true_collision_number,
                             'precision': precision,
                             'recall': recall}
        print(experiment_result)
        self.saver.write(str(experiment_result))
        print("fog_base_algorithm result saved")

    def fog_correction_algorithm(self):
        recall_tp = 0
        recall_fn = 0

        precision_tp = 0
        precision_fp = 0

        true_collision_number = 0

        packets_in_seconds = self.dr.get_packet_in_seconds()
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
                             'headway': self.dr.headway,
                             'recall_tp': recall_tp,
                             'precision_tp': precision_tp,
                             'FP': precision_fp,
                             'FN': recall_fn,
                             'true collision number': true_collision_number,
                             'precision': precision,
                             'recall': recall}
        print(experiment_result)
        self.saver.write(str(experiment_result))
        print("fog_correction_algorithm result saved")

    def fog_correction_hmm_algorithm(self):
        recall_tp = 0
        recall_fn = 0

        precision_tp = 0
        precision_fp = 0

        true_collision_number = 0

        packets_in_seconds = self.dr.get_packet_in_seconds()
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
                             'headway': self.dr.headway,
                             'recall_tp': recall_tp,
                             'precision_tp': precision_tp,
                             'FP': precision_fp,
                             'FN': recall_fn,
                             'true collision number': true_collision_number,
                             'precision': precision,
                             'recall': recall}
        print(experiment_result)
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
    table = PrettyTable(['time','scenario', 'scen_range',  'during',  'coll_dis', 'traffic_density', 'vehicle_speed', 'vehicle_acceleration'])
    for dr in drs:
        time = dr.time
        scenario = dr.scenario
        scenario_range = dr.scenario_range
        during_time = dr.during_time
        collision_distance = dr.collision_distance
        features = dr.get_features_of_data_ready()
        traffic_density = features['traffic_density']
        vehicle_speed = features['vehicle_speed']
        vehicle_acceleraction = features['vehicle_acceleration']
        row = []
        row.append(str(time))
        row.append(str(scenario))
        row.append(str(scenario_range))
        row.append(str(during_time))
        row.append(str(collision_distance))
        row.append(str(traffic_density))
        row.append(str(vehicle_speed))
        row.append(str(vehicle_acceleraction))
        table.add_row(row)

    print(table)
    saver.write(table)



def start_experiment():
    fog_saver = result_save(type="fog_with", start_time=start_time, scenario=scenario, packet_loss_rate=packet_loss_rate, headway=headway)
    fog_without_saver = result_save(type="fog_without", start_time=start_time, scenario=scenario, packet_loss_rate=packet_loss_rate, headway=headway)
    cloud_without_saver = result_save(type="cloud_without", start_time=start_time, scenario=scenario, packet_loss_rate=packet_loss_rate, headway=headway)

    print("-" * 64)
    print(start_time)
    print(during_time)
    print(headway)
    print(packet_loss_rate)
    print(scenario)
    show_time()

    hmm_type = 'discrete'
    status_number = 37
    train_number = 5000
    accuracy = 0.01
    le_model_file = open(
        get_le_model_file_path(status_number=status_number, train_number=train_number, accuracy=accuracy), 'rb')
    hmm_model_file = open(get_hmm_model_file_path(type=hmm_type, status_number=status_number, train_number=train_number,
                                                  accuracy=accuracy), 'rb')
    my_hmm_model = hmm_model(type='discrete', le_model=pickle.load(le_model_file),
                             hmm_model=pickle.load(hmm_model_file), packet_loss_rate=packet_loss_rate)

    my_experiment = experiment(start_time=start_time, during_time=during_time, scenario_range=scenario_range,
                               collision_distance=collision_distance, hmm_model=my_hmm_model,
                               prediction_time=headway)

    my_experiment.set_headway(headway)
    my_experiment.set_packet_loss_rate(packet_loss_rate)
    my_experiment.set_scenario(scenario)

    DR_PATH = r'dr'
    dr_name = DR_PATH + '_type_fog_with_' + 'time' + str(start_time) + '_scenario_' + str(
        scenario) + '_plr_' \
                    '' + str(packet_loss_rate) + '_headway_' + str(headway) + '.pkl'
    if first == 1:
        dr = my_experiment.get_data_ready(fog_saver)
        with open(dr_name, "wb") as file:
            pickle.dump(dr, file)
        my_experiment.fog_node_with_real_time_view_experiment(dr, fog_saver)
        fog_without_saver.write(str(dr.show_detail()))
        cloud_without_saver.write(str(dr.show_detail()))
        my_experiment.fog_node_without_real_time_view_experiment(dr, fog_without_saver)
        my_experiment.cloud_node_without_real_time_view_experiment(dr, cloud_without_saver)

    elif first == 0:
        with open(dr_name, "rb") as file:
            dr = pickle.load(file)
            fog_saver.write(str(dr.show_detail()))
            fog_without_saver.write(str(dr.show_detail()))
            cloud_without_saver.write(str(dr.show_detail()))
            my_experiment.fog_node_with_real_time_view_experiment(dr, fog_saver)
            my_experiment.fog_node_without_real_time_view_experiment(dr, fog_without_saver)
            my_experiment.cloud_node_without_real_time_view_experiment(dr, cloud_without_saver)
    else:
        pass




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


def main():
    different_start_time = ['10am', '3pm', '12am', '6pm', '9pm']
    different_scenario = ['6', '7', '5', '8', '9']
    different_headway = [1, 2, 4, 5, 10]
    different_packet_loss_rate = [0, 1.5, 3, 6, 12]

    start_time = different_start_time[2]
    during_time = 100
    scenario_range = 300
    collision_distance = 3.5
    prediction_time = 1
    parameters_list = []

    for i in range(5):
        parameters = {'start_time': start_time,
                      'during_time': during_time,
                      'headway': different_headway[i],
                      'scenario': different_scenario[2],
                      'scenario_range': scenario_range,
                      'collision_distance': collision_distance,
                      'prediction_time': prediction_time,
                      'packet_loss_rate': different_packet_loss_rate[2]}
        parameters1 = {'start_time': start_time,
                       'during_time': during_time,
                       'headway': different_headway[1],
                       'scenario': different_scenario[i],
                       'scenario_range': scenario_range,
                       'collision_distance': collision_distance,
                       'prediction_time': prediction_time,
                       'packet_loss_rate': different_packet_loss_rate[2]}
        parameters2 = {'start_time': start_time,
                       'during_time': during_time,
                       'headway': different_headway[1],
                       'scenario': different_scenario[2],
                       'scenario_range': scenario_range,
                       'collision_distance': collision_distance,
                       'prediction_time': prediction_time,
                       'packet_loss_rate': different_packet_loss_rate[i]}
        parameters_list.append(parameters)
        parameters_list.append(parameters1)
        parameters_list.append(parameters2)

    pool = mp.Pool(processes=15)
    jobs = []
    for parameters in parameters_list:

        jobs.append(pool.apply_async(start_experiment,
                                     (1, parameters['start_time'], parameters['during_time'],
                                      parameters['headway'], parameters['packet_loss_rate'], parameters['scenario'],
                                      parameters['scenario_range'], parameters['collision_distance'],
                                      parameters['prediction_time'])))

    for job in jobs:
        job.get()
    pool.close()


def main_test():
    different_start_time = ['10am', '3pm', '12am', '6pm', '9pm']
    different_scenario = ['6', '7', '5', '8', '9']
    different_headway = [1, 2, 4, 5, 10]
    different_packet_loss_rate = [0, 1.5, 3, 6, 12]

    start_time = different_start_time[2]
    during_time = 100
    scenario_range = 300
    collision_distance = 3.5
    prediction_time = 1
    parameters_list = []

    for i in range(5):
        parameters = {'start_time': start_time,
                      'during_time': during_time,
                      'headway': different_headway[i],
                      'scenario': different_scenario[2],
                      'scenario_range': scenario_range,
                      'collision_distance': collision_distance,
                      'prediction_time': prediction_time,
                      'packet_loss_rate': different_packet_loss_rate[2]}
        parameters1 = {'start_time': start_time,
                       'during_time': during_time,
                       'headway': different_headway[1],
                       'scenario': different_scenario[i],
                       'scenario_range': scenario_range,
                       'collision_distance': collision_distance,
                       'prediction_time': prediction_time,
                       'packet_loss_rate': different_packet_loss_rate[2]}
        parameters2 = {'start_time': start_time,
                       'during_time': during_time,
                       'headway': different_headway[1],
                       'scenario': different_scenario[2],
                       'scenario_range': scenario_range,
                       'collision_distance': collision_distance,
                       'prediction_time': prediction_time,
                       'packet_loss_rate': different_packet_loss_rate[i]}
        parameters_list.append(parameters)
        parameters_list.append(parameters1)
        parameters_list.append(parameters2)

    pool = mp.Pool(processes=15)
    jobs = []
    for parameters in parameters_list:
        jobs.append(pool.apply_async(start_experiment,
                                     (1, parameters['start_time'], parameters['during_time'],
                                      parameters['headway'], parameters['packet_loss_rate'], parameters['scenario'],
                                      parameters['scenario_range'], parameters['collision_distance'],
                                      parameters['prediction_time'])))

    for job in jobs:
        job.get()
    pool.close()


if __name__ == '__main__':
    main_test()

