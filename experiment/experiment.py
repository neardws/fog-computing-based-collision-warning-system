from .data_ready import data_ready
from .fog_node import fog_node
from .hmm_model import hmm_model
import pickle
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
                       hmm_type, hmm_model, le_model, prediction_time):
        self.start_time = start_time
        self.during_time = during_time
        self.scenario_range = scenario_range
        self.collision_distance = collision_distance
        self.hmm_type = hmm_type
        self.hmm_model = hmm_model
        self.le_model = le_model
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

    def get_hmm_model_file_path(self, type, status_number, train_number, accuracy):
        file_path = '../model/model_'
        if type == 'discrete':
            file_path += 'mu_status_'
        elif type == 'continuous':
            file_path += 'gm_status_'
        else:
            pass
            file_path = file_path + str(status_number) + '_number_' + str(train_number) + '_' + str(accuracy) + '_hmm.pkl'
        return file_path

    def get_le_model_file_path(self, status_number, train_number, accuracy):
        file_path = '../model/model_mu_status_' + str(status_number) + '_number_' + str(train_number) + '_' + str(accuracy) + '_le.pkl'
        return file_path

    def experiment_setup(self):
        pass

    def one_experiment(self):
        dr = data_ready(time=self.start_time, scenario=self.scenario, scenario_range=self.scenario_range,
                        during_time=self.during_time, packet_loss_rate=self.packet_loss_rate, collision_distance=self.collision_distance)
        dr.set_vehicle_traces_and_collision_time_matrix_and_vehicle_id_array()

        fg = fog_node(scenario=self.scenario, range=self.scenario_range, hmm_model=self.hmm_model,
                      prediction_time=self.prediction_time, collision_distance=self.collision_distance)

        for time in range(start=dr.get_start_time(self.start_time), stop=(dr.get_start_time(self.start_time) + self.during_time)):
            send_packet = dr.get_send_packets(time=time)
            fg.set_headway(self.headway)
            fg.set_unite_time(time+1)
            fg.receive_packets(send_packet)
            fg.selected_packet_under_communication_range()
            fg.form_real_time_view()
            fg.prediction()

