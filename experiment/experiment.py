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



'''
Todo get test data in each second and 
'''
def get_test_data():
    pass


'''
Todo add delay and packet loss in each second  
'''
def vehicles_knowledge():
    pass


'''
Todo form real-time view of vehicles in fog node
'''
def fog_real_time_view():
    pass


'''
Todo statistics precision and recall in each second
'''
def prediction_and_statistics():
    pass


