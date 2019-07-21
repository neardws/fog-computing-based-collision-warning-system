import numpy as np
from transmission_model import transmission_model
class fog_node:
    def __init__(self, scenario, range, hmm_model, prediction_time, collision_distance):
        self.location_x = self.get_scenario_xy(scenario)[0]
        self.location_y = self.get_scenario_xy(scenario)[1]
        self.communication_range = range
        self.hmm_model = hmm_model
        self.prediction_time = prediction_time
        self.collision_distance = collision_distance
        self.headway = None
        self.unite_time = None
        self.prediction_time = None
        self.vehicles = None
        self.selected_vehicles = []
        self.history_record = []
        self.prediction_traces = []
        self.collision_warning_messages = []
        self.selected_vehicle_collision_time_matrix = None

    def set_headway(self, headway):
        self.headway = headway

    def set_unite_time(self, time):
        self.unite_time = time

    def set_prediction_time(self, time):
        self.prediction_time = time

    def receive_packets(self, vehicles):
        self.vehicles = vehicles

    def get_selected_vehicle_id(self):
        selected_vehicle_id = []
        for vehicle in self.selected_vehicles:
            selected_vehicle_id.append(vehicle.vehicleID)
        return selected_vehicle_id

    def get_collision_warning_messages(self):
        return self.collision_warning_messages

    def selected_packet_under_communication_range(self):
        for vehicle in self.vehicles:
            if self.packet_is_under_communication_range(vehicle):
                if not vehicle.packet_loss:
                    self.selected_vehicles.append(vehicle)
        self.history_record.append(self.selected_vehicles)

    def packet_is_under_communication_range(self, vehicle):
        under_communication_range = False
        distance = np.sqrt(np.square(self.location_x - vehicle.location_x) + np.square(self.location_y - vehicle.location_y))
        if distance <= self.communication_range:
            under_communication_range = True
        return under_communication_range

    '''
    Step one: fix packet loss
    Step two: fix transmission delay
    '''
    def form_real_time_view(self):
        self.fix_packet_loss()
        self.fix_transmission_delay(self.unite_time)

    def fix_transmission_delay(self, unite_time):
        for vehicle in self.selected_vehicles:
            receive_time = vehicle.time + vehicle.transmission_delay
            estimated_delay = transmission_model.get_transmission_delay()
            estimated_send_time = receive_time - estimated_delay / 1000
            history_vehicle = self.get_vehicle_form_history_record(vehicle.vehicleID)
            if history_vehicle is not None:
                speed_x = vehicle.location_x - history_vehicle.location_x
                speed_y = vehicle.location_y - history_vehicle.location_y
                time_difference = unite_time - estimated_send_time
                add_x = speed_x * time_difference
                add_y = speed_y * time_difference
                vehicle.location_x = vehicle.location_x + add_x
                vehicle.location_y = vehicle.location_y + add_y

    def fix_packet_loss(self):
        if len(self.history_record) == 0:
            pass
        else:
            if len(self.vehicles) < len(self.history_record[-1]):
                add_vehicle_id = self.get_all_history_record_id() - self.get_all_vehicles_id()
                for id in add_vehicle_id:
                    self.selected_vehicles.append(self.get_vehicle_form_history_record(id))

    def get_all_vehicles_id(self):
        vehicles_ID = set()
        for vehicle in self.vehicles:
            vehicles_ID.add(vehicle.vehicleID)
        return vehicles_ID

    def get_all_history_record_id(self):
        history_vehicles_ID = set()
        if len(self.history_record[-1]):
            for vehicle in self.history_record[-1]:
                history_vehicles_ID.add(vehicle.vehicleID)
        return history_vehicles_ID

    def get_vehicle_form_history_record(self, vehicleID):
        return_vehicle = None
        for vehicle in self.history_record[-1]:
            if vehicleID == vehicle.vehicleID:
                return_vehicle = vehicle
        return return_vehicle

    def get_trace_from_history_record(self, vehicleID):
        trace = []
        for i in range(len(self.history_record)):
            for vehicle in self.history_record[i]:
                if vehicleID == vehicle.vehicleID:
                    trace.append(vehicle)
        return trace

    '''
    :return a result matrix, which contains collision warning messages,
    namely, id of vehicles, which are going to collision
    '''
    def prediction(self):
        self.hmm_model.set_prediction_seconds(self.prediction_time)
        '''Prediction'''
        for vehicle in self.selected_vehicles:
            id = vehicle.vehicleID
            origin_trace = self.get_trace_from_history_record(id).append(vehicle)
            self.hmm_model.set_origin_trace(origin_trace)
            self.prediction_traces.append(self.hmm_model.get_prediction_trace())
        '''Judge'''
        selected_vehicle_number = len(self.selected_vehicles)
        self.selected_vehicle_collision_time_matrix = np.zeros(selected_vehicle_number, selected_vehicle_number)
        for i in range(selected_vehicle_number - 1):
            for j in range(i, selected_vehicle_number):
                collision_time = self.get_collision_time(self.prediction_traces[i], self.prediction_traces[j])
                if collision_time == 0:
                    pass
                else:   # it get collision
                    the_headway = collision_time - self.unite_time
                    if the_headway < 0:
                        print("Error: The headway < 0")
                    else:
                        if the_headway < self.headway:
                            if self.selected_vehicles[i].vehicleID in self.collision_warning_messages:
                                pass
                            else:
                                self.collision_warning_messages.append(self.selected_vehicles[i].vehicleID)
                            if self.selected_vehicles[j].vehicleID in self.collision_warning_messages:
                                pass
                            else:
                                self.collision_warning_messages.append(self.selected_vehicles[j].vehicleID)

    def get_collision_time(self, trace_one, trace_two):
        collision_time = 0

        one_max_time = trace_one[-1].time
        one_min_time = trace_one[0].time
        one_time = set(range(one_min_time, one_max_time + 1))
        two_max_time = trace_two[-1].time
        two_min_time = trace_two[0].time
        two_time = set(range(two_min_time, two_max_time + 1))

        intersection_time = one_time & two_time
        if len(intersection_time):
            for time in range(min(intersection_time), max(intersection_time) + 1):
                one_xy = self.get_xy_from_trace(time=time, trace=trace_one)
                two_xy = self.get_xy_from_trace(time=time, trace=trace_two)
                if one_xy is not None:
                    if two_xy is not None:
                        distance = np.sqrt(np.square(one_xy[0] - two_xy[0]) + np.square(one_xy[1] - two_xy[1]))
                        if distance <= self.collision_distance:
                            collision_time = time
        return collision_time

    def get_xy_from_trace(self, time, trace):
        xy = None
        for vehicle in trace:
            if time == vehicle.time:
                xy = []
                xy.append(vehicle.location_x)
                xy.append(vehicle.location_y)
        return xy

    def get_scenario_xy(self, number):
        dic_scenario_xy = {
            '1': [5241.17, 14185.2],
            '2': [6097.06, 14870.0],
            '3': [6581.00, 26107.6],
            '4': [7653.15, 19486.6],
            '5': [9447.04, 18721.4],
            '6': [10422.00, 12465.3],
            '7': [10435.80, 17528.2],
            '8': [11227.50, 20388.4],
            '9': [11550.60, 18791.4]
        }
        try:
            return dic_scenario_xy[number]
        except KeyError:
            print("Key Error in get_scenario_xy")