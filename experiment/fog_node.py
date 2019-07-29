import numpy as np
from transmission_model import fog_transmission_model
class fog_node:
    def __init__(self, scenario, range, hmm_model, prediction_time, collision_distance):
        self.location_x = self.get_scenario_xy(scenario)[0]
        self.location_y = self.get_scenario_xy(scenario)[1]
        self.communication_range = range
        self.hmm_model = hmm_model
        self.prediction_time = prediction_time
        # self.collision_distance = collision_distance
        self.collision_distance = collision_distance
        self.fog_transmission_model = fog_transmission_model(0) # never mind the packet loss rate
        self.headway = None
        self.unite_time = None
        self.receive_vehicle = None
        self.selected_vehicles = []
        self.history_record = []
        self.prediction_traces = []
        self.collision_warning_messages = []

    def re_init(self):
        self.headway = None
        self.unite_time = None
        self.receive_vehicle = None
        self.selected_vehicles = []
        self.prediction_traces = []
        self.collision_warning_messages = []

    def set_headway(self, headway):
        self.headway = headway

    def set_unite_time(self, time):
        self.unite_time = time

    def set_prediction_time(self, time):
        self.prediction_time = time

    def receive_packets(self, vehicles):
        self.receive_vehicle = vehicles

    def get_selected_vehicle_id(self):
        selected_vehicle_id = []
        for vehicle in self.selected_vehicles:
            selected_vehicle_id.append(vehicle.vehicleID)
        return selected_vehicle_id

    def get_collision_warning_messages(self):
        return self.collision_warning_messages

    def selected_packet_under_communication_range(self):
        for vehicle in self.receive_vehicle:
            if self.packet_is_under_communication_range(vehicle):
                if not vehicle.packet_loss:
                    self.selected_vehicles.append(vehicle)
        # self.history_record.append(self.selected_vehicles)

    def packet_is_under_communication_range(self, vehicle):
        under_communication_range = False
        distance = np.sqrt(np.square(self.location_x - vehicle.location_x) + np.square(self.location_y - vehicle.location_y))
        if distance <= self.communication_range:
            under_communication_range = True
        return under_communication_range

    def form_cloud_not_real_time_view(self):
        for vehicle in self.selected_vehicles:
            receive_time = vehicle.time + vehicle.cloud_transmission_delay / 1000
            history_vehicle = self.get_vehicle_form_history_record(vehicle.vehicleID)
            if history_vehicle is not None:
                speed_x = vehicle.location_x - history_vehicle.location_x
                speed_y = vehicle.location_y - history_vehicle.location_y
                time_difference = self.unite_time - receive_time
                add_x = speed_x * time_difference
                add_y = speed_y * time_difference
                vehicle.location_x = vehicle.location_x + add_x
                vehicle.location_y = vehicle.location_y + add_y

    def form_fog_not_real_time_view(self):
        for vehicle in self.selected_vehicles:
            receive_time = vehicle.time + vehicle.fog_transmission_delay / 1000
            history_vehicle = self.get_vehicle_form_history_record(vehicle.vehicleID)
            if history_vehicle is not None:
                speed_x = vehicle.location_x - history_vehicle.location_x
                speed_y = vehicle.location_y - history_vehicle.location_y
                time_difference = self.unite_time - receive_time
                add_x = speed_x * time_difference
                add_y = speed_y * time_difference
                vehicle.location_x = vehicle.location_x + add_x
                vehicle.location_y = vehicle.location_y + add_y

    '''
    Step one: fix packet loss
    Step two: fix transmission delay
    '''
    def form_fog_real_time_view(self):
        self.fix_packet_loss()
        self.fix_fog_transmission_delay(self.unite_time)

    def fix_fog_transmission_delay(self, unite_time):
        for vehicle in self.selected_vehicles:
            receive_time = vehicle.time + vehicle.fog_transmission_delay / 1000
            estimated_delay = self.fog_transmission_model.get_transmission_delay()
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
            if len(self.receive_vehicle) < len(self.history_record[-1]):
                add_vehicle_id = self.get_all_history_record_id() - self.get_all_vehicles_id()
                for id in add_vehicle_id:
                    self.selected_vehicles.append(self.get_vehicle_form_history_record(id))

    def get_all_vehicles_id(self):
        vehicles_ID = set()
        for vehicle in self.receive_vehicle:
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
        if len(self.history_record) == 0:
            pass
        else:
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
        add_history_record = []
        for vehicle in self.selected_vehicles:
            id = vehicle.vehicleID
            origin_trace = self.get_trace_from_history_record(id)
            # print("len(origin_trace)")
            # print(len(origin_trace))
            if len(origin_trace) == 0:
                add_history_record.append(vehicle)
            else:
                origin_trace.append(vehicle)
                # print("Origin trace:")
                # print(origin_trace)
                self.hmm_model.set_origin_trace(origin_trace)
                prediction_trace = self.hmm_model.get_prediction_trace()
                # print("prediction trace is")
                # print(prediction_trace)
                if prediction_trace is not None:
                    # print("prediction_trace is NOTNONE in fog node")
                    self.prediction_traces.append(prediction_trace)
                    # print(self.prediction_traces)
                # else:
        #         #     print("prediction_trace is None in fog node")
        # print("history_record length is")
        # print(len(self.history_record))
        self.history_record.append(add_history_record)
        '''Judge'''
        prediction_traces_number = len(self.prediction_traces)
        # print("prediction_traces_number is")
        # print(prediction_traces_number)
        # for i in range(prediction_traces_number):
        #     print(self.prediction_traces[i])

        for i in range(prediction_traces_number - 1):
            for j in range(i, prediction_traces_number):
                collision_time = self.get_collision_time(self.prediction_traces[i], self.prediction_traces[j])
                if collision_time == 0:
                    pass
                else:  # it get collision
                    the_headway = collision_time - self.unite_time
                    if the_headway < 0:
                        # print("Error: The headway < 0")
                        pass
                    else:
                        if the_headway < self.headway:
                            if self.prediction_traces[i][0].vehicleID in self.collision_warning_messages:
                                pass
                            else:
                                self.collision_warning_messages.append(self.prediction_traces[i][0].vehicleID)
                            if self.prediction_traces[j][0].vehicleID in self.collision_warning_messages:
                                pass
                            else:
                                self.collision_warning_messages.append(self.prediction_traces[j][0].vehicleID)


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