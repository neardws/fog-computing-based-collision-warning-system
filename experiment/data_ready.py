import numpy as np
import pandas as pd
from vehicels_info import vehicle_info
from vehicle import vehicle

np.set_printoptions(threshold=np.inf)

class data_ready:
    def __init__(self, time, scenario, scenario_range, during_time, packet_loss_rate, collision_distance):
        self.time = time
        self.scenario = scenario
        self.scenario_range = scenario_range
        self.during_time = during_time
        self.packet_loss_rate = packet_loss_rate
        self.collision_distance = collision_distance
        self.vehicle_traces = None
        self.vehicle_number = None
        self.collision_time_matrix = None
        self.collision_number = None
        self.collision_message = None
        self.vehicle_id_array = None

    def get_vehicle_trace(self):
        return self.vehicle_traces

    def get_vehicle_number(self):
        return self.vehicle_number

    def get_collision_time_matrix(self):
        return self.collision_time_matrix

    def get_vehicle_id_array(self):
        return self.vehicle_id_array

    def get_collision_number(self):
        return self.collision_number

    def get_collision_message(self):
        return self.collision_message

    def show_detail(self):
        print("-"*64)
        print("Below is the detail of data")
        print("The vehicle number is")
        print(self.get_vehicle_number())
        print("The vehicle_id_array is")
        print(self.get_vehicle_id_array())
        print("The collision_number is")
        print(self.get_collision_number())
        print("The collision message is ")
        print(self.get_collision_message())
        return "-"*64 + '\n' + "Below is the detail of data\nThe vehicle number is " + str(self.get_vehicle_number()) + "\nThe vehicle_id_array is" + str(self.get_vehicle_id_array()) + "\n" \
                "The collision_number is " + str(self.get_collision_number()) + "\nThe collision message is " + str(self.get_collision_message()) + "\n"
    '''
    :return csv_file, which contains vehicles traces
    according to the selected time in a range of 1AM to 11PM during one day
    '''
    def get_csv_file(self, time):
        csv_file = r'E:\NearXu\trace\trace_'
        dic_csv_file = {
            '1am': csv_file + '0.csv',   '2am': csv_file + '0.csv',
            '3am': csv_file + '0.csv',   '4am': csv_file + '0.csv',
            '5am': csv_file + '0.csv',   '6am': csv_file + '0.csv',
            '7am': csv_file + '3.csv',   '8am': csv_file + '7.csv',
            '9am': csv_file + '8.csv',   '10am': csv_file + '9.csv',
            '11am': csv_file + '10.csv', '12am': csv_file + '12.csv',
            '1pm': csv_file + '13.csv',  '2pm': csv_file + '14.csv',
            '3pm': csv_file + '15.csv',  '4pm': csv_file + '17.csv',
            '5pm': csv_file + '20.csv',  '6pm': csv_file + '23.csv',
            '7pm': csv_file + '24.csv',  '8pm': csv_file + '26.csv',
            '9pm': csv_file + '27.csv',  '10pm': csv_file + '27.csv',
            '11pm': csv_file + '28.csv'
        }
        try:
            return dic_csv_file[time]
        except KeyError:
            print("Key Error in get_csv_file")

    '''
    :return start time in format of seconds number
    for example, 1AM is 1h = 60min = 60*60 = 3600s
    '''
    def get_start_time(self, time):
        dic_start_time = {
            '1am': 3600,   '2am': 7200,   '3am': 10800,  '4am': 14400,  '5am': 18000,
            '6am': 21600,  '7am': 25200,  '8am': 28800,  '9am': 32400,  '10am': 36000,
            '11am': 39600, '12am': 43200, '1pm': 46800,  '2pm': 50400,  '3pm': 54000,
            '4pm': 57600,  '5pm': 61200,  '6pm': 64800,  '7pm': 68400,  '8pm': 72000,
            '9pm': 75600,  '10pm': 79200, '11pm': 82800
        }
        try:
            return dic_start_time[time]
        except KeyError:
            print("Key Error in get_start_time")

    '''
    we have selected some intersections, and marked it as locations
    :return scenario location, as x and y coordinate
    '''
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

    '''
    :return list of vehicles_info 
    all vehicles information in selected time, scenario, scenario_range and during time
    '''
    def get_trace(self):
        csv_file = self.get_csv_file(self.time)
        start_time = self.get_start_time(self.time)
        scenario_xy = self.get_scenario_xy(self.scenario)
        scenario_x = scenario_xy[0]
        scenario_y = scenario_xy[1]
        x_min = scenario_x - self.scenario_range
        x_max = scenario_x + self.scenario_range
        y_min = scenario_y - self.scenario_range
        y_max = scenario_y + self.scenario_range
        chunk_size = 100000
        # return variable
        vehicles = []
        for chunk in pd.read_csv(csv_file, error_bad_lines=False, chunksize=chunk_size):
            selected_traces = chunk[(chunk['x'] >= x_min) & (chunk['x'] <= x_max) &
                                    (chunk['y'] >= y_min) & (chunk['y'] <= y_max) &
                                    (chunk['time'] >= start_time) & (chunk['time'] <= start_time + self.during_time)]
            trace_id = selected_traces['traceID'].drop_duplicates()
            for id in trace_id:
                trace = selected_traces[selected_traces['traceID'] == id]
                if len(trace):
                    x = trace['x']
                    y = trace['y']
                    trace_time = trace['time']
                    new_vehicle = vehicle_info()
                    vehicles.append(new_vehicle.set_vehicleID(vehicleID=id).set_trace(x=x, y=y, time=trace_time))
        return vehicles

    '''
    :return collision time 
    if the collision exists, it return collision time, else return zero    
    '''
    def get_collision_time(self, vehicle_one, vehicle_two):
        collision_time = 0

        my_max_time = max(vehicle_one.get_trace_time())
        my_min_time = min(vehicle_one.get_trace_time())
        my_time = set(range(my_min_time, my_max_time + 1))
        # print(my_time)
        another_max_time = max(vehicle_two.get_trace_time())
        another_min_time = min(vehicle_two.get_trace_time())
        another_time = set(range(another_min_time, another_max_time + 1))
        # print(another_time)

        intersection_time = my_time & another_time
        # print(intersection_time)
        if len(intersection_time):
            # print('*' * 64)
            for time in range(min(intersection_time), max(intersection_time) + 1):
                my_xy = vehicle_one.get_xy_from_time(time)
                another_xy = vehicle_two.get_xy_from_time(time)
                if my_xy is not None:
                    if another_xy is not None:
                        my_x = my_xy[0]
                        my_y = my_xy[1]
                        another_x = another_xy[0]
                        another_y = another_xy[1]
                        distance = np.sqrt(np.square(my_x - another_x) + np.square(my_y - another_y))
                        if distance <= self.collision_distance:
                            '''
                            ToDo 标记
                            '''
                            collision_time = time
                            # print(str(distance))

        return collision_time

    '''
    set vehicles traces and collision time matrix
    '''
    def set_vehicle_traces_and_collision_time_matrix_and_vehicle_id_array(self):
        self.vehicle_traces = self.get_trace()
        self.vehicle_number = len(self.vehicle_traces)
        self.collision_time_matrix = np.zeros((self.vehicle_number, self.vehicle_number))
        self.vehicle_id_array = []
        self.collision_number = 0
        self.collision_message = []
        for i in range(self.vehicle_number - 1):
            for j in range(i + 1, self.vehicle_number):
                collision_time = self.get_collision_time(vehicle_one=self.vehicle_traces[i], vehicle_two=self.vehicle_traces[j])
                self.collision_time_matrix[i, j] = collision_time
                if collision_time == 0:
                    pass
                else:
                    self.collision_number += 1
                    self.collision_message.append({'vehicleOne': self.vehicle_traces[i].get_vehicleID(),
                                                   'vehicleTwo': self.vehicle_traces[j].get_vehicleID(),
                                                   'collisionTime': collision_time})
                # print(self.vehicle_traces[i].get_vehicleID())
            # print(type(self.vehicle_traces[i].get_vehicleID()))
            self.vehicle_id_array.append(int(self.vehicle_traces[i].get_vehicleID()))
        self.vehicle_id_array.append(int(self.vehicle_traces[-1].vehicleID)) # get the last one

    def get_collision_traces(self):
        collision_traces = []
        for i in range(self.vehicle_number - 1):
            for j in range(i + 1, self.vehicle_number):
                collision_time = self.get_collision_time(vehicle_one=self.vehicle_traces[i], vehicle_two=self.vehicle_traces[j])
                if collision_time == 0:
                    pass
                else:
                    collision_traces.append({
                        "collisionTime": collision_time,
                        'vehicleOne': self.vehicle_traces[i].get_vehicleID(),
                        'vehicleTwo': self.vehicle_traces[j].get_vehicleID(),
                        'vehicleOneTrace': self.vehicle_traces[i].show_trace_detail(),
                        'vehicleTwoTrace': self.vehicle_traces[j].show_trace_detail()
                    })
        return collision_traces

    '''
    :return list of vehicles in "time" 
    get packets, which are send to fog node
    '''
    def get_send_packets(self, time):
        vehicles = []
        for trace in self.vehicle_traces:
            if trace.get_xy_from_time(time) is not None:
                v = vehicle(packet_loss_rate=self.packet_loss_rate)
                v.set_vehicleID(trace.get_vehicleID())
                v.set_time(time)
                v.set_location(trace.get_xy_from_time(time=time))
                v.set_packet_loss()
                v.set_transmission_delay()
                vehicles.append(v)
        return vehicles