import numpy as np

class vehicle:
    vehicleId = 0
    trace = []
    trace_time = []
    def __init__(self):
        self.vehicleId = 0
        self.trace = []

    def get_vehicleID(self):
        return self.vehicleId

    def get_trace(self):
        return self.trace

    def set_vehicleID(self, vehicleID):
        self.vehicleId = vehicleID
        return self

    def set_trace(self, x, y, time):
        for i in range(len(time)):
            self.__add_trace(x.iloc[i], y.iloc[i], time.iloc[i])
        self.trace.append(x)
        self.trace.append(y)
        self.trace.append(time)
        return self

    def get_trace_x(self):
        return self.trace[0]

    def get_trace_y(self):
        return self.trace[1]

    def get_trace_time(self):
        return self.trace[2]

    def __add_trace(self, trace_x, trace_y, time):
        trace_info = {
            'trace_x' : trace_x,
            'trace_y' : trace_y,
            'time'    : time
        }
        self.trace_time.append(trace_info)

    def get_distance(self, vehicle):
        my_max_time = max(self.get_trace_time())
        my_min_time = min(self.get_trace_time())
        my_time = set(range(my_min_time, my_max_time + 1))
        print(my_time)
        another_max_time = max(vehicle.get_trace_time())
        another_min_time = min(vehicle.get_trace_time())
        another_time = set(range(another_min_time, another_max_time + 1))
        print(another_time)

        intersection_time = my_time & another_time
        if len(intersection_time):
            for time in range(min(intersection_time), max(intersection_time)+1):
                my_x , my_y = self.get_xy_from_time(time)
                another_x , another_y = vehicle.get_xy_from_time(time)
                distance = np.sqrt(np.square(my_x - another_x) + np.square(my_y - another_y))
                print(str(distance) + " ")

    def get_xy_from_time(self, time):
        for trace_info in self.trace_time:
            if time == trace_info['time']:
                return trace_info['trace_x'], trace_info['trace_y']

if __name__ == '__main__':
    pass