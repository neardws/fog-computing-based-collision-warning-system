class vehicle:
    vehicleId = 0
    trace = []
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
        # for i in range(length):
        #     self.__add_trace(x.iloc[i], y.iloc[i], time.iloc[i])
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
    # def __add_trace(self, trace_x, trace_y, time):
    #     trace_info = {
    #         'trace_x' : trace_x,
    #         'trace_y' : trace_y,
    #         'time'    : time
    #     }
    #     self.trace.append(trace_info)