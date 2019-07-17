class vehicle_info:
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

    def get_xy_from_time(self, time):
        for i in range(len(self.get_trace_time())):
            if self.get_trace_time().iloc[i] == time:
                return self.get_trace_x().iloc[i], self.get_trace_y().iloc[i]

if __name__ == '__main__':
    pass