class vehicle_info:
    def __init__(self):
        self.vehicleID = 0
        self.trace = []

    def get_vehicleID(self):
        return self.vehicleID

    def get_trace(self):
        return self.trace

    def set_vehicleID(self, vehicleID):
        self.vehicleID = vehicleID
        return self

    def show_trace_detail(self):
        return {
            'x' : self.get_trace_x(),
            'y' : self.get_trace_y(),
            'time' : self.get_trace_time()
        }

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
        xy = None
        for i in range(len(self.get_trace_time())):
            if self.get_trace_time().iloc[i] == time:
                xy = []
                xy.append(self.get_trace_x().iloc[i])
                xy.append(self.get_trace_y().iloc[i])
                return xy
        return xy

if __name__ == '__main__':
    pass