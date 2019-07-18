class fog_node:
    def __init__(self, x, y):
        self.location_x = x
        self.location_y = y
        self.vehicles = None
        self.collision_time_matrix = None

    def receive_packets(self, vehicles):
        self.vehicles = vehicles

    def form_real_time_view(self):
        pass

    def prediction_and_statistics(self):
        pass