class vehicle:
    vehicleID = 0
    location_x = 0
    location_y = 0
    transmission_delay = 0
    packet_loss = False

    def __init__(self):
        pass

    def __setattr__(self, key, value):
        if key == 'vehicleID':
            self.vehicleID = value
        elif key == 'location_x':
            self.location_x = value


