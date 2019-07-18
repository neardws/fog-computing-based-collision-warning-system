from transmission_model import transmission_model
class vehicle:
    def __init__(self, packet_loss_rate):
        self.vehicleID = None
        self.location_x = None
        self.location_y = None
        self.transmission_delay = None
        self.packet_loss = None
        self.trans_model = transmission_model(packet_loss_rate)

    def set_transmission_delay(self):
        self.transmission_delay = self.trans_model.get_transmission_delay()

    def set_packet_loss(self):
        self.packet_loss = self.trans_model.get_packet_loss()

    def set_vehicleID(self, id):
        self.vehicleID = id

    def set_location(self, x, y):
        self.location_x = x
        self.location_y = y

if __name__ == '__main__':
    v = vehicle(3.08)
    v.set_packet_loss()
    v.set_transmission_delay()
    print(v.packet_loss)
    print(v.transmission_delay)
