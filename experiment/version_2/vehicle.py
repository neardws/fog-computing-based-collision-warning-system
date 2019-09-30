from transmission_model import transmission_model
class vehicle:
    def __init__(self):
        self.vehicleID = None
        self.time = None
        self.location_x = None
        self.location_y = None
        self.fog_transmission_delay = None
        self.cloud_transmission_delay = None
        self.packet_loss = None
        self.trans_model = transmission_model()

    def __len__(self):
        return 1

    def set_transmission_delay(self):
        self.trans_model.set_type(0)
        self.fog_transmission_delay = self.trans_model.get_transmission_delay()
        self.trans_model.set_type(1)
        self.cloud_transmission_delay = self.trans_model.get_transmission_delay()

    def set_packet_loss(self, packet_loss_rate):
        self.trans_model.set_packet_loss_rate(packet_loss_rate)
        self.packet_loss = self.trans_model.get_packet_loss()

    def set_vehicleID(self, id):
        self.vehicleID = id

    def set_time(self, time):
        self.time = time

    def set_location(self, xy):
        self.location_x = xy[0]
        self.location_y = xy[1]

    # update the info
    def update_time(self, time):
        self.time = time

    def update_location(self, xy):
        self.location_x = xy[0]
        self.location_y = xy[1]

    def update_packet_loss(self, packet_loss_rate):
        self.trans_model.set_packet_loss_rate(packet_loss_rate)
        self.packet_loss = self.trans_model.get_packet_loss()

    def update_transmission_delay(self):
        self.trans_model.set_type(0)
        self.fog_transmission_delay = self.trans_model.get_transmission_delay()
        self.trans_model.set_type(1)
        self.cloud_transmission_delay = self.trans_model.get_transmission_delay()
#
# if __name__ == '__main__':
#     v = vehicle(3.08)
#     v.set_packet_loss()
#     v.set_transmission_delay()
#     print(v.packet_loss)
#     print(v.fog_transmission_delay)
