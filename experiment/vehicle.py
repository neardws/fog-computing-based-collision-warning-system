from transmission_model import fog_transmission_model
from transmission_model import cloud_transmission_model
class vehicle:
    def __init__(self, packet_loss_rate):
        self.vehicleID = None
        self.time = None
        self.location_x = None
        self.location_y = None
        self.fog_transmission_delay = None
        self.cloud_transmission_delay = None
        self.packet_loss = None
        self.fog_trans_model = fog_transmission_model(packet_loss_rate)
        self.cloud_trans_model = cloud_transmission_model()

    def __len__(self):
        return 1

    def set_transmission_delay(self):
        self.fog_transmission_delay = self.fog_trans_model.get_transmission_delay()
        self.cloud_transmission_delay = self.cloud_trans_model.get_transmission_delay()

    def set_packet_loss(self):
        self.packet_loss = self.fog_trans_model.get_packet_loss()

    def update_packet_loss(self, packet_loss_rate):
        self.fog_trans_model = fog_transmission_model(packet_loss_rate)
        self.packet_loss = self.fog_trans_model.get_packet_loss()

    def set_vehicleID(self, id):
        self.vehicleID = id

    def set_time(self, time):
        self.time = time

    def set_location(self, xy):
        self.location_x = xy[0]
        self.location_y = xy[1]

if __name__ == '__main__':
    v = vehicle(3.08)
    v.set_packet_loss()
    v.set_transmission_delay()
    print(v.packet_loss)
    print(v.fog_transmission_delay)
