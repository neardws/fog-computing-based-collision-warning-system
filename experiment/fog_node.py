import numpy as np
from transmission_model import transmission_model
from hmm_model import predict
class fog_node:
    def __init__(self, x, y, range):
        self.location_x = x
        self.location_y = y
        self.communication_range = range
        self.unite_time = None
        self.vehicles = None
        self.selected_vehicles = []
        self.history_record = []

    def receive_packets(self, vehicles):
        self.vehicles = vehicles

    def selected_packet_under_communication_range(self):
        for vehicle in self.vehicles:
            if self.packet_is_under_communication_range(vehicle):
                if not vehicle.packet_loss:
                    self.selected_vehicles.append(vehicle)
        self.history_record.append(self.selected_vehicles)

    def packet_is_under_communication_range(self, vehicle):
        under_communication_range = False
        distance = np.sqrt(np.square(self.location_x - vehicle.location_x) + np.square(self.location_y - vehicle.location_y))
        if distance <= self.communication_range:
            under_communication_range = True
        return under_communication_range

    '''
    Step one: fix packet loss
    Step two: fix transmission delay
    '''
    def form_real_time_view(self):
        self.fix_packet_loss()
        self.fix_transmission_delay(self.unite_time)

    def fix_transmission_delay(self, unite_time):
        for vehicle in self.selected_vehicles:
            receive_time = vehicle.time + vehicle.transmission_delay
            estimated_delay = transmission_model.get_transmission_delay()
            estimated_send_time = receive_time - estimated_delay / 1000
            history_vehicle = self.get_vehicle_form_history_record(vehicle.vehicleID)
            if history_vehicle is not None:
                speed_x = vehicle.location_x - history_vehicle.location_x
                speed_y = vehicle.location_y - history_vehicle.location_y
                time_difference = unite_time - estimated_send_time
                add_x = speed_x * time_difference
                add_y = speed_y * time_difference
                vehicle.location_x = vehicle.location_x + add_x
                vehicle.location_y = vehicle.location_y + add_y

    def fix_packet_loss(self):
        if len(self.history_record) == 0:
            pass
        else:
            if len(self.vehicles) < len(self.history_record[-1]):
                add_vehicle_id = self.get_all_history_record_id() - self.get_all_vehicles_id()
                for id in add_vehicle_id:
                    self.selected_vehicles.append(self.get_vehicle_form_history_record(id))

    def get_all_vehicles_id(self):
        vehicles_ID = set()
        for vehicle in self.vehicles:
            vehicles_ID.add(vehicle.vehicleID)
        return vehicles_ID

    def get_all_history_record_id(self):
        history_vehicles_ID = set()
        if len(self.history_record[-1]):
            for vehicle in self.history_record[-1]:
                history_vehicles_ID.add(vehicle.vehicleID)
        return history_vehicles_ID

    def get_vehicle_form_history_record(self, vehicleID):
        return_vehicle = None
        for vehicle in self.history_record[-1]:
            if vehicleID == vehicle.vehicleID:
                return_vehicle = vehicle
        return return_vehicle

    def prediction(self):
