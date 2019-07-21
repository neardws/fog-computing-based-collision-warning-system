import random
from scipy.stats import levy_stable
'''
transmission_model
used:
get_packet_loss()
get_transmission_delay()
'''
class fog_transmission_model:
    def __init__(self, packet_loss_rate):
        self.packet_loss_rate = packet_loss_rate
        self.levy_stable_alpha = 1.77395
        self.levy_stable_beta = 1
        self.levy_stable_loc = 72.7343
        self.levy_stable_scale = 13.3685


    '''
    it is depend on the packet loss rate, which is given by transmission model
    for example, packet loss rate is 3.08%
    input 3.08
    :return 
     packet_loss(True or False)
     True when the packet is loss
     False when the packet is not loss
    '''
    def get_packet_loss(self):
        packet_loss = False

        if random.randint(1, 10000) <= (self.packet_loss_rate * 100):
            packet_loss = True

        return packet_loss

    '''
    it is depend on delay model, which is described as an distribution model, like normal distribution or stable distribution
    for example, delay model is an stable distribution, which has some var
    :return
     packet transmission delay(ms)
    '''
    def get_transmission_delay(self):

        transmission_delay = levy_stable.rvs(alpha=self.levy_stable_alpha,
                                             beta=self.levy_stable_beta,
                                             loc=self.levy_stable_loc,
                                             scale=self.levy_stable_scale)

        return transmission_delay

class cloud_transmission_model:
    def __init__(self):
        self.mean_transmission_delay = 120

    def get_transmission_delay(self):
        return self.mean_transmission_delay

if __name__ == '__main__':
    m = fog_transmission_model(packet_loss_rate=3.05)
    for i in range(100):
        print(m.get_packet_loss())
        print(m.get_transmission_delay())