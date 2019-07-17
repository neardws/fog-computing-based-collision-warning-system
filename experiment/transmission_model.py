import random
'''
transmission_model
used:
get_packet_loss()
get_transmission_delay()
'''
class transmission_model:
    def __init__(self, packet_loss_rate):
        self.packet_loss_rate = packet_loss_rate

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

        pass

if __name__ == '__main__':
    print(random.randint(1, 10000))