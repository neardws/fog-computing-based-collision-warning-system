class result_save:
    def __init__(self, type,  start_time, scenario, packet_loss_rate, headway):
        self.start_time = start_time
        self.scenario = scenario
        self.packet_loss_rate = packet_loss_rate
        self.headway = headway
        self.type = type
        self.RESULT_PATH = r'E:\NearXu\result\dr_result'
        self.result_name = self.RESULT_PATH + '_type_' + str(self.type) + '_time_' + str(self.start_time) + '_scenario_' + str(
            self.scenario) + '_plr_' \
                             '' + str(self.packet_loss_rate) + '_headway_' + str(self.headway) + '.txt'

    def write(self, string):
        with open(self.result_name, 'a+') as file:
            file.writelines(string)
            file.writelines('\n')