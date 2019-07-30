class result_save:
    def __init__(self, start_time, scenario, packet_loss_rate, headway):
        self.start_time = start_time
        self.scenario = scenario
        self.packet_loss_rate = packet_loss_rate
        self.headway = headway
        self.RESULT_PATH = r'E:\NearXu\result\test_result'
        self.result_name = self.RESULT_PATH + '_type_fog_with_' + 'time' + str(self.start_time) + '_scenario_' + str(
            self.scenario) + '_plr_' \
                             '' + str(self.packet_loss_rate) + '_headway_' + str(self.headway) + '.txt'

    def write(self, string):
        with open(self.result_name, 'a+') as file:
            file.writelines(string)
            file.writelines('\n')