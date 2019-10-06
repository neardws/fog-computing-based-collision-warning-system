from prettytable import PrettyTable
import json
import ast

File_Head = '/Users/near/NearXu/result/different_packet_loss_rate/dr_result'


def get_results(type, start_time, scenario, packet_loss_rate, headway):
    result_name = File_Head + '_type_' + str(type) + '_time_' + str(start_time) + '_scenario_' + str(scenario) + '_plr_' \
                  + str(packet_loss_rate) + '_headway_' + str(headway) + '.txt'
    with open(result_name, 'r+') as file:
        result = file.readline()
        result = result.replace("'", '"')
        result_json = json.loads(result)
        return result_json


def show_information():
    types = ['result_threading_different_scenario_cloud', 'result_threading_different_scenario_fog',
             'result_threading_different_scenario_fog_re', 'result_threading_different_scenario_fog_gvcw']

    default_packet_loss_rate = 3
    default_headway = 3

    default_start_time = '7pm'
    default_scenario = '6'
    different_packet_loss_rate = [0, 1.5, 3, 4.5, 6]

    table = PrettyTable(['time', 'scenario', 'packet_loss_rate', 'type', 'precision', 'recall'])

    for plr in different_packet_loss_rate:
        for type in types:
            time = default_start_time
            scenario = default_scenario
            result_json = get_results(type, time, scenario, plr, default_headway)
            precision = result_json['precision']
            recall = result_json['recall']
            table_row = []
            table_row.append(time)
            table_row.append(scenario)
            table_row.append(plr)
            table_row.append(type)
            table_row.append(precision)
            table_row.append(recall)
            table.add_row(table_row)
    print(table)


if __name__ == '__main__':
    # default_packet_loss_rate = 3
    # default_headway = 3
    # get_results(type='result_threading_different_scenario_fog_gvcw', start_time='7am', scenario='2', packet_loss_rate=default_packet_loss_rate, headway=default_headway)

    show_information()