from prettytable import PrettyTable
import json
import ast

File_Head = '/Users/near/NearXu/result/dr_result'


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

    jsons_time_and_scenario = []
    time_and_scenario = {"start_time": "7am", "scenario": "2"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "7am", "scenario": "6"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "7am", "scenario": "9"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "8am", "scenario": "2"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "8am", "scenario": "6"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "9am", "scenario": "6"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "12am", "scenario": "5"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "12am", "scenario": "6"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "1pm", "scenario": "6"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "4pm", "scenario": "1"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "4pm", "scenario": "2"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "4pm", "scenario": "7"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "4pm", "scenario": "9"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "5pm", "scenario": "1"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "5pm", "scenario": "2"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "5pm", "scenario": "6"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "5pm", "scenario": "7"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "5pm", "scenario": "9"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "6pm", "scenario": "1"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "6pm", "scenario": "2"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "6pm", "scenario": "6"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "6pm", "scenario": "7"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "6pm", "scenario": "8"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "6pm", "scenario": "9"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "7pm", "scenario": "6"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "8pm", "scenario": "6"}
    jsons_time_and_scenario.append(time_and_scenario)
    time_and_scenario = {"start_time": "10pm", "scenario": "6"}
    jsons_time_and_scenario.append(time_and_scenario)

    table = PrettyTable(['time', 'scenario', 'type', 'precision', 'recall'])
    for json_t_s in jsons_time_and_scenario:
        for type in types:
            time = json_t_s['start_time']
            scenario = json_t_s['scenario']
            result_json = get_results(type, time, scenario, default_packet_loss_rate, default_headway)
            precision = result_json['precision']
            recall = result_json['recall']
            table_row = []
            table_row.append(time)
            table_row.append(scenario)
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