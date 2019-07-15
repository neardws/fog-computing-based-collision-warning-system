import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
from vehicels_info import vehicle
import matplotlib.pyplot as plot

def f_1(x, A, B):
    return A * x +  B


def f_2(x, A, B, C):
    return A * x * x + B * x + C


def curve_fitting(scenario_x, scenario_range, trace_x, trace_y):
    plt.figure()
    plt.scatter(trace_x[:], trace_y[:], 10, 'red')

    A, B, C = optimize.curve_fit(f_2, trace_x, trace_y)[0]
    x = np.arange(scenario_x-scenario_range, scenario_x+scenario_range, 0.01)
    y = A * x * x + B * x + C
    plt.plot(x, y, 'green')

    plt.show()


def get_trace(time, scenario, scenario_range, during_time):
    csv_file = get_csv_file(time)
    start_time = get_start_time(time)
    scenario_xy = get_scenario_xy(scenario)
    scenario_x = scenario_xy[0]
    scenario_y = scenario_xy[1]
    x_min = scenario_x - scenario_range
    x_max = scenario_x + scenario_range
    y_min = scenario_y - scenario_range
    y_max = scenario_y + scenario_range
    chunk_size = 100000
    # return variable
    vehicles = []
    for chunk in pd.read_csv(csv_file, error_bad_lines=False, chunksize=chunk_size):
        selected_traces = chunk[(chunk['x'] >= x_min) & (chunk['x'] <= x_max) &
                                (chunk['y'] >= y_min) & (chunk['y'] <= y_max) &
                                (chunk['time'] >= start_time) & (chunk['time'] <= start_time + during_time)]
        trace_id = selected_traces['traceID'].drop_duplicates()
        for id in trace_id:
            trace = selected_traces[selected_traces['traceID'] == id]
            if len(trace):
                x = trace['x']
                y = trace['y']
                trace_time = trace['time']
                new_vehicle = vehicle()
                vehicles.append(new_vehicle.set_vehicleID(vehicleID=trace_id).set_trace(x=x, y=y, time=trace_time))
    return vehicles


# selected_scenario:
# No      Location        Time        Density         Speed
# 1           6            6pm          485           12.448
# 2           6           11pm          94            13.2631
# 3           4            6pm          355           26.5014
# 4           4           11pm          52            28.7100
# 5           3            6pm          119           18.3845
# 6           3           11pm          28            18.2582
def show_trace():
    time = '6pm'
    scenario = '6'
    during_time = 600
    scenario_range = 500
    vehicles = get_trace(time=time, scenario=scenario, scenario_range=scenario_range, during_time=during_time)
    print("vehicle number is " + str(len(vehicles)))
    for i in range(len(vehicles)):
        x = vehicles[i].get_trace_x()
        y = vehicles[i].get_trace_y()
        time = vehicles[i].get_trace_time()
        scenario_x = get_scenario_xy(scenario)[0]
        curve_fitting(scenario_x=scenario_x, scenario_range=scenario_range, trace_x=x, trace_y=y)


def draw_all_scenario():
    for i in range(1,10):
        print(str(i))
        draw_each_cellular_base_station('6pm', str(i), 500)


def draw_each_cellular_base_station(time, scenario, scenario_range):
    scenario_x = get_scenario_xy(scenario)[0]
    scenario_y = get_scenario_xy(scenario)[1]
    x_min = scenario_x - scenario_range
    x_max = scenario_x + scenario_range
    y_min = scenario_y - scenario_range
    y_max = scenario_y + scenario_range
    plot.xlim(x_min, x_max)
    plot.ylim(y_min, y_max)

    chunk_size = 10000
    chunk_number = 0

    for chunk in pd.read_csv(get_csv_file(time), error_bad_lines=False, chunksize=chunk_size):
        trace_id = chunk['traceID'].drop_duplicates()
        for id in trace_id:
            trace = chunk[(chunk['traceID'] == id) & (chunk['x'] >= x_min) & (chunk['x'] <= x_max) & (chunk['y'] >= y_min) & (chunk['y'] <= y_max)]
            if len(trace):
                x = trace['x']
                y = trace['y']
                plot.scatter(x, y, 0.1, 'deeppink')
                print(id)
        chunk_number += 1
        if chunk_number == 200:
            break

    plot.scatter(scenario_x, scenario_y, 10, '#8B0000')
    plot.show()




def get_csv_file(time):
    csv_file = r'E:\NearXu\trace\trace_'
    dic_csv_file = {
        '1am'   :   csv_file + '0.csv',
        '2am'   :   csv_file + '0.csv',
        '3am'   :   csv_file + '0.csv',
        '4am'   :   csv_file + '0.csv',
        '5am'   :   csv_file + '0.csv',
        '6am'   :   csv_file + '0.csv',
        '7am'   :   csv_file + '3.csv',
        '8am'   :   csv_file + '7.csv',
        '9am'   :   csv_file + '8.csv',
        '10am'  :   csv_file + '9.csv',
        '11am'  :   csv_file + '10.csv',
        '12am'  :   csv_file + '12.csv',
        '1pm'   :   csv_file + '13.csv',
        '2pm'   :   csv_file + '14.csv',
        '3pm'   :   csv_file + '15.csv',    # no good
        '4pm'   :   csv_file + '17.csv',
        '5pm'   :   csv_file + '20.csv',
        '6pm'   :   csv_file + '23.csv',
        '7pm'   :   csv_file + '24.csv',    # no good
        '8pm'   :   csv_file + '26.csv',
        '9pm'   :   csv_file + '27.csv',
        '10pm'  :   csv_file + '27.csv',
        '11pm'  :   csv_file + '28.csv'
    }
    try:
        return dic_csv_file[time]
    except KeyError:
        print("Key Error in get_csv_file")


def get_start_time(time):
    dic_start_time = {
        '1am'   :   3600,
        '2am'   :   7200,
        '3am'   :   10800,
        '4am'   :   14400,
        '5am'   :   18000,
        '6am'   :   21600,
        '7am'   :   25200,
        '8am'   :   28800,
        '9am'   :   32400,
        '10am'  :   36000,
        '11am'  :   39600,
        '12am'  :   43200,
        '1pm'   :   46800,
        '2pm'   :   50400,
        '3pm'   :   54000,
        '4pm'   :   57600,
        '5pm'   :   61200,
        '6pm'   :   64800,
        '7pm'   :   68400,
        '8pm'   :   72000,
        '9pm'   :   75600,
        '10pm'  :   79200,
        '11pm'  :   82800
    }
    try:
        return dic_start_time[time]
    except KeyError:
        print("Key Error in get_start_time")


def get_scenario_xy(number):
    dic_scenario_xy = {
        '1' :  [5241.17,14185.2],
        '2' :  [6097.06,14870.0],
        '3' :  [6581.00,26107.6],
        '4' :  [7653.15,19486.6],
        '5' :  [9447.04,18721.4],
        '6' :  [10422.00,12465.3],
        '7' :  [10435.80,17528.2],
        '8' :  [11227.50,20388.4],
        '9' :  [11550.60,18791.4]
    }
    try:
        return dic_scenario_xy[number]
    except KeyError:
        print("Key Error in get_scenario_xy")


def main():
    # show_trace()
    draw_all_scenario()
    # draw_each_cellular_base_station('6pm', '2', 450)

if __name__ == '__main__':
    main()