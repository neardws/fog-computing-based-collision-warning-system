import pandas as pd
import matplotlib.pyplot as plot
import multiprocessing as mp
import numpy as np
import math
from prettytable import PrettyTable


CSV_FILE = r'E:\NearXu\trace\trace_9.csv'
SORTED_CELLULAR_LOCATION_FILE = r'C:\Users\user4\PycharmProjects\fog-computing-based-collision-warning-system\visualization\sorted_cellular_location.csv'
SELECTED_CELLULAR_LOCATION = r'C:\Users\user4\PycharmProjects\fog-computing-based-collision-warning-system\visualization\selected_cellular_location.csv'


def draw_all_cellular_base_stations():
    x_list = []
    y_list = []
    cellular_df = pd.read_csv(SELECTED_CELLULAR_LOCATION, error_bad_lines=False, sep=',')
    print(cellular_df.head(5))
    x = cellular_df['x']
    y = cellular_df['y']
    stations_number = 0
    for i in range(len(x)):
        # if x[i] >= 10000 and x[i] <= 15000:
        #     if y[i] >= 10000 and y[i] <= 20000:
                x_list.append(x[i])
                y_list.append(y[i])
                stations_number += 1
                # if stations_number == 10:
                #     break
    print("Stations number is")
    print(str(stations_number) + "\n")
    pool = mp.Pool(processes=10)
    jobs = []
    for i in range(stations_number):
        jobs.append(pool.apply_async(draw_each_cellular_base_station, (x[i], y[i])))
    for job in jobs:
        job.get()
    pool.close()


def draw_each_cellular_base_station(cell_x, cell_y):
    range = 250
    x_min = cell_x - range
    x_max = cell_x + range
    y_min = cell_y - range
    y_max = cell_y + range
    plot.xlim(x_min, x_max)
    plot.ylim(y_min, y_max)

    chunk_size = 10000
    chunk_number = 0

    for chunk in pd.read_csv(CSV_FILE, error_bad_lines=False, chunksize=chunk_size):
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

    plot.scatter(cell_x, cell_y, 10, '#8B0000')
    plot.show()


def show_trace(CSV_FILE):
    chunk_size = 1000000
    for chunk in pd.read_csv(CSV_FILE, error_bad_lines=False, chunksize=chunk_size):
        print(chunk.head(1))


def show_information():
    cellular_range = 500
    during_time = 600  # 10 min
    cellular_df = pd.read_csv(SELECTED_CELLULAR_LOCATION, error_bad_lines=False, sep=',')
    times = ('6am',  '7am',  '8am', '9am', '10am',
            '11am', '12am', '1pm', '2pm', '3pm',
            '4pm',  '5pm',  '6pm', '7pm', '8pm',
            '9pm',  '10pm', '11pm')
    x = cellular_df['x']
    y = cellular_df['y']
    table = PrettyTable(['time','type', '1',  '2',  '3', '4', '5', '6', '7', '8', '9'])
    for time in times:
        row_traffic_density = []
        row_vehicle_speed = []
        row_traffic_density.append(time)
        row_traffic_density.append('density')
        row_vehicle_speed.append(time)
        row_vehicle_speed.append('speed')
        for i in range(9):
            values = show_scenario_information(CSV_FILE=get_csv_file(time), cell_x=x[i], cell_y=y[i],
                                      start_time=get_start_time(time), during_time=during_time,
                                      cellular_range=cellular_range)
            row_traffic_density.append(str(values['traffic_density']))
            row_vehicle_speed.append(str(values['vehicle_speed']))
        table.add_row(row_traffic_density)
        table.add_row(row_vehicle_speed)
    print(table)


def show_scenario_information(CSV_FILE, cell_x, cell_y, start_time, during_time, cellular_range):
    # 需要统计的值
    values = {'traffic_density':0, 'vehicle_speed':0}
    traffic_density = 0
    vehicle_speed = np.array([])
    chunk_size = 10000
    x_min = cell_x - cellular_range
    x_max = cell_x + cellular_range
    y_min = cell_y - cellular_range
    y_max = cell_y + cellular_range
    for chunk in pd.read_csv(CSV_FILE, error_bad_lines=False, chunksize=chunk_size):
        selected_traces = chunk[(chunk['x'] >= x_min) & (chunk['x'] <= x_max) &
                                (chunk['y'] >= y_min) & (chunk['y'] <= y_max) &
                                (chunk['time'] >= start_time) & (chunk['time'] <= start_time + during_time)]
        trace_id = selected_traces['traceID'].drop_duplicates()
        for id in trace_id:
            trace = selected_traces[selected_traces['traceID'] == id]
            if len(trace):
                x = trace['x']
                y = trace['y']
                time = trace['time']
                average_speed = get_average_speed(length=len(x), x=x, y=y, time=time)
                if not math.isnan(average_speed):
                    vehicle_speed = np.append(vehicle_speed, average_speed)
                traffic_density += 1
    print('*'*64)
    print('location is ' + str(cell_x) + ' ' + str(cell_y))
    print('start time is ' + str(start_time) + ' during is ' + str(during_time))
    print('traffic density is ' + str(traffic_density))
    print('vehicle speed is ' + str(vehicle_speed))
    print('mean vehicle speed is '+ str(vehicle_speed.mean()))
    values['traffic_density'] = traffic_density
    values['vehicle_speed'] = vehicle_speed.mean()
    return values

def get_average_speed(length, x, y, time):
    speeds = np.array([])
    for i in range(length-1):
        # print(i)
        speed = np.sqrt(np.square(x.iloc[i+1] - x.iloc[i]) + np.square(y.iloc[i+1] - y.iloc[i])) / (time.iloc[i+1] - time.iloc[i])
        speeds = np.append(speeds, speed)
    return speeds.mean()


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


if __name__ == '__main__':
    # draw_all_cellular_base_stations()
    # CSV_FILE = r'E:\NearXu\trace\trace_'
    # for i in range(0, 10):
    #     csv_file = CSV_FILE + str(i) + '.csv'
    #     print('*'*1000)
    #     print('now id is '+ str(i))
    #     show_trace(csv_file)


    # length = 4
    # x = [1, 2, 3, 4]
    # y = [1,2,3,4]
    # time = [1, 2, 3, 4]
    # print(str(get_average_speed(length, x, y, time)))

    show_information()