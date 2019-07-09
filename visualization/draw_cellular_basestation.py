import pandas as pd
import matplotlib.pyplot as plot
import multiprocessing as mp
import numpy as np

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


def show_trace():
    chunk_size = 10000
    for chunk in pd.read_csv(CSV_FILE, error_bad_lines=False, chunksize=chunk_size):
        print(chunk.head(5))


def show_information():
    morning_time = 32400
    night_time = 79200
    during_time = 1800  # half hours
    morning_csv_file = r'E:\NearXu\trace\trace_9.csv'
    night_csv_file = r'E:\NearXu\trace\trace_27.csv'


    cellular_df = pd.read_csv(SELECTED_CELLULAR_LOCATION, error_bad_lines=False, sep=',')
    print(cellular_df.head(5))
    x = cellular_df['x']
    y = cellular_df['y']
    show_scenario_information(CSV_FILE=night_csv_file, cell_x=x[0], cell_y=y[0], start_time=night_time, during_time=during_time)


def show_scenario_information(CSV_FILE, cell_x, cell_y, start_time, during_time):
    # 需要统计的值
    traffic_density = 0
    vehicle_speed = np.array([])
    range = 250
    chunk_size = 10000
    x_min = cell_x - range
    x_max = cell_x + range
    y_min = cell_y - range
    y_max = cell_y + range
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
                vehicle_speed = np.append(vehicle_speed, get_average_speed(length=len(x), x=x, y=y, time=time))
                traffic_density += 1
    print('*'*32)
    print('traffic density is ' + str(traffic_density))
    print('vehicle speed is ' + str(vehicle_speed))


def get_average_speed(length, x, y, time):
    speeds = np.array([])
    for i in range(length-1):
        print(i)
        speed = np.sqrt(np.square(x.iloc[i+1] - x.iloc[i]) + np.square(y.iloc[i+1] - y.iloc[i])) / (time.iloc[i+1] - time.iloc[i])
        speeds = np.append(speeds, speed)
    return speeds.mean()

if __name__ == '__main__':
    # draw_all_cellular_base_stations()
    # show_trace()


    # length = 4
    # x = [1, 2, 3, 4]
    # y = [1,2,3,4]
    # time = [1, 2, 3, 4]
    # print(str(get_average_speed(length, x, y, time)))

    show_information()