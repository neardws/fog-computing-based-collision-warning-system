import pandas as pd
import matplotlib.pyplot as plot
import multiprocessing as mp

CSV_FILE = r'E:\NearXu\trace\trace_0.csv'
SORTED_CELLULAR_LOCATION_FILE = r'E:\NearXu\visualization\sorted_cellular_location.csv'


def draw_all_cellular_base_stations():
    x_list = []
    y_list = []
    cellular_df = pd.read_csv(SORTED_CELLULAR_LOCATION_FILE, error_bad_lines=False, sep=',')
    print(cellular_df.head(5))
    x = cellular_df['x']
    y = cellular_df['y']
    stations_number = 0
    for i in range(len(x)):
        if x[i] >= 10000 and x[i] <= 15000:
            if y[i] >= 10000 and y[i] <= 20000:
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


if __name__ == '__main__':
    draw_all_cellular_base_stations()