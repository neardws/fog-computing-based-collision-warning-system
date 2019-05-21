import pandas as pd
import multiprocessing as mp
CSV_FILE_PATH = r'E:\NearXu\trace\trace_'
TRAIN_DATA = r'E:\NearXu\train\train_'


def get_status_by_hmmlearn(add_x, add_y):
    ad_x = add_x
    ad_y = add_y
    if add_x > 33:
        ad_x = 33
    if add_x < -33:
        ad_x = 33
    if add_y > 33:
        ad_y = 33
    if add_y < -33:
        ad_y = -33
    status = (ad_y - 1) * 67 + ad_x
    return status


def get_status_by_hohmm(add_x, add_y):
    pass


def get_trace(num):
    print(num)
    with open(TRAIN_DATA + str(int(num)) + '.txt', 'a+', encoding='utf-8') as file:
        csv_file = CSV_FILE_PATH + str(num) + '.csv'
        df = pd.read_csv(csv_file, encoding='utf-8')
        trace_id = df['traceID']
        for id in range(len(trace_id)):
            print(id+1)
            status = []
            trace = df[df['traceID'] == (id + 1)]
            x = trace['x']
            y = trace['y']
            for i in range(len(trace)-1):
                add_x = int(round(x.values[i+1] - x.values[i]))
                add_y = int(round(y.values[i+1] - y.values[i]))
                sta = get_status_by_hmmlearn(add_x=add_x, add_y=add_y)
                status.append(sta)
                print(str(add_x) + ' ' + str(add_y))
            print(status)
            for i in range(len(status)):
                file.write(str(status[i]) + ' ')
            file.write('\n')
            # file.writelines(str(status))


def main():
    # init objects
    pool = mp.Pool(processes=20)
    jobs = []
    #create jobs
    for i in range(29):
        trace_id = []
        trace_id.append(i)
        # print(trace_id[0])
        jobs.append(pool.apply_async(get_trace, trace_id))
    #wait for all jobs to finish
    for job in jobs:
        job.get()
    #clean up
    pool.close()


if __name__ == '__main__':
    main()