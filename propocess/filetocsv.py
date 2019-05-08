import multiprocessing as mp

ORIGIN_FILE_NAME = '../../koln.tr/koln.tr'
CSV_FILE_NAME = '../../koln.tr/datafiletocsv.csv'


def process(line):
    info = line.split()
    time = info[0]
    id = info[1]
    x_coordinates = info[2]
    y_coordinates = info[3]
    speed = info[4]
    if id.isdigit():
        with open(CSV_FILE_NAME, 'a+', encoding="utf-8") as csvf:
            csvf.writelines(
                str(id) + ',' + str(time) + ',' + str(x_coordinates) + ',' + str(y_coordinates) + ',' + str(speed))
            csvf.writelines('\n')


if __name__ == '__main__':
    # init objects
    pool = mp.Pool(processes=10)
    jobs = []
    #create jobs
    with open(ORIGIN_FILE_NAME) as f:
        for line in f:
            jobs.append( pool.apply_async(process,(line)) )

    #wait for all jobs to finish
    for job in jobs:
        job.get()

    #clean up
    pool.close()
