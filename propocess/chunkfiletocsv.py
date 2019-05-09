import multiprocessing as mp
import os

ORIGIN_FILE_NAME = '../../koln.tr/koln.tr'
CSV_FILE_NAME = '../../koln.tr/data.csv'

global csvf
csvf = open(CSV_FILE_NAME, 'a+', encoding="utf-8")

def is_number(n):
    try:
        num = float(n)
        # 检查 "nan"
        is_number = num == num   # 或者使用 `math.isnan(num)`
    except ValueError:
        is_number = False
    return is_number


def not_comma_in(n):
    s = str(n)
    if s.find(',') == -1:
        return True
    else:
        return False


def process_wrapper(chunkStart, chunkSize):
    with open(ORIGIN_FILE_NAME) as f:
        f.seek(chunkStart)
        lines = f.read(chunkSize).splitlines()
        for line in lines:
            info = line.split()
            time = info[0]
            id = info[1]
            x_coordinates = info[2]
            y_coordinates = info[3]
            speed = info[4]
            if is_number(id) and is_number(time) and is_number(x_coordinates) and is_number(y_coordinates) and is_number(speed)\
                    and not_comma_in(id) and not_comma_in(time) and not_comma_in(x_coordinates) and not_comma_in(y_coordinates) and not_comma_in(speed):
                linedata = str(id) + ',' + str(time) + ',' + str(x_coordinates) + ',' + str(y_coordinates) + ',' + str(speed)
                datas = linedata.split(',')
                if 5 == len(datas):
                    csvf.writelines(linedata)
                    csvf.writelines('\n')


def chunkify(fname, size=6138*10240):
    fileEnd = os.path.getsize(fname)
    with open(fname,'rb') as f:
        chunkEnd = f.tell()
        while True:
            chunkStart = chunkEnd
            f.seek(size,1)
            f.readline()
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break


def main():
    # init objects
    pool = mp.Pool(processes=60)
    jobs = []

    # create jobs
    csv_title = 'vehicleID,time,x_coordinates,y_coordinates,speed'
    csvf.writelines(csv_title)
    csvf.writelines('\n')

    for chunkStart, chunkSize in chunkify(ORIGIN_FILE_NAME):
        jobs.append(pool.apply_async(process_wrapper, (chunkStart, chunkSize)))

    # wait for all jobs to finish
    for job in jobs:
        job.get()

    # clean up
    pool.close()


if __name__ == '__main__':
    main()