import multiprocessing as mp
import os

ORIGIN_FILE_NAME = '../../koln.tr/data.csv'
CSV_FILE_NAME = '../../koln.tr/data_process.csv'

global csvf
csvf = open(CSV_FILE_NAME, 'a', encoding="utf-8")



def process_wrapper(chunkStart, chunkSize):
    with open(ORIGIN_FILE_NAME) as f:
        f.seek(chunkStart)
        lines = f.read(chunkSize).splitlines()
        for line in lines:
            if line.count(',') == 4:
                csvf.writelines(line)
                csvf.writelines('\n')
            else:
                print(line.count(','))


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
    # csv_title = 'vehicleID,time,x_coordinates,y_coordinates,speed'
    # csvf.writelines(csv_title)
    # csvf.writelines('\n')

    for chunkStart, chunkSize in chunkify(ORIGIN_FILE_NAME):
        jobs.append(pool.apply_async(process_wrapper, (chunkStart, chunkSize)))

    # wait for all jobs to finish
    for job in jobs:
        job.get()

    # clean up
    pool.close()


if __name__ == '__main__':
    main()