import numpy as np
import pandas as pd
import os
import random
import multiprocessing as mp
TRAIN_DATA = r'E:\NearXu\train_data\train_'
AUTOENCODER_TRAIN_PATH_CSV = r'E:\NearXu\autoencoder2\train_'
AUTOENCODER_TRAIN_CSV = r'E:\NearXu\autoencoder2\train.csv'
AUTOENCODER_TEST_CSV = r'E:\NearXu\autoencoder2\test.csv'


# global train_csv_file
# train_csv_file = open(AUTOENCODER_TRAIN_CSV, 'a+', encoding='utf-8')
# global test_csv_file
# test_csv_file = open(AUTOENCODER_TEST_CSV, 'a+', encoding='utf-8')


def merge_file():
    Folder_Path = r'E:\NearXu\autoencoder\train_'  # 要拼接的文件夹及其完整路径，注意不要包含中文
    SaveFile_Path = r'E:\NearXu\autoencoder\train_all'  # 拼接后要保存的文件路径
    SaveFile_Name = r'all.csv'  # 合并后要保存的文件名

    # 修改当前工作目录
    os.chdir(Folder_Path)
    # 将该文件夹下的所有文件名存入一个列表
    file_list = os.listdir()

    # 读取第一个CSV文件并包含表头
    df = pd.read_csv(Folder_Path + '\\' + file_list[0])  # 编码默认UTF-8，若乱码自行更改

    # 将读取的第一个CSV文件写入合并后的文件保存
    df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False)

    # 循环遍历列表中各个CSV文件名，并追加到合并后的文件
    for i in range(1, len(file_list)):
        df = pd.read_csv(Folder_Path + '\\' + file_list[i])
        df.to_csv(SaveFile_Path + '\\' + SaveFile_Name, encoding="utf_8_sig", index=False, header=False, mode='a+')


def process(trace_id):
    #print(trace_id)
    with open(AUTOENCODER_TRAIN_PATH_CSV + str(int(trace_id)) + '.csv', 'a+', encoding='utf-8') as csv_file:
        with open(TRAIN_DATA + str(int(trace_id)) + '.txt', 'r+', encoding='utf-8') as file:
            for line in file:
                points = line.split(' ')
                # print(len(points))
                # random_number = np.random.randint(len(points))
                # print(random_number)
                # print(points[random_number])
                for point in points:
                    xy = point.split(',')
                    # print(xy)
                    # print(len(xy))
                    if (len(xy) == 2):
                        x = xy[0]
                        y = xy[1]
                        try:
                            float(x)
                            float(y)
                            csv_str = str(x) + ',' + str(y) + '\n'
                            if csv_str.count(',') > 1:
                                print('count > 1')
                            else:
                                csv_file.writelines(csv_str)
                        except ValueError:
                            print("ValueError")
                # xy = points[random_number].split(',')
                # # print(xy)
                # # print(len(xy))
                # if (len(xy) == 2):
                #     x = xy[0]
                #     y = xy[1]
                #     try:
                #         float(x)
                #         float(y)
                #         csv_str = str(x) + ',' + str(y) + '\n'
                #         if csv_str.count(',') > 1:
                #             print('count > 1')
                #         else:
                #             csv_file.writelines(csv_str)
                #     except ValueError:
                #         print("ValueError")


def process_random(trace_id):
    with open(AUTOENCODER_TRAIN_PATH_CSV + str(int(trace_id)) + '.csv', 'r+', encoding='utf-8') as csv_file:
        print(trace_id)
        lines = csv_file.readlines()
        lines_number = len(lines)
        print(lines_number)
        random_list = range(0, lines_number)
        random_train_id = random.sample(random_list, int(0.5 * lines_number))
        random_test_id = random.sample(random_list, int(0.1 * lines_number))
        for id in random_train_id:
            train_csv_file.writelines(lines[id])
        for id in random_test_id:
            test_csv_file.writelines(lines[id])


def main():
    # pool = mp.Pool(processes=1)
    # jobs = []
    # for i in range(29):
    #     trace_id = []
    #     trace_id.append(i)
    #     jobs.append(pool.apply_async(process_random, trace_id))
    # for job in jobs:
    #     job.get()
    # pool.close()

    merge_file()


if __name__ == '__main__':
    main()
