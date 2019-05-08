#!/usr/bin/python
# -*- coding: UTF-8 -*-
def read_file_to_csv(filename, csvfilename):
    init_csv(csvfilename)
    with open(filename, 'r', encoding='utf-8') as f:
        i = 1
        for line in f:
            process_line(line, i, csvfilename)
            i = i + 1


def process_line(line, i, csvfilename):
    if i <= 50:
        print(line + '\n')
        info = line.split()
        time = info[0]
        id = info[1]
        x_coordinates = info[2]
        y_coordinates = info[3]
        speed = info[4]
        if id.isdigit():
            pass
        else:
            write_to_csv(csvfilename, id, time, x_coordinates, y_coordinates, speed)


def init_csv(filename):
    csv_title = 'vehicleID,time,x_coordinates,y_coordinates,speed'
    with open(filename, 'a+', encoding="utf-8") as f:
        f.writelines(csv_title)
        f.writelines('\n')


def write_to_csv(filename, vehicleID, time, x_coordinates, y_coordinates, speed):
    with open(filename, 'a+', encoding="utf-8") as f:
        f.writelines(str(vehicleID) +',' + str(time) + ','+ str(x_coordinates) + ','+ str(y_coordinates) + ',' + str(speed))
        f.writelines('\n')