def read_file_to_csv(filename, csvfilename):
    csv_title = 'vehicleID,time,x_coordinates,y_coordinates,speed'
    with open(csvfilename, 'a+', encoding="utf-8") as csvf:
        csvf.writelines(csv_title)
        csvf.writelines('\n')
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                info = line.split()
                time = info[0]
                id = info[1]
                x_coordinates = info[2]
                y_coordinates = info[3]
                speed = info[4]
                if id.isdigit():
                    csvf.writelines(str(id) + ',' + str(time) + ',' + str(x_coordinates) + ',' + str(y_coordinates) + ',' + str(speed))
                    csvf.writelines('\n')


# def init_csv(filename):
#
#     return f


# def write_to_csv(f, vehicleID, time, x_coordinates, y_coordinates, speed):
#     f.writelines(str(vehicleID) +',' + str(time) + ','+ str(x_coordinates) + ','+ str(y_coordinates) + ',' + str(speed))
#     f.writelines('\n')