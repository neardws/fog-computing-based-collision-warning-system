RANDOM_TRAIN_SET = r'E:\NearXu\random_set\train.txt'
AUTOENCODER_CSV = r'E:\NearXu\autoencoder\train.csv'


def main():
    with open(AUTOENCODER_CSV, 'a+', encoding='utf-8') as csv_file:
        with open(RANDOM_TRAIN_SET, 'r+', encoding='utf-8') as train_set:
            for line in train_set:
                points = line.split(' ')
                for point in points:
                    # print(point)
                    xy = point.split(',')
                    # print(xy)
                    # print(len(xy))
                    if(len(xy) == 2):
                        x = xy[0]
                        y = xy[1]
                        try:
                            x = float(x) / 30.
                            y = float(y) / 30.
                            csv_file.writelines(str(x) + ',' + str(y)+"\n")
                        except ValueError:
                            print("ValueError")


if __name__ == '__main__':
    main()
