from readfiletocsv import read_file_to_csv


def main():
    ORIGIN_FILE_NAME = '../../koln.tr/koln.tr'
    CSV_FILE_NAME = '../../koln.tr/data.csv'
    read_file_to_csv(ORIGIN_FILE_NAME, CSV_FILE_NAME)


if __name__ == '__main__':
    main()