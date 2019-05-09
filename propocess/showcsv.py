def read_csv(filename):
    i = 1
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if i <= 500:
                with open("test.csv",'a+',encoding='utf-8') as test:
                    test.writelines(line)
            else:
                break
            i += 1


def is_number(n):
    is_number = True
    try:
        num = float(n)
        # 检查 "nan"
        is_number = num == num   # 或者使用 `math.isnan(num)`
    except ValueError:
        is_number = False
    return is_number


if __name__ == '__main__':
    # CSV_FILE_NAME = '../../koln.tr/data_process.csv'
    # read_csv(CSV_FILE_NAME)
    print(is_number("14198,8008.235322970174,18637.125729230338,14.059999999999999"))
    print(is_number("14198"))
    print(is_number("8008.235322970174"))
    print("1131490,18719,11222.778931546617,24046,14410.85048601347,12915.269128447311,10.08".count(',') == 6)