num = 100

def add_Num():
    global num
    num = num + 1


def main():
    add_Num()
    print(num)


if __name__ == '__main__':
    main()