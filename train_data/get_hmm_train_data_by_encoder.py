# from keras.models import load_model
import numpy as np

RANDOM_TRAIN_SET = r'E:\NearXu\random_set\train.txt'
RANDOM_TEST_SET = r'E:\NearXu\random_set\test.txt'
ENCODER_MODEL = r'E:\NearXu\autoencoder\model3\encoder.h5'
HMM_TRAIN_DATA_PATH = r'E:\NearXu\hmm_train_data\train_'


def main():
    with open(HMM_TRAIN_DATA_PATH + '2dim.txt', 'a+', encoding='utf-8') as hmm_train_file:
        # encoder = load_model(ENCODER_MODEL)
        with open(RANDOM_TRAIN_SET, 'r+', encoding='utf-8') as train_file:
            for line in train_file:
                xys = line.split()
                if len(xys) > 10:
                    for xy in xys:
                        if ',' in xy:
                            try:
                                x_y = xy.split(',')
                                input_data = np.array(x_y).astype('float32')
                                # print(input_data)
                                # input_normal = (input_data + 30.0) / 60.
                                # x_input = np.expand_dims(np.array(input_data), axis=0)
                                # print(x_input)
                                # encode = encoder.predict(x_input)
                                # print(encode[0][0])
                                hmm_train_file.writelines(str(xy) + ' ')
                            except ValueError:
                                print('Value Error')
                    hmm_train_file.writelines('\n')


if __name__ == '__main__':
    main()