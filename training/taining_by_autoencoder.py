import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense,Input


AUTOENCODER_TRAIN_CSV = r'E:\NearXu\autoencoder2\train.csv'
AUTOENCODER_TEST_CSV = r'E:\NearXu\autoencoder2\test.csv'

AUTOENCODER_MODEL = r'E:\NearXu\autoencoder2\autoencoder2.h5'
ENCODER_MODEL = r'E:\NearXu\autoencoder2\encoder2.h5'
DECODER_MODEL = r'E:\NearXu\autoencoder2\decoder2.h5'


def main():
    # 训练数据集
    df_train = pd.read_csv(AUTOENCODER_TRAIN_CSV)
    train = np.array(df_train)
    train = train.astype('float32') / 30.
    print(train.shape)
    # 测试数据集
    df_test = pd.read_csv(AUTOENCODER_TEST_CSV)
    test = np.array(df_test)
    test = test.astype('float32') / 30.
    print(test.shape)

    #压缩特征到1维
    encoding_dim = 1

    input_two_dim = Input(shape=(2,))

    # 编码层
    encoded = Dense(encoding_dim * 10, activation='relu')(input_two_dim)
    encoded = Dense(encoding_dim * 64, activation='relu')(encoded)
    encoded = Dense(encoding_dim * 10, activation='relu')(encoded)
    encoded = Dense(encoding_dim)(encoded)

    # 解码层
    decoded = Dense(encoding_dim * 10, activation='relu')(encoded)
    decoded = Dense(encoding_dim * 64, activation='relu')(decoded)
    decoded = Dense(encoding_dim * 10, activation='relu')(decoded)
    decoded = Dense(2, activation='tanh')(decoded)

    # 构建自编码器
    autoencoder = Model(inputs= input_two_dim, outputs= decoded)

    # 构建编码模型
    encoder = Model(inputs= input_two_dim, outputs= encoded)

    # 构建解码模型
    encoder_input = Input(shape=(encoding_dim,))

    decoded_layer1 = autoencoder.layers[-4]
    decoded_layer2 = autoencoder.layers[-3]
    decoded_layer3 = autoencoder.layers[-2]
    decoded_layer4 = autoencoder.layers[-1]

    decoder = Model(inputs= encoder_input, outputs= decoded_layer4(decoded_layer3(decoded_layer2(decoded_layer1(encoder_input)))))

    # 编译模型
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(train, train, epochs=10, batch_size=128, shuffle=True, validation_data=(test, test))

    autoencoder.save(AUTOENCODER_MODEL)
    encoder.save(ENCODER_MODEL)
    decoder.save(DECODER_MODEL)



if __name__ == '__main__':
    main()