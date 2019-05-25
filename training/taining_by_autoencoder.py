import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense,Input


AUTOENCODER_TRAIN_CSV = r'E:\NearXu\autoencoder2\train.csv'
AUTOENCODER_TEST_CSV = r'E:\NearXu\autoencoder2\test.csv'


def main():
    # 训练数据集
    df_train = pd.read_csv(AUTOENCODER_TRAIN_CSV)
    train = np.array(df_train)
    print(train.shape)
    # 测试数据集
    df_test = pd.read_csv(AUTOENCODER_TEST_CSV)
    test = np.array(df_test)
    print(test.shape)

    #压缩特征到1维
    encoding_dim = 1

    input_two_dim = Input(shape=(2,))

    # 编码层
    encoded = Dense(64, activation='relu')(input_two_dim)
    encoded = Dense(10, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # 解码层
    decoded = Dense(10, activation='tanh')(encoder_output)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(2, activation='tanh')(decoded)

    # 构建自编码器
    autoencoder = Model(inputs= input_two_dim, outputs= decoded)

    # # 构建编码模型
    # encoder = Model(inputs= input_two_dim, outputs= encoded_output)
    #
    # # 构建解码模型
    # decoder = Model(inputs= encoded_output, outputs= decoded)

    # 编译模型
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(train, train, epochs=200, batch_size=256, shuffle=True, validation_data=(test, test))



if __name__ == '__main__':
    main()