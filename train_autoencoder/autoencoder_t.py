from keras.models import load_model
from keras.models import Model
from keras.layers import Input
import numpy as np
import pandas as pd

AUTOENCODER_TEST_CSV = r'E:\NearXu\autoencoder\train_0.csv'

AUTOENCODER_MODEL = r'E:\NearXu\autoencoder\model2\autoencoder.h5'
ENCODER_MODEL = r'E:\NearXu\autoencoder\model2\encoder.h5'
DECODER_MODEL = r'E:\NearXu\autoencoder\model2\decoder.h5'

# encoding_dim = 1
#
# input_two_dim = Input(shape=(2,))
#
# # 构建自编码器
# autoencoder = Model(inputs= input_two_dim, outputs= decoded)
#
# # 构建编码模型
# encoder = Model(inputs= input_two_dim, outputs= encoded)
#
# # 构建解码模型
# encoder_input = Input(shape=(encoding_dim,))
#
# decoded_layer1 = autoencoder.layers[-4]
# decoded_layer2 = autoencoder.layers[-3]
# decoded_layer3 = autoencoder.layers[-2]
# decoded_layer4 = autoencoder.layers[-1]
#
# decoder = Model(inputs= encoder_input, outputs= decoded_layer4(decoded_layer3(decoded_layer2(decoded_layer1(encoder_input)))))




autoencoder = load_model(AUTOENCODER_MODEL)
encoder = load_model(ENCODER_MODEL)
decoder = load_model(DECODER_MODEL)

autoencoder.compile(optimizer='adagrad', loss='mse')

test_df = pd.read_csv(AUTOENCODER_TEST_CSV)
test_input = np.array(test_df)
test_normal = (test_input.astype('float32') + 30.0 ) / 60.0
print(test_input.shape)
predict_auto = autoencoder.predict(test_normal)

for i in range(test_input.shape[0]):
    print(test_input[i])
    print(test_normal[i])
    # print(test_normal[i].shape)
    print(predict_auto[i])
    print(predict_auto[i] * 30.0)
    print('\n')

    # for line in test_file:
    #     test_input = np.array(line)
    #     test_input = test_input.astype('float32') / 30.0
    #     print(test_input.shape)
    #     print('TEST_INPUT:')
    #     print(test_input)
    #     print('AUTOENCODER:')
    #     print(autoencoder.predict(test_input))
