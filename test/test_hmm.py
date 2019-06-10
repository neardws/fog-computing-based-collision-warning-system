import pickle

MODEL_PATH = r'E:\NearXu\model\model_5_322.pkl'

model_file = open(MODEL_PATH, 'rb')

model = pickle.load(model_file)

model.predict()