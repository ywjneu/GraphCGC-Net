from utils import *


def load_data(DATA_DIR):
    data_array = np.load('./{}/total_reduce.npy'.format(DATA_DIR), allow_pickle=True)  # ASD866
    train_attr_vecs = []
    return data_array, train_attr_vecs
