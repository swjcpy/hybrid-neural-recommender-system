import numpy as np
from keras.utils import Sequence

class Generator(Sequence):
    def __init__(self, user_input, item_input, user_feature, movie_feature, labels , batch_size):
        self.user_input = user_input
        self.item_input = item_input
        self.user_feature = user_feature
        self.movie_feature = movie_feature
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.user_input)) / float(self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        stop = (idx+1) * self.batch_size
        batch_x = [np.array(self.user_input[start:stop]), np.array(self.item_input[start:stop]),
                        np.array(self.user_feature[start:stop]), np.array(self.movie_feature[start:stop])]
        batch_y = np.array(self.labels[start:stop])

        return batch_x, batch_y
