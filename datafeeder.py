import numpy as np


class DataFeeder():
    def __init__(self):
        self.index = 0

    def generate_next_set(self, batch_size, window_lenght):
        merged_sounds = np.zeros((batch_size, window_lenght))
        pure_sound = np.zeros((batch_size, window_lenght))
        cls = np.zeros((batch_size, 2))
        return merged_sounds, pure_sound, cls
