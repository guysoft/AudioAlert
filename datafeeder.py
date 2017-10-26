import numpy as np
import os
import glob
from pydub import AudioSegment
import random

class Feeder():


    def __init__(self, folder):
        self.sounds_folder = folder
        self.sounds = []
        self.classes_set = ['glass', 'gunshot', 'scream']
        for class_folder in glob.glob(os.path.join(self.sounds_folder, "*")):
            for wav_path in glob.glob(os.path.join(class_folder, "*.wav")):
                sound = AudioSegment.from_file(wav_path)
                out_sound = np.array(sound.get_array_of_samples())
                item = [out_sound, out_sound, [os.path.basename(class_folder)]]
                self.sounds.append(item)

    def next(self, batch_size, window_lengh):
        random.shuffle(self.sounds)
        merged_sounds = []
        pure_sound = []
        cls = []

        for i in range(batch_size):
            merged_sounds.append(self.sounds[i][0])
            pure_sound.append(self.sounds[i][1])
            cls.append(self.sounds[i][2])
        x = np.zeros((batch_size, 3))
        for n in range(batch_size):
            x[n, self.classes_set.index(cls[n][0])] = 1

        return merged_sounds, pure_sound, x

class DataFeeder():
    def __init__(self, datafolder):
        self.index = 0
        self.feeder = Feeder(datafolder)

    def generate_next_set(self, batch_size, window_lengh):
        return self.feeder.next(batch_size, window_lengh)

    def generate_next_set_dummy(self, batch_size, window_lenght):
        merged_sounds = np.zeros((batch_size, window_lenght, 1))
        pure_sound = np.zeros((batch_size, window_lenght, 1))
        cls = np.zeros((batch_size, 2))
        return merged_sounds, pure_sound, cls



if __name__ == "__main__":
    a = Feeder()


