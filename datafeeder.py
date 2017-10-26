import numpy as np
import os
import glob
from pydub import AudioSegment
import random

class Feeder():


    def __init__(self, folder):
        self.sounds_folder = folder
        self.sounds = []
        self.classes_set = ['glass', 'gunshot', 'scream','negative']

        self.negative_folder = os.path.join(folder, "negative")

        for class_folder in glob.glob(os.path.join(self.sounds_folder, "*")):
            for wav_path in glob.glob(os.path.join(class_folder, "*.wav")):
                sound = AudioSegment.from_file(wav_path)
                out_sound = np.array(sound.get_array_of_samples())

                negative_file = os.path.join(self.negative_folder, random.choice(os.listdir(self.negative_folder)))
                negative_sound = AudioSegment.from_file(negative_file)[:] - (60 - 60* random.uniform(0.1, 1))

                merge_sound = AudioSegment.from_file(wav_path).overlay(negative_sound)



                item = [merge_sound, out_sound, [os.path.basename(class_folder)]]
                self.sounds.append(item)

    def next(self, batch_size, window_lengh):
        random.shuffle(self.sounds)
        merged_sounds = []
        pure_sound = []
        cls = []

        for i in range(batch_size):
            merged_sounds.append(i[0])
            pure_sound.append(i[1])
            cls.append(i[2])
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
    import argparse
    parser = argparse.ArgumentParser(add_help=True,
                                     description="Testing data feeder")

    parser.add_argument('in_folder', type=str, help='input folder)')

    args = parser.parse_args()
    a = Feeder(args.in_folder)


