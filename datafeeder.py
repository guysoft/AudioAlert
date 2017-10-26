import numpy as np
import os
import glob
from pydub import AudioSegment
import random
from joblib import Parallel, delayed

class Feeder():

    def get_new_item(self, wav_path):
        negative_file = os.path.join(self.negative_folder, random.choice(self.negative_folder_list))
        item = [[wav_path, negative_file], wav_path, [os.path.basename(os.path.dirname(wav_path))]]
        return item

    def __init__(self, folder):
        self.classes_set = ['glass', 'gunshot', 'scream', 'negative']
        self.sounds_folder = folder
        self.sounds = []

        self.negative_folder = os.path.join(folder, "negative")
        self.negative_folder_list = os.listdir(self.negative_folder)

        for class_folder in glob.glob(os.path.join(self.sounds_folder, "*")):
            print(class_folder)
            self.sounds = Parallel(n_jobs=16, backend="threading")(delayed(self.get_new_item)(wav_path)
                                                          for wav_path in glob.glob(os.path.join(class_folder, "*.wav")))

            # old code without joblib
            # for wav_path in glob.glob(os.path.join(class_folder, "*.wav")):
            #     self.sounds.append(item)

    def next(self, batch_size, window_lengh):
        random.shuffle(self.sounds)
        merged_sounds = []
        pure_sound = []
        cls = []

        for i in range(batch_size):
            wav_path = self.sounds[i][0][0]
            negative_file = self.sounds[i][0][1]

            sound = AudioSegment.from_file(wav_path)

            negative_sound = AudioSegment.from_file(negative_file)[:] - (60 - 60 * random.uniform(0.1, 1))
            merge_sound = np.array(sound.overlay(negative_sound).get_array_of_samples())


            merged_sounds.append(merge_sound)
            pure_sound.append(sound)


            cls.append(self.sounds[i][2])
        x = np.zeros((batch_size, len(self.classes_set)))
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

    a.next( 2, 23)


