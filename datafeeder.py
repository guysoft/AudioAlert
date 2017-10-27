import numpy as np
import os
import glob
from pydub import AudioSegment
import random
from joblib import Parallel, delayed

class Feeder():

    def get_new_item(self, wav_path):
        negative_file = os.path.join(self.negative_folder, random.choice(self.negative_folder_list))
        item = [negative_file, wav_path, [os.path.basename(os.path.dirname(wav_path))]]
        return item

    def __init__(self, folder):
        self.classes_set = ['glass', 'gunshot', 'scream', 'negative']
        self.sounds_folder = folder
        self.sounds = []

        self.negative_folder = os.path.join(folder, "negative")
        self.negative_folder_list = os.listdir(self.negative_folder)
        self.sounds = []
        for class_folder in glob.glob(os.path.join(self.sounds_folder, "*")):
            print(class_folder)
            self.sounds += Parallel(n_jobs=16, backend="threading")(delayed(self.get_new_item)(wav_path)
                                                                    for wav_path in glob.glob(os.path.join(class_folder, "*.wav")))

            # old code without joblib
            # for wav_path in glob.glob(os.path.join(class_folder, "*.wav")):
            #     self.sounds.append(item)

    def next(self, batch_size, window_lengh):
        random.shuffle(self.sounds)
        merged_sounds = []
        pure_sound = []
        cls = []

        for i in range(min(batch_size, len(self.sounds))):
            wav_path = self.sounds[i][1]
            negative_file = self.sounds[i][0]

            sound = AudioSegment.from_file(wav_path)

            negative_sound = AudioSegment.from_file(negative_file)[:] - (60 - 60 * random.uniform(0.1, 1))
            scale = 1 + random.uniform(-0.3, 0.3)
            merge_sound = scale * np.array(
                sound.overlay(negative_sound, times=1 + random.uniform(-0.3, 0.3)).get_array_of_samples())

            tmp = np.zeros(32000)
            if merge_sound.shape[0] < 32000:
                tmp[0:merge_sound.shape[0]] = merge_sound
            else:
                tmp = merge_sound[:32000]
            merged_sounds.append(tmp.tolist())

            tmp = np.zeros(32000)
            s = scale * np.array(sound.get_array_of_samples())
            if s.shape[0] < 32000:
                tmp[0:s.shape[0]] = s
            else:
                tmp = s[:32000]
            pure_sound.append(tmp.tolist())


            cls.append(self.sounds[i][2])
        x = np.zeros((batch_size, len(self.classes_set)))
        for n in range(batch_size):
            x[n, self.classes_set.index(cls[n][0])] = 1



        return np.array(merged_sounds), np.array(pure_sound), x

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

    print(a.next( 2, 23))


