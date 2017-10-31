import csv
import glob
import os
import random

import numpy as np
from joblib import Parallel, delayed
from pydub import AudioSegment


class DataFeeder():
    def __init__(self, datafolder=r'C:\Projects\DataHack\UrbanSound\data'):
        # read all csvs
        self.index = 0
        self.sounds_folder = datafolder
        self.classes_set = ['gunshot', 'negative']
        self.negative = []
        self.positive = []
        for class_folder in glob.glob(os.path.join(self.sounds_folder, "*")):
            print(class_folder)
            if "gun" in class_folder:
                self.positive += Parallel(n_jobs=16, backend="threading")(
                    delayed(self.get_new_item)(os.path.join(class_folder, wav_path))
                    for wav_path in glob.glob(
                        os.path.join(class_folder, "*.csv")))
            else:
                self.negative += Parallel(n_jobs=16, backend="threading")(
                    delayed(self.get_new_item)(os.path.join(class_folder, wav_path))
                    for wav_path in glob.glob(
                        os.path.join(class_folder, "*.csv")))

    def get_new_item(self, csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            events = list(reader)
        item = [csv_path[:-3] + 'wav', events]
        return item

    @staticmethod
    def mix_waves(a, b, length_sec=1, rate=40000):
        p1 = DataFeeder.correct_file_name(a)
        p2 = DataFeeder.correct_file_name(b)
        s1 = AudioSegment.from_file(p1)
        s2 = AudioSegment.from_file(p2)
        out_vec = np.zeros(int(length_sec * rate))
        rate1 = int(rate * (random.uniform(0.9, 1.1)))
        rate2 = int(rate * (random.uniform(0.9, 1.1)))
        start1 = int(length_sec * rate * random.uniform(0, 0.5))
        start2 = int(length_sec * rate * random.uniform(0, 0.5))
        s1.set_frame_rate(rate1)
        s2.set_frame_rate(rate2)
        n = np.random.choice(range(len(a[1])))
        m = np.random.choice(range(len(b[1])))
        d1 = np.array(
            s1.get_sample_slice(int(float(a[1][n][0]) * rate1),
                                int(float(a[1][n][1]) * rate1)).get_array_of_samples())
        d2 = np.array(
            s2.get_sample_slice(int(float(b[1][m][0]) * rate2),
                                int(float(b[1][m][1]) * rate2)).get_array_of_samples())
        d1 = d1 / max(np.abs(d1))
        d2 = d2 / max(np.abs(d2))

        weight = random.uniform(0.2, 1)
        out_vec[start1:min(start1 + len(d1), int(length_sec * rate))] += d1[0:min(len(d1),
                                                                                  length_sec * rate - start1)]
        mix_in = weight * out_vec.copy()
        mix_in[start2:min(start2 + len(d2), int(length_sec * rate))] += (1 - weight) * d2[0:min(len(d2),
                                                                                                 int(
                                                                                                     length_sec * rate) - start2)]

        return out_vec, mix_in

    @staticmethod
    def correct_file_name(a):
        if os.path.exists(a[0]):
            p1 = a[0]
        elif os.path.exists(a[0][:-3] + 'mp3'):
            p1 = a[0][:-3] + 'mp3'
        elif os.path.exists(a[0][:-3] + 'flac'):
            p1 = a[0][:-3] + 'flac'
        elif os.path.exists(a[0][:-3] + 'aiff'):
            p1 = a[0][:-3] + 'aiff'
        elif os.path.exists(a[0][:-3] + 'aif'):
            p1 = a[0][:-3] + 'aif'
        else:
            print(a[0])
            assert "unable to find " + a[0]
        return p1

    def generate_next_set(self, batch_size, window_lengh):
        random.shuffle(self.negative)
        random.shuffle(self.positive)
        merged_sounds = []
        pure_sound = []
        cls = []
        skip = 0
        for i in range(int(min(batch_size / 2, len(self.positive)))):
            while True:
                try:
                    p = self.positive[i + skip]
                    n = self.negative[i + skip]
                    pure_sound_tmp, merged_sounds_tmp = DataFeeder.mix_waves(p, n)
                    break
                except:
                    skip += 1
                    print(skip)

            pure_sound += [pure_sound_tmp]
            merged_sounds += [merged_sounds_tmp]
            cls.append([0, 1])
        for i in range(int(min(batch_size / 2, len(self.negative)))):
            while True:
                try:
                    n1 = self.negative[int(batch_size / 2) + i + skip]
                    n2 = self.negative[batch_size + i + skip]
                    pure_sound_tmp, merged_sounds_tmp = DataFeeder.mix_waves(n1, n2)
                    break
                except:
                    skip += 1
                    print(skip)
            pure_sound += [pure_sound_tmp]
            merged_sounds += [merged_sounds_tmp]
            cls.append([1, 0])

        return merged_sounds, pure_sound, cls

    def generate_next_set_dummy(self, batch_size, window_lenght):
        merged_sounds = np.zeros((batch_size, window_lenght, 1))
        pure_sound = np.zeros((batch_size, window_lenght, 1))
        cls = np.zeros((batch_size, 2))
        return merged_sounds, pure_sound, cls


if __name__ == "__main__":
    feeder = DataFeeder()
