import csv
import glob
import os
import random
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from pydub import AudioSegment


class DataFeeder():
    def __init__(self, datafile=r'C:\Projects\DataHack\UrbanSound\data', path=''):
        # read all csvs
        self.path = path
        self.index = 0
        self.data_file = datafile
        self.scenario_table = pd.read_csv(datafile, '\t', header=None)
        self.classes_set = sorted(self.scenario_table[1].unique().tolist())
        self.file_list = list(zip(self.scenario_table[0], self.scenario_table[1]))

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
        random.shuffle(self.file_list)

        #     p1 = DataFeeder.correct_file_name(a)
        #     p2 = DataFeeder.correct_file_name(b)
        #     s1 = AudioSegment.from_file(p1)

        merged_sounds = []
        pure_sound = []
        cls = []
        skip = 0
        for i in range(int(min(batch_size, len(self.file_list)))):
            while True:
                try:
                    single_a = self.file_list[i + skip]
                    cls_type = single_a[1]
                    s1 = AudioSegment.from_file(os.path.join(self.path, single_a[0]))
                    s1.set_channels(1)
                    # s1.set_frame_rate(20000)
                    track = np.array(s1.get_array_of_samples())
                    pure_sound_tmp = (track / max(abs(track))).tolist()
                    break
                except:
                    skip += 1
                    print(skip)

            pure_sound += [pure_sound_tmp]
            line_cls = [0 for x in self.classes_set]
            line_cls[self.classes_set.index(cls_type)] = 1
            cls.append(line_cls)

        return pure_sound, cls

    def generate_next_set_dummy(self, batch_size, window_lenght):
        merged_sounds = np.zeros((batch_size, window_lenght, 1))
        pure_sound = np.zeros((batch_size, window_lenght, 1))
        cls = np.zeros((batch_size, 2))
        return merged_sounds, pure_sound, cls


#
# @staticmethod
# def mix_waves(a, b, length_sec=1, rate=40000):
#     p1 = DataFeeder.correct_file_name(a)
#     p2 = DataFeeder.correct_file_name(b)
#     s1 = AudioSegment.from_file(p1)
#     s2 = AudioSegment.from_file(p2)
#     out_vec = np.zeros(int(length_sec * rate))
#     rate1 = int(rate * (random.uniform(0.9, 1.1)))
#     rate2 = int(rate * (random.uniform(0.9, 1.1)))
#     start1 = int(length_sec * rate * random.uniform(0, 0.5))
#     start2 = int(length_sec * rate * random.uniform(0, 0.5))
#     s1.set_frame_rate(rate1)
#     s2.set_frame_rate(rate2)
#     s1.set_channels(1)
#     s2.set_channels(1)
#     n = np.random.choice(range(len(a[1])))
#     m = np.random.choice(range(len(b[1])))
#     d1 = np.array(
#         s1.get_sample_slice(int(float(a[1][n][0]) * rate1),
#                             int(float(a[1][n][1]) * rate1)).get_array_of_samples())
#     d2 = np.array(
#         s2.get_sample_slice(int(float(b[1][m][0]) * rate2),
#                             int(float(b[1][m][1]) * rate2)).get_array_of_samples())
#     d1 = d1 / max(np.abs(d1))
#     d2 = d2 / max(np.abs(d2))
#
#     weight = random.uniform(0.2, 1)
#     out_vec[start1:min(start1 + len(d1), int(length_sec * rate))] += d1[0:min(len(d1),
#                                                                               length_sec * rate - start1)]
#     mix_in = weight * out_vec.copy()
#     mix_in[start2:min(start2 + len(d2), int(length_sec * rate))] += (1 - weight) * d2[0:min(len(d2),
#                                                                                             int(
#                                                                                                 length_sec * rate) - start2)]
#
#     return out_vec, mix_in
if __name__ == "__main__":
    feeder = DataFeeder()
