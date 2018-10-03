import csv
import glob
import os
import random

import numpy as np
from pydub import AudioSegment


class DataFeeder():
    def __init__(self, datafolder, exclude_list=[], include_list=[]):
        # read all csvs
        self.index = 0
        self.sounds_folder = datafolder
        self.classes_set = sorted(os.listdir(datafolder))  # ['', 'negative']
        self.words = ['up', 'down', 'silence', 'unknown']
        self.word_dic = {x: self.words.index(x) for x in self.words}
        self.word_dic['_background_noise_'] = self.words.index('silence')
        for x in self.classes_set:
            if not x in self.word_dic.keys():
                self.word_dic[x] = self.words.index('unknown')

        self.negative = [np.zeros(20000)]
        self.positive = []
        for class_folder in glob.glob(os.path.join(self.sounds_folder, "*")):
            print(class_folder)
            if "_background_noise_" in class_folder:
                for wav_path in glob.glob(os.path.join(class_folder, "*.wav")):
                    samples = np.array(AudioSegment.from_file(wav_path).get_array_of_samples())
                    samples = np.nan_to_num(samples)  # max(np.abs(samples)))
                    # print( max(np.abs(samples)))
                    self.negative += [samples]
            else:
                for wav_path in glob.glob(os.path.join(class_folder, "*.wav")):
                    name = os.path.basename(os.path.dirname(wav_path)) + '/' + os.path.basename(wav_path)
                    if ((not name in exclude_list) and len(include_list) == 0) or name in include_list:
                        basename = os.path.basename(class_folder)
                        self.positive += [(basename, wav_path)]
                        # if basename in self.words:
                            # for _ in range(3):
                            # self.positive += [(basename, wav_path)]

    @staticmethod
    def mix_waves(target, background, length_sec=0.25, rate=44100):
        s1 = AudioSegment.from_file(target)
        d1 = DataFeeder.pitchshift(np.array(s1.get_array_of_samples()),snr=128)
        # d1 = d1/ max(np.max(np.abs(d1)),1)
        # print(max(np.abs(d1)))
        d1 = np.nan_to_num(d1)  # max(np.abs(d1)))
        d1 = np.roll(d1,random.randint(-200,200),axis=0)
        start = int(random.uniform(0, len(background) - length_sec * rate))

        weight = random.uniform(0.7, 1)
        background_segment = background[start:int(start + length_sec * rate)]

        out_vec = np.nan_to_num(
            DataFeeder.pitchshift(background_segment, background=True))  # * np.max(np.abs(background_segment))/2**15)
        out_vec = (1 - weight) * out_vec  # / max(np.max(np.abs(out_vec)),1)

        if len(d1)>len(out_vec):
            d1=d1[:len(out_vec)]
        out_vec[:len(d1)] += weight * d1
        # factor =  random.uniform(1/max(abs(out_vec)), 1/3)
        # out_vec = out_vec * factor
        out_vec = np.nan_to_num(out_vec / np.max(np.abs(out_vec)))
        return out_vec

    def generate_next_set(self, batch_size, window_lengh):
        # random.shuffle(self.negative)
        back_num = len(self.negative)
        random.shuffle(self.positive)
        merged_sounds = np.zeros((batch_size, window_lengh))
        cls = np.zeros((batch_size, len(self.words)))
        N = 0
        skip = 0
        empty_samples = 0.3
        for i in range(int(min(batch_size * (1 - empty_samples), len(self.positive)))):
            p = self.positive[i+skip]
            if (random.uniform(0,1)>0.3):
                for _ in range(5):
                    if (self.word_dic[p[0]]==11):
                        skip+=1
                        p = self.positive[i+skip]
            n = self.negative[random.randint(0, back_num - 1)]
            merged_sounds_tmp = DataFeeder.mix_waves(p[1], n)

            merged_sounds[i, :] = merged_sounds_tmp
            cls[i, self.word_dic[p[0]]] = 1
            N = i

        for i in range(int(batch_size * empty_samples)):
            n = self.negative[random.randint(0, back_num - 1)]
            start = int(random.uniform(0, len(n) - window_lengh))
            merged_sounds_tmp = n[start:start + window_lengh]
            vec_n = np.array(merged_sounds_tmp)  / max(np.max(np.abs(merged_sounds_tmp)),1)
            # factor = random.uniform(max([max(abs(vec_n)), 0.001]), 3)

            merged_sounds[N + i + 1, :] = DataFeeder.pitchshift(vec_n)# / factor
            cls[N + i + 1, self.word_dic["_background_noise_"]] = 1

        return np.nan_to_num(merged_sounds), cls

    def generate_next_set_dummy(self, batch_size, window_lenght):
        merged_sounds = np.zeros((batch_size, window_lenght, 1))
        pure_sound = np.zeros((batch_size, window_lenght, 1))
        cls = np.zeros((batch_size, 2))
        return merged_sounds, pure_sound, cls

    @staticmethod
    def speedx(sound_array, factor):
        """ Multiplies the sound's speed by some `factor` """
        indices = np.round(np.arange(0, len(sound_array), factor))
        indices = indices[indices < len(sound_array)].astype(int)
        return sound_array[indices.astype(int)]

    @staticmethod
    def stretch(sound_array, f, window_size, h):
        """ Stretches the sound by a factor `f` """

        phase = np.zeros(window_size)
        hanning_window = np.hanning(window_size)
        result = np.zeros(int(len(sound_array) / f + window_size))

        for i in np.arange(0, len(sound_array) - (window_size + h), h * f).astype(int):
            # two potentially overlapping subarrays
            a1 = sound_array[i: i + window_size]
            a2 = sound_array[i + h: i + window_size + h]

            # resynchronize the second array on the first
            s1 = np.fft.fft(hanning_window * a1)
            s2 = np.fft.fft(hanning_window * a2)
            phase = (phase + np.angle(s2 / s1)) % 2 * np.pi
            a2_rephased = np.real(np.fft.ifft(np.abs(s2) * np.exp(1j * phase)))

            # add to result
            i2 = int(i / f)
            result[i2: i2 + window_size] += (hanning_window * a2_rephased)

        result = ((2 ** (16 - 4)) * result / result.max())  # normalize (16bit)

        return result.astype('int16')

    @staticmethod
    def pitchshift(snd_array, n=0, window_size=2 ** 11, h=2 ** 7, snr=16, background=False):
        """ Changes the pitch of a sound by ``n`` semitones. """
        snr = snr * random.uniform(1, 128)
        amp = np.max(np.abs(snd_array))
        if amp < 0.000000001:
            amp = 1.0
        noise_vec = np.random.normal(0, amp / snr, len(snd_array))
        # result = snd_array + noise_vec
        # return result / np.max(np.abs(result))
        if n == 0:
            n = random.uniform(-3, 3)
            # n = random.uniform(-3, 3)
            if random.uniform(0, 1) <0.3 or background == True:
                result = snd_array + noise_vec
                return result / np.max(np.abs(result))
        factor = 2 ** (1.0 * n / 12.0)
        stretched = DataFeeder.stretch(snd_array + noise_vec, 1.0 / factor, window_size, h)
        res = DataFeeder.speedx(stretched[:], factor)
        result = np.zeros(snd_array.shape)
        result = res[0:len(result)] * np.max(np.abs(snd_array + noise_vec)) / np.max(np.abs(res[0:len(result)]))
        # scale = 1
        # if background == True:
        #     scale = 32
        return result / np.max(np.abs(result))  # / scale




if __name__ == "__main__":

    import numpy as np
    import sounddevice as sd


    feeder = DataFeeder()
    random.seed(2018)
    val_list = [x[0] + '/' +  os.path.basename(x[1]) for x in random.choices(feeder.positive,k=500)]
    import pickle
    pickle.dump(val_list, open("val_set.pkl", 'wb'))
    import pickle

    val_list = pickle.load(open("val_set.pkl", 'rb'))
    DATAFOLDER = r'D:\GIT\AudioAlert\dataset\violin'

    feeder = DataFeeder(datafolder=DATAFOLDER, include_list=val_list)
    fs = 44100
    x = feeder.generate_next_set(20, int(fs*0.25))
    sd.play(x[0][0] / max(x[0][0]), fs)
    sd.play(x[0][6] / max(x[0][9]), fs)
    y1 = feeder.pitchshift(x[0][0], n=0, window_size=2 ** 11, h=2 ** 7, background=False)
    y2 = feeder.pitchshift(np.zeros(16000), n=0, window_size=2 ** 11, h=2 ** 7, background=True)
    y3 = feeder.mix_waves(feeder.positive[0][1], feeder.negative[0])
    # y=y1+y2
    y=y3
    sd.play(y / max(abs(y)), fs)

